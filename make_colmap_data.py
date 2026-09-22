from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.calibration import distort_by_calibration
import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import (
    get_nearest_pose,
    filter_points_from_confidence,
)


from collections import defaultdict
from typing import Literal
import cv2
import numpy as np
import os
from PIL import Image
from fractions import Fraction
import json


def load_vrs(vrs_file: str) -> VrsDataProvider:
    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"
    return provider


def get_camera_params(
    provider,
    camera_label: Literal["camera-slam-left", "camera-slam-right", "camera-rgb"],
):
    device_calib = provider.get_device_calibration()
    sensor_calib = device_calib.get_camera_calib(camera_label)

    return sensor_calib


# def get_camera_extrinsics(
#    provider,
#    camera_label: Literal["camera-slam-left", "camera-slam-right", "camera-rgb"],
# ):
#    device_calib = provider.get_device_calibration()
#    transform_device_sensor = device_calib.get_transform_device_sensor(camera_label)
#    return transform_device_sensor


def get_undistorted_params(orig_calibration, rotate_90=True, crop_pixels=150, out_resolution=None):
    input_calib_width = orig_calibration.get_image_size()[0]
    input_calib_height = orig_calibration.get_image_size()[1]
    input_calib_focal = orig_calibration.get_focal_lengths()[0]
    if out_resolution is not None:
        input_calib_width = input_calib_height = out_resolution

    # Undistort the image
    pinhole = calibration.get_linear_camera_calibration(
        int(input_calib_width - 2 * crop_pixels),
        int(input_calib_height - 2 * crop_pixels),
        input_calib_focal,
        "camera-rgb-pinhole",
        orig_calibration.get_transform_device_camera(),
    )

    if rotate_90:
        pinhole = calibration.rotate_camera_calib_cw90deg(pinhole)
    return pinhole


def pinhole_params_to_str(camera_dict):
    """
    Creates a string record for cameras.txt file based on undistorted pinhole camera model
    """
    params = (
        [camera_dict["id"], "PINHOLE"]
        + camera_dict["undistorted_params"].get_image_size().tolist()[::-1]
        + camera_dict["undistorted_params"].get_projection_params().tolist()
    )
    params = [str(x) for x in params]
    return " ".join(params)


def undistort_image(img_numpy, camera_dict, rotate_90=True):
    orig_calib = camera_dict["distorted_params"]
    updated_calib = camera_dict["undistorted_params"]

    undistorted = distort_by_calibration(img_numpy, updated_calib, orig_calib)
    if rotate_90:
        return np.rot90(undistorted, k=3)
    return undistorted


def default_camera_stream(provider, camera_dict):
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(camera_dict["stream_id"])
    deliver_option.set_subsample_rate(
        camera_dict["stream_id"], camera_dict["downsample_rate"]
    )
    yield from provider.deliver_queued_sensor_data(deliver_option)


def load_images(provider, output_dir_parent, camera_dict):
    output_dir = output_dir_parent + "/images"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        if os.listdir(output_dir):
            print(
                f"WARNING: image path {output_dir} exists and is not empty. The data may be inconsistent!"
            )

    imgs = []
    imgs_numpy = []
    for data in default_camera_stream(provider, camera_dict):
        img_timestamp = data.get_time_ns(TimeDomain.DEVICE_TIME)
        print("Processing timestamp: ", img_timestamp)
        image_data = data.image_data_and_record()

        img = image_data[0].to_numpy_array()
        imgs_numpy.append(img)

        filename = f"{img_timestamp}_{camera_dict['camera_label']}.jpg"
        output_path = f"{output_dir}/{filename}"

        exposure = image_data[1].exposure_duration
        gain = image_data[1].gain
        temperature = image_data[1].temperature

        imgs.append(
            {
                "id": f"{img_timestamp}_{camera_dict['camera_label']}",
                "timestamp": img_timestamp,
                "path": output_path,
                "file": filename,
                "camera_id": camera_dict["id"],
                "exposure": exposure,
                "gain": gain,
                "temperature": temperature
            }
        )

    return imgs, imgs_numpy


def save_image(undistorted_img, image_data, output_path):
    # extract capture metadata
    exposure = image_data[1].exposure_duration
    gain = image_data[1].gain
    temperature = image_data[1].temperature


    # Try to write EXIF using piexif + Pillow. If unavailable, write image with OpenCV
    # and save a sidecar JSON with the same metadata.
    try:
        import piexif
        from piexif.helper import UserComment

        # prepare PIL image (convert BGR->RGB for color)
        if undistorted_img.ndim == 2:
            pil_img = Image.fromarray(undistorted_img)
        else:
            rgb = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # If we have a reasonable exposure (0 < t < 30s), store as ExposureTime rational
        if exposure is not None and exposure > 0 and exposure < 30:
            frac = Fraction(exposure).limit_denominator(1000000)
            exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (frac.numerator, frac.denominator)

        # Put full metadata into UserComment (JSON encoded)
        meta = {"exposure": exposure, "gain": gain, "temperature": temperature}
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = UserComment.dump(json.dumps(meta))

        exif_bytes = piexif.dump(exif_dict)
        pil_img.save(output_path, "jpeg", exif=exif_bytes)
    except ImportError:
        print("piexif could not be imported. Saving images without EXIF tags")
        # fallback: write image with OpenCV and save sidecar JSON
        cv2.imwrite(output_path, undistorted_img)


def undistort_images(provider, output_dir_parent, subfolder, camera_dict):
    output_dir = output_dir_parent + "/images"
    if subfolder.strip() != "":
        output_dir = output_dir + "/" + subfolder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if os.listdir(output_dir):
            print(
                f"WARNING: image path {output_dir} exists and is not empty. The data may be inconsistent!"
            )

    imgs = []
    for data in default_camera_stream(provider, camera_dict):
        img_timestamp = data.get_time_ns(TimeDomain.DEVICE_TIME)
        print("Timestamp: ", img_timestamp)
        image_data = data.image_data_and_record()

        # .copy() can be necessary, because undistortion uses the raw bytes
        img = image_data[0].to_numpy_array()
        undistorted_img = cv2.cvtColor(
            undistort_image(img, camera_dict), cv2.COLOR_RGB2BGR
        )
        if camera_dict["convert_grayscale"]:
            undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)

        # undistorted_img = np.rot90(img, k=3)  # TODO: remove, only for testing

        img_id = f"{camera_dict['id']}00{img_timestamp}"
        filename = f"{camera_dict['camera_label']}_{img_timestamp}.jpg"
        output_path = f"{output_dir}/{filename}"

        save_image(undistorted_img, image_data, output_path)

        imgs.append(
            {
                "id": img_id,
                "timestamp": img_timestamp,
                "path": output_path,
                "file": filename,
                "subfolder": subfolder,
                "camera_id": camera_dict["id"],
            }
        )

    return imgs


def load_closed_loop_trajectory(path):
    return mps.read_closed_loop_trajectory(path)


def get_camera_pose(trajectory, camera_calib, timestamp_ns):
    """
    Returns:
        T_camera_world (sophus.SE3): world -> camera SE3 transformation at the specified timestamp
    """

    pose_info = get_nearest_pose(trajectory, timestamp_ns)
    if pose_info is None:
        return None
    T_world_device = pose_info.transform_world_device
    T_device_camera = camera_calib.get_transform_device_camera()
    T_world_camera = T_world_device @ T_device_camera
    return T_world_camera.inverse()


def pose_to_str(camera_pose):
    quat_trans = camera_pose.to_quat_and_translation()[0]
    return " ".join([str(x) for x in quat_trans])


def get_images_pose(img_infos, camera, closed_trajectory_file, delete_wout_poses=True):
    trajectory = load_closed_loop_trajectory(closed_trajectory_file)
    imgs_with_poses = []
    for img_dict in img_infos:
        camera_pose = get_camera_pose(
            trajectory, camera["undistorted_params"], img_dict["timestamp"]
        )
        if camera_pose is None:
            if delete_wout_poses:
                os.remove(img_dict["path"])
            continue

        pose_str = pose_to_str(camera_pose)
        img_dict["quat_trans"] = pose_str
        img_dict["pose"] = camera_pose
        imgs_with_poses.append(img_dict)
    return imgs_with_poses


def get_camera_infos(
    provider,
    camera_label,
    camera_id,
    downsample_rate=1,
    convert_grayscale=False,
    crop_pixels=0,
    rotate_90=True,
    out_resolution=None
):
    stream_id = provider.get_stream_id_from_label(camera_label)
    camera_info = {
        "id": camera_id,
        "stream_id": stream_id,
        "camera_label": camera_label,
        "downsample_rate": downsample_rate,
        "convert_grayscale": convert_grayscale,
        "crop_pixels": crop_pixels,
    }
    orig_params = get_camera_params(provider, camera_label)
    camera_info["distorted_params"] = orig_params
    camera_info["undistorted_params"] = get_undistorted_params(
        orig_params, rotate_90=rotate_90, crop_pixels=crop_pixels, out_resolution=out_resolution
    )
    return camera_info


def images_to_str(img_infos):
    img_strings = []
    for img in img_infos:
        if img["subfolder"] != "":
            file = img["subfolder"] + "/" + img["file"]
        else:
            file = img["file"]
        img_info_list = [
            img["id"],
            img["quat_trans"],
            img["camera_id"],
            file,
        ]
        img_strings.append(" ".join([str(x) for x in img_info_list]))
        img_strings.append("")  # Empty newline
    img_strings.append("")  # empty newline at the end
    return img_strings


### POINT CLOUD DATA   ####
def load_point_cloud(
    path, inverse_distance_std_threshold=0.001, distance_std_threshold=0.05
):
    points = mps.read_global_point_cloud(path)

    filtered_points = filter_points_from_confidence(
        points, inverse_distance_std_threshold, distance_std_threshold
    )

    points_list = [
        {"uid": point.uid, "pos": point.position_world} for point in filtered_points
    ]
    return points_list


def load_observations(path, points=None):
    ## Returns a list of point observations with the attributes:
    # uid - point uid
    # frame_tracking_timestamp_us - Aria device timestamp of the host frame’s center of exposure, in microsecond
    # camera_serial - The serial number of the camera which observes this point
    # u, v - The sub-pixel-accuracy observed measurements of the point in pixels, in the observing frame’s camera
    observations = mps.read_point_observations(path)
    if points is not None:
        ## Filter the observations of the given points, only

        points_uid = [x["uid"] for x in points]

        """from datetime import datetime

        now = datetime.now
        point_load = now()
        print("Point uid loading:", now() - point_load)

        new_observations = []
        for x in observations:
            observation_load = now()

            if x.point_uid in points_uid:
                new_observations.append(x)
            print("Observation filter:", now() - observation_load)
        

        max_min = {}
        for x in observations:
            if x.camera_serial not in max_min:
                max_min[x.camera_serial] = {
                    "min_u": x.uv[0],
                    "max_u": x.uv[0],
                    "min_v": x.uv[1],
                    "max_v": x.uv[1],
                }
            if x.uv[0] < max_min[x.camera_serial]["min_u"]:
                max_min[x.camera_serial]["min_u"] = x.uv[0]
            if x.uv[0] > max_min[x.camera_serial]["max_u"]:
                max_min[x.camera_serial]["max_u"] = x.uv[0]
            if x.uv[1] < max_min[x.camera_serial]["min_v"]:
                max_min[x.camera_serial]["min_v"] = x.uv[1]
            if x.uv[1] > max_min[x.camera_serial]["max_v"]:
                max_min[x.camera_serial]["max_v"] = x.uv[1]
        print(max_min)
        
        print(*list(set(x.camera_serial for x in observations)), sep="\n")
        """
        observations = (x for x in observations if x.point_uid in points_uid)

    return [
        {
            "point_uid": x.point_uid,
            "uv": x.uv,
            "frame_timestamp": x.frame_capture_timestamp,
            "camera_serial": x.camera_serial,
        }
        for x in observations
    ]


def get_point_colors(points, observations):
    pixels_by_uid = defaultdict(list)
    for obs in observations:
        pixels_by_uid[obs.uid].append(
            (
                obs.frame_tracking_timestamp_us,
                obs.camera_serial,
                obs.u,
                obs.v,
            )
        )

    for point in points:
        point["observations"] = pixels_by_uid.get(point["uid"], [])

    return points


def add_image_observations(provider, images, observations):
    # TODO: undistortion of the observation points is needed, as well
    print(images[0])
    print(observations[0])

    return images


def points_to_str(points):
    points_str = []
    for point in points:
        point_pos_str = " ".join([str(x) for x in point["pos"].tolist()])
        uid = point["uid"]
        points_str.append(f"{uid} {point_pos_str} 255 255 255 1")
    return points_str


def save_data(path, camera_dicts, image_list, points_list):
    camera_strings = [
        pinhole_params_to_str(camera_dict) for camera_dict in camera_dicts
    ]
    camera_str = "\n".join(camera_strings)
    images_str = "\n".join(images_to_str(image_list))
    points_str = "\n".join(points_to_str(points_list))

    files = ("cameras.txt", "images.txt", "points3D.txt")
    contents = (camera_str, images_str, points_str)

    for filename, content in zip(files, contents):
        filepath = path + "/" + filename
        with open(filepath, "w") as file:
            file.write(content)


if __name__ == "__main__":
    DOWNSAMPLE_RATE = 5
    filename = "first_floor"
    vrsfile = f"inputs/first_floor/{filename}.vrs"
    closed_trajectory = (
        f"inputs/first_floor/mps_{filename}_vrs/slam/closed_loop_trajectory.csv"
    )
    semidense_points = (
        f"inputs/first_floor/mps_{filename}_vrs/slam/semidense_points.csv.gz"
    )
    semidense_observations = (
        f"inputs/first_floor/mps_{filename}_vrs/slam/semidense_observations.csv.gz"
    )

    output_dir = "outputs/first_floor_grayscale_downsample5"

    sparse_dir = output_dir + "/sparse_prior"
    os.makedirs(sparse_dir, exist_ok=True)

    #### CAMERA
    provider = load_vrs(vrsfile)

    camera_rgb = get_camera_infos(
        provider,
        "camera-rgb",
        1,
        downsample_rate=DOWNSAMPLE_RATE,
        convert_grayscale=False,
        crop_pixels=150,
    )
    camera_left_slam = get_camera_infos(
        provider, "camera-slam-left", 2, downsample_rate=DOWNSAMPLE_RATE, crop_pixels=0, out_resolution=680
    )
    camera_right_slam = get_camera_infos(
        provider, "camera-slam-right", 3, downsample_rate=DOWNSAMPLE_RATE, crop_pixels=0, out_resolution=680
    )
    cameras = [camera_left_slam, camera_right_slam]
    image_subfolders = ["left", "right"]
    test_camera = camera_rgb

    print(
        f"Distorted camera parameters ({test_camera['camera_label']}):",
        test_camera["distorted_params"],
        end="\n\n",
    )
    print(
        "Undistorted camera parameters:", test_camera["undistorted_params"], end="\n\n"
    )
    print(pinhole_params_to_str(test_camera))

    ### IMAGES
    images = []
    for camera, subfolder in zip(cameras, image_subfolders):
        camera_images = undistort_images(
            provider,
            output_dir,
            subfolder,
            camera_dict=camera,
        )
        camera_images = get_images_pose(
            camera_images, camera, closed_trajectory, delete_wout_poses=True
        )
        images += camera_images

    print(*images_to_str(images)[:5], sep="\n")

    ## Get a single image:
    frame = provider.get_image_data_by_index(test_camera["stream_id"], 0)
    img = frame[0].to_numpy_array()

    undistorted_img = undistort_image(img, test_camera)
    print(
        "First and last pixel of undistorted image:",
        undistorted_img[0, 0],
        undistorted_img[-1, -1],
    )
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img)
    ax[0].set_title(f"Original Image {test_camera['camera_label']}")
    ax[1].imshow(undistorted_img)
    ax[1].set_title("Undistorted Image")
    plt.savefig("outputs/undistortion_example.jpg")
    plt.close()

    ### POINTS
    point_positions = load_point_cloud(
        semidense_points,
        inverse_distance_std_threshold=0.001,
        distance_std_threshold=0.05,
    )
    # observations = load_observations(semidense_observations, points=point_positions)
    # print("Observations loaded!")

    # images = add_image_observations(provider, images, observations)

    ### OUTPUT
    save_data(sparse_dir, cameras, images, point_positions)

"""
TODO:
    Currently, the stream only allows to have 1 FPS (or lower)
        -> use different recording profile for future experiments

    Maybe add proper point colors
    DONE: Use image cropping to remove black corners

    
    DONE: Resolve COLMAP format by running
        colmap model_converter --input_path sparse --output_path sparse --output_type TXT
    DONE: <- in images.txt, separate each record by an empty line


    What exactly does OpenMVS need for DensifyPointCloud
        Probably all point observations per point in images.txt
        Tracks in points3D.txt (?)
    Textures in 2.4.0 are broken


    # 1. Convert COLMAP model
    InterfaceCOLMAP \
        -i colmap_sparse \
        -o scene.mvs

    # 2. Dense reconstruction
    DensifyPointCloud scene.mvs

    # 3. Mesh
    ReconstructMesh scene_dense.mvs

    # 4. Texture
    TextureMesh scene_dense_mesh.mvs --resolution-level=3  (--decimate=0.1 does not work)

    
    cd /home/stani/aria/outputs && InterfaceCOLMAP -i ground_floor_aria -o scene.mvs && DensifyPointCloud scene.mvs



    ## Pipeline with OpenMVS 2.4.0 (https://github.com/cdcseacave/openMVS/wiki/Usage)
    # 1. Convert COLMAP model
    InterfaceCOLMAP \
        -i colmap_sparse \
        -o scene.mvs

    # 2. Dense reconstruction
    DensifyPointCloud scene.mvs

    (optional) Dense reconstruction with Semi-global matching
    DensifyPointCloud scene.mvs --fusion-mode -1
    DensifyPointCloud scene.mvs --fusion-mode -2
    
    # 3. Rough Mesh reconstruction 
    ReconstructMesh scene_dense.mvs -p scene_dense.ply
        
    # 4. Mesh refinement
    RefineMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_dense_mesh_refine.mvs

    # 5. Mesh Texturing
    From rough mesh:
        TextureMesh scene_dense.mvs -m scene_dense_mesh.ply -o scene_dense_mesh_texture.mvs

    From refined mesh:
        TextureMesh scene_dense.mvs -m scene_dense_mesh_refine.ply -o scene_dense_mesh_refine_texture.mvs --global-seam-leveling 0 --local-seam-leveling 0

        


    COLMAP commands in Neuralangelo:

    colmap feature_extractor \
    --database_path=${1}/database.db \
    --image_path=${1}/images_raw \
    --ImageReader.camera_model=SIMPLE_RADIAL \
    --ImageReader.single_camera=true \
    --SiftExtraction.use_gpu=true \
    --SiftExtraction.num_threads=32

colmap exhaustive_matcher \
    --database_path=${1}/database.db \
    --SiftMatching.use_gpu=true

mkdir -p ${1}/sparse
colmap mapper \
    --database_path=${1}/database.db \
    --image_path=${1}/images_raw \
    --output_path=${1}/sparse

cp ${1}/sparse/0/*.bin ${1}/sparse/
for path in ${1}/sparse/*/; do
    m=$(basename ${path})
    if [ ${m} != "0" ]; then
        colmap model_merger \
            --input_path1=${1}/sparse \
            --input_path2=${1}/sparse/${m} \
            --output_path=${1}/sparse
        colmap bundle_adjuster \
            --input_path=${1}/sparse \
            --output_path=${1}/sparse
    fi
done

colmap image_undistorter \
    --image_path=${1}/images_raw \
    --input_path=${1}/sparse \
    --output_path=${1} \
    --output_type=COLMAP
"""
