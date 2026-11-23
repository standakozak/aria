"""Export a Project Aria recording as a COLMAP sparse model (text format).

Images are undistorted to a pinhole model, rotated upright and written to
``<output-dir>/images``. Camera poses come from the MPS closed-loop trajectory
and the 3D points from the MPS semidense point cloud. The COLMAP files
(``cameras.txt``, ``images.txt``, ``points3D.txt``) are written to
``<output-dir>/sparse_prior``. Frames without a pose are skipped.

By default only the RGB camera is exported, at full frame rate.

Example:
    python make_colmap_data.py inputs/first_floor/first_floor.vrs \
        --output-dir outputs/first_floor

    # RGB + both SLAM cameras, every 3rd frame
    python make_colmap_data.py inputs/first_floor/first_floor.vrs \
        --output-dir outputs/first_floor_all --cameras rgb slam-left slam-right \
        --downsample-rate 3
"""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import piexif
from piexif.helper import UserComment
from PIL import Image
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.calibration import distort_by_calibration
from projectaria_tools.core.data_provider import VrsDataProvider
from projectaria_tools.core.mps.utils import (
    filter_points_from_confidence,
    get_nearest_pose,
)
from projectaria_tools.core.sensor_data import TimeDomain

# Per-camera export settings: COLMAP camera id, VRS stream label, image
# subfolder (used when more than one camera is exported), border crop in
# pixels and optional square output size before cropping.
CAMERAS = {
    "rgb": {"id": 1, "label": "camera-rgb", "subfolder": "rgb", "crop_pixels": 150, "out_size": None},
    "slam-left": {"id": 2, "label": "camera-slam-left", "subfolder": "left", "crop_pixels": 0, "out_size": 680},
    "slam-right": {"id": 3, "label": "camera-slam-right", "subfolder": "right", "crop_pixels": 0, "out_size": 680},
}


def load_vrs(vrs_file: str) -> VrsDataProvider:
    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"
    return provider


def get_undistorted_params(orig_calibration, crop_pixels=150, out_size=None):
    """Linear (pinhole) calibration with the same focal length, optionally cropped."""
    width, height = orig_calibration.get_image_size()
    if out_size is not None:
        width = height = out_size
    return calibration.get_linear_camera_calibration(
        int(width - 2 * crop_pixels),
        int(height - 2 * crop_pixels),
        orig_calibration.get_focal_lengths()[0],
        "camera-rgb-pinhole",
        orig_calibration.get_transform_device_camera(),
    )


def get_camera_infos(
    provider,
    camera_label,
    camera_id,
    downsample_rate=1,
    convert_grayscale=False,
    crop_pixels=0,
    img_out_size=None,
    color_correction=True,
):
    if color_correction:
        provider.set_color_correction(True)
    orig_params = provider.get_device_calibration().get_camera_calib(camera_label)
    undistorted_unrotated = get_undistorted_params(
        orig_params, crop_pixels=crop_pixels, out_size=img_out_size
    )
    return {
        "id": camera_id,
        "stream_id": provider.get_stream_id_from_label(camera_label),
        "camera_label": camera_label,
        "downsample_rate": downsample_rate,
        "convert_grayscale": convert_grayscale,
        "crop_pixels": crop_pixels,
        "params_distorted": orig_params,
        "params_undistorted_unrotated": undistorted_unrotated,
        "params_undistorted": calibration.rotate_camera_calib_cw90deg(undistorted_unrotated),
    }


def pinhole_params_to_str(camera_dict):
    """Record for cameras.txt based on the undistorted pinhole camera model."""
    params = (
        [camera_dict["id"], "PINHOLE"]
        + camera_dict["params_undistorted"].get_image_size().tolist()[::-1]
        + camera_dict["params_undistorted"].get_projection_params().tolist()
    )
    return " ".join(str(x) for x in params)


def undistort_image(img_numpy, camera_dict, rotate_90=True):
    undistorted = distort_by_calibration(
        img_numpy, camera_dict["params_undistorted_unrotated"], camera_dict["params_distorted"]
    )
    if rotate_90:
        return np.rot90(undistorted, k=3)
    return undistorted


def default_camera_stream(provider, camera_dict):
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(camera_dict["stream_id"])
    deliver_option.set_subsample_rate(camera_dict["stream_id"], camera_dict["downsample_rate"])
    yield from provider.deliver_queued_sensor_data(deliver_option)


def save_image(img, image_record, output_path):
    """Save an RGB or grayscale image as JPEG with capture metadata in EXIF."""
    exposure = image_record.exposure_duration
    gain = image_record.gain
    temperature = image_record.temperature

    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    # Store a reasonable exposure (0 < t < 30s) as an ExposureTime rational
    if exposure is not None and 0 < exposure < 30:
        frac = Fraction(exposure).limit_denominator(1000000)
        exif_dict["Exif"][piexif.ExifIFD.ExposureTime] = (frac.numerator, frac.denominator)
    # Full metadata as JSON in UserComment
    meta = {"exposure": exposure, "gain": gain, "temperature": temperature}
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = UserComment.dump(json.dumps(meta))

    pil_img = Image.fromarray(np.ascontiguousarray(img))
    pil_img.save(output_path, "jpeg", exif=piexif.dump(exif_dict))


def undistort_images(provider, output_dir_parent, subfolder, camera_dict, limit=None, timestamp_name_first=False):
    output_dir = os.path.join(output_dir_parent, "images", subfolder)
    os.makedirs(output_dir, exist_ok=True)
    if os.listdir(output_dir):
        print(f"WARNING: image path {output_dir} exists and is not empty. The data may be inconsistent!")

    imgs = []
    for index, data in enumerate(default_camera_stream(provider, camera_dict)):
        if limit is not None and index >= limit:
            break
        img_timestamp = data.get_time_ns(TimeDomain.DEVICE_TIME)
        print("Timestamp: ", img_timestamp)
        image, image_record = data.image_data_and_record()

        # .copy() can be necessary, because undistortion uses the raw bytes
        undistorted_img = undistort_image(image.to_numpy_array().copy(), camera_dict)
        if camera_dict["convert_grayscale"] and undistorted_img.ndim == 3:
            undistorted_img = cv2.cvtColor(np.ascontiguousarray(undistorted_img), cv2.COLOR_RGB2GRAY)

        if timestamp_name_first:
            filename = f"img_{img_timestamp}_{camera_dict['camera_label']}.jpg"
        else:
            filename = f"{camera_dict['camera_label']}_{img_timestamp}.jpg"
        output_path = os.path.join(output_dir, filename)
        save_image(undistorted_img, image_record, output_path)

        imgs.append(
            {
                "id": f"{camera_dict['id']}00{img_timestamp}",
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
    T_world_camera = pose_info.transform_world_device @ camera_calib.get_transform_device_camera()
    return T_world_camera.inverse()


def get_images_pose(img_infos, camera, trajectory, delete_wout_poses=True):
    """Attach COLMAP poses to images; images without a pose are dropped."""
    imgs_with_poses = []
    for img_dict in img_infos:
        camera_pose = get_camera_pose(trajectory, camera["params_undistorted"], img_dict["timestamp"])
        if camera_pose is None:
            if delete_wout_poses:
                os.remove(img_dict["path"])
            continue
        quat_trans = camera_pose.to_quat_and_translation()[0]
        img_dict["quat_trans"] = " ".join(str(x) for x in quat_trans)
        imgs_with_poses.append(img_dict)
    return imgs_with_poses


def images_to_str(img_infos):
    img_strings = []
    for img in img_infos:
        file = f"{img['subfolder']}/{img['file']}" if img["subfolder"] else img["file"]
        img_strings.append(f"{img['id']} {img['quat_trans']} {img['camera_id']} {file}")
        img_strings.append("")  # COLMAP expects an (empty) 2D-points line per image
    img_strings.append("")
    return img_strings


def load_point_cloud(path, inverse_distance_std_threshold=0.001, distance_std_threshold=0.05):
    points = mps.read_global_point_cloud(path)
    filtered_points = filter_points_from_confidence(
        points, inverse_distance_std_threshold, distance_std_threshold
    )
    return [{"uid": point.uid, "pos": point.position_world} for point in filtered_points]


def points_to_str(points):
    return [
        f"{point['uid']} {' '.join(str(x) for x in point['pos'].tolist())} 255 255 255 1"
        for point in points
    ]


def save_data(path, camera_dicts, image_list, points_list):
    contents = {
        "cameras.txt": "\n".join(pinhole_params_to_str(camera) for camera in camera_dicts),
        "images.txt": "\n".join(images_to_str(image_list)),
        "points3D.txt": "\n".join(points_to_str(points_list)),
    }
    for filename, content in contents.items():
        with open(os.path.join(path, filename), "w") as file:
            file.write(content)


def save_undistortion_preview(provider, camera, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = provider.get_image_data_by_index(camera["stream_id"], 0)[0].to_numpy_array()
    undistorted_img = undistort_image(img.copy(), camera)
    cmap = "gray" if img.ndim == 2 else None
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img, cmap=cmap)
    ax[0].set_title(f"Original Image {camera['camera_label']}")
    ax[1].imshow(undistorted_img, cmap=cmap)
    ax[1].set_title("Undistorted Image")
    plt.savefig(output_path)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("vrs", type=Path, help="Aria recording VRS file")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output dataset directory (images/ and sparse_prior/)")
    parser.add_argument("--mps-dir", type=Path, default=None,
                        help="MPS SLAM output directory (default: <vrs dir>/mps_<vrs name>_vrs/slam)")
    parser.add_argument("--trajectory", type=Path, default=None,
                        help="closed_loop_trajectory.csv (default: <mps-dir>/closed_loop_trajectory.csv)")
    parser.add_argument("--points", type=Path, default=None,
                        help="semidense_points.csv.gz (default: <mps-dir>/semidense_points.csv.gz)")
    parser.add_argument("--cameras", nargs="+", choices=list(CAMERAS), default=["rgb"],
                        help="Cameras to export (default: rgb)")
    parser.add_argument("--downsample-rate", type=int, default=1,
                        help="Keep every N-th frame of each camera stream (default: 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Export at most this many frames per camera")
    parser.add_argument("--grayscale", action="store_true",
                        help="Convert RGB camera images to grayscale")
    parser.add_argument("--no-color-correction", dest="color_correction", action="store_false",
                        help="Disable Aria RGB color correction")
    parser.add_argument("--timestamp-name-first", action="store_true",
                        help="Name images img_<timestamp>_<camera>.jpg instead of <camera>_<timestamp>.jpg")
    parser.add_argument("--inverse-distance-std-threshold", type=float, default=0.001,
                        help="Semidense point filter threshold (default: 0.001)")
    parser.add_argument("--distance-std-threshold", type=float, default=0.05,
                        help="Semidense point filter threshold in metres (default: 0.05)")
    parser.add_argument("--preview", action="store_true",
                        help="Save an original vs. undistorted comparison image of the first frame")
    args = parser.parse_args()
    if args.downsample_rate < 1:
        parser.error("--downsample-rate must be >= 1")

    args.cameras = list(dict.fromkeys(args.cameras))  # deduplicate, keep order
    mps_dir = args.mps_dir or args.vrs.parent / f"mps_{args.vrs.stem}_vrs" / "slam"
    args.trajectory = args.trajectory or mps_dir / "closed_loop_trajectory.csv"
    args.points = args.points or mps_dir / "semidense_points.csv.gz"
    for path in (args.vrs, args.trajectory, args.points):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    return args


def main():
    args = parse_args()
    sparse_dir = args.output_dir / "sparse_prior"
    os.makedirs(sparse_dir, exist_ok=True)

    provider = load_vrs(str(args.vrs))
    cameras = []
    for name in args.cameras:
        config = CAMERAS[name]
        camera = get_camera_infos(
            provider,
            config["label"],
            config["id"],
            downsample_rate=args.downsample_rate,
            convert_grayscale=args.grayscale,
            crop_pixels=config["crop_pixels"],
            img_out_size=config["out_size"],
            color_correction=args.color_correction,
        )
        camera["subfolder"] = config["subfolder"] if len(args.cameras) > 1 else ""
        cameras.append(camera)
        print(f"{camera['camera_label']} undistorted pinhole camera: {pinhole_params_to_str(camera)}")

    if args.preview:
        preview_path = args.output_dir / "undistortion_example.jpg"
        save_undistortion_preview(provider, cameras[0], preview_path)
        print(f"Saved undistortion preview to {preview_path}")

    trajectory = load_closed_loop_trajectory(str(args.trajectory))
    images = []
    for camera in cameras:
        camera_images = undistort_images(
            provider,
            str(args.output_dir),
            camera["subfolder"],
            camera_dict=camera,
            limit=args.limit,
            timestamp_name_first=args.timestamp_name_first,
        )
        images += get_images_pose(camera_images, camera, trajectory, delete_wout_poses=True)

    points = load_point_cloud(
        str(args.points),
        inverse_distance_std_threshold=args.inverse_distance_std_threshold,
        distance_std_threshold=args.distance_std_threshold,
    )
    save_data(str(sparse_dir), cameras, images, points)
    print(f"Exported {len(images)} posed images and {len(points)} points to {args.output_dir}")


if __name__ == "__main__":
    main()
