import os
import random
from random import sample

import cv2
from projectaria_tools.core import mps

from make_colmap_data import (
    get_camera_infos,
    get_nearest_pose,
    load_closed_loop_trajectory,
    load_point_cloud,
    load_vrs,
    pinhole_params_to_str,
    pose_to_str,
    undistort_images,
)

random.seed(0)


def load_observations(path, points=None, timestamp_min=None, timestamp_max=None):
    ## Returns a list of point observations with the attributes:
    # uid - point uid
    # frame_tracking_timestamp_us - Aria device timestamp of the host frame’s center of exposure, in microsecond
    # camera_serial - The serial number of the camera which observes this point
    # u, v - The sub-pixel-accuracy observed measurements of the point in pixels, in the observing frame’s camera
    observations = mps.read_point_observations(path)
    if timestamp_min and timestamp_max is not None:
        observations = (
            x
            for x in observations
            if timestamp_min
            <= (
                x.frame_capture_timestamp.seconds * 1e9
                + x.frame_capture_timestamp.microseconds * 1e3
            )
            <= timestamp_max
        )
        print("Observations filtered by timestamps")

    if points is not None:
        ## Filter the observations of the given points, only
        points_uid = {x["uid"] for x in points}
        observations = (x for x in observations if x.point_uid in points_uid)
        print("Observations filtered by point cloud")

    return [
        {
            "uid": x.point_uid,
            "uv": x.uv,
            "frame_timestamp": x.frame_capture_timestamp,
            "camera_serial": x.camera_serial,
        }
        for x in observations
    ]


def filter_by_column(list_1, list_2, col="uid"):
    list_1_vals = {x[col] for x in list_1}
    list_2_vals = {x[col] for x in list_2}
    common_vals = list_1_vals.intersection(list_2_vals)
    list_1 = [x for x in list_1 if x[col] in common_vals]
    list_2 = [x for x in list_2 if x[col] in common_vals]
    return list_1, list_2


def plot_point(img, point_pos, size=5, point_color=(255, 255, 255)):
    # Point color is in BGR
    img = cv2.circle(
        img.copy(),
        list(map(int, point_pos)),
        radius=1,
        thickness=size,
        color=point_color,
    )
    return img


def get_metric_depth(point_world, T_camera_world):
    """
    Return metric z-depth in meters for a world-space 3D point.

    This is the point's z coordinate in the camera frame, not Euclidean range
    and not stereo disparity. Camera intrinsics are unused because depth is
    determined by the extrinsic transform.
    """
    point_camera = T_camera_world @ point_world["pos"]
    return float(point_camera.flatten()[2])


def get_point_projection(point_world, T_camera_world, camera_params):
    transformed_point = T_camera_world @ point_world["pos"]
    return camera_params.project_no_checks(transformed_point)


def get_device_world_pose(trajectory, timestamp_ns):
    """
    Returns:
    T_device_world (sophus.SE3): world -> device (no specific camera) SE3 transformation at the specified timestamp
    """
    pose_info = get_nearest_pose(trajectory, timestamp_ns)
    if pose_info is None:
        return None
    return pose_info.transform_world_device


def get_camera_pose(trajectory, camera_calib, timestamp_ns):
    """
    Returns:
        T_camera_world (sophus.SE3): world -> camera SE3 transformation at the specified timestamp
    """

    T_world_device = get_device_world_pose(trajectory, timestamp_ns)
    T_device_camera = camera_calib.get_transform_device_camera()
    T_world_camera = T_world_device @ T_device_camera
    return T_world_camera.inverse()


def get_images_pose(img_infos, camera, closed_trajectory_file, delete_wout_poses=True):
    trajectory = load_closed_loop_trajectory(closed_trajectory_file)
    imgs_with_poses = []
    for img_dict in img_infos:
        camera_pose = get_camera_pose(
            trajectory, camera["params_undistorted"], img_dict["timestamp"]
        )
        camera_pose_distorted = get_camera_pose(
            trajectory, camera["params_distorted"], img_dict["timestamp"]
        )
        if camera_pose is None:
            if delete_wout_poses:
                os.remove(img_dict["path"])
            continue

        pose_str = pose_to_str(camera_pose)
        img_dict["quat_trans"] = pose_str
        img_dict["pose_device_world"] = get_device_world_pose(
            trajectory, img_dict["timestamp"]
        )
        img_dict["pose_undistorted"] = camera_pose
        img_dict["pose_distorted"] = camera_pose_distorted
        imgs_with_poses.append(img_dict)
    return imgs_with_poses


def is_pixel_in_bounds(uv, width, height):
    return bool((0 <= uv[0] < width) and (0 <= uv[1] < height))


def get_sparse_rgbd(points, rgb_img, T_camera_world, camera_params):
    ## NOTE: uses nearest pixel interpolation
    rgbd_data = []

    img_width = camera_params.get_image_size()[0]
    img_height = camera_params.get_image_size()[1]

    for point in points:
        projected_point_uv = get_point_projection(point, T_camera_world, camera_params)
        depth = get_metric_depth(point, T_camera_world)
        bgr = None
        uv_rounded = (round(projected_point_uv[0]), round(projected_point_uv[1]))
        pixel_valid = is_pixel_in_bounds(uv_rounded, img_width, img_height)

        if pixel_valid:
            bgr = rgb_img[uv_rounded]
        rgbd_data.append(
            {
                "point": point,
                "valid": pixel_valid,
                "depth": depth,
                "bgr": bgr,
                "uv": projected_point_uv,
                "uv_rounded": uv_rounded,
            }
        )

    return rgbd_data


def project_points_to_img(
    points,
    orig_img,
    T_camera_world,
    camera_params,
    out_filename,
    check_bounds=True,
    point_color=(255, 255, 255),
    point_size=10,
):
    img_width = camera_params.get_image_size()[0]
    img_height = camera_params.get_image_size()[1]

    out_of_bounds_points = 0

    projections = []

    for point in points:
        projected_point = get_point_projection(point, T_camera_world, camera_params)
        projections.append(projected_point)
        if check_bounds and not is_pixel_in_bounds(
            projected_point, img_width, img_height
        ):
            out_of_bounds_points += 1
            continue
        orig_img = plot_point(
            orig_img, projected_point, size=point_size, point_color=point_color
        )
    if check_bounds:
        print(
            f"Out of bounds points: {out_of_bounds_points}/{len(points)} ({out_of_bounds_points / len(points) * 100}%)"
        )
    cv2.imwrite(out_filename, orig_img)
    return projections


if __name__ == "__main__":
    filename = "ground_floor"
    vrsfile = f"inputs/{filename}/{filename}.vrs"
    closed_trajectory = (
        f"inputs/{filename}/mps_{filename}_vrs/slam/closed_loop_trajectory.csv"
    )
    semidense_points = (
        f"inputs/{filename}/mps_{filename}_vrs/slam/semidense_points.csv.gz"
    )
    semidense_observations = (
        f"inputs/{filename}/mps_{filename}_vrs/slam/semidense_observations.csv.gz"
    )

    output_dir = "outputs/ground_floor_aria_TEMP"

    sparse_dir = output_dir + "/sparse_prior"
    os.makedirs(sparse_dir, exist_ok=True)

    #### CAMERA
    provider = load_vrs(vrsfile)

    camera_rgb = get_camera_infos(
        provider,
        "camera-rgb",
        1,
        downsample_rate=1,
        convert_grayscale=False,
        crop_pixels=150,
    )
    camera_left_slam = get_camera_infos(
        provider,
        "camera-slam-left",
        2,
        downsample_rate=5,
        crop_pixels=0,
        img_out_size=640,
        convert_grayscale=True,
    )
    camera_right_slam = get_camera_infos(
        provider,
        "camera-slam-right",
        3,
        downsample_rate=5,
        crop_pixels=0,
        img_out_size=640,
        convert_grayscale=True,
    )
    cameras = [camera_rgb, camera_left_slam, camera_right_slam]
    camera_serials = (
        "0072510f1b2104050900000821100001",
        "0072510f1b210005030000192d170001",
    )  # left and right in this order

    test_camera = camera_rgb

    print(
        f"Distorted camera parameters ({test_camera['camera_label']}):",
        test_camera["params_distorted"],
        end="\n\n",
    )
    print(
        "Undistorted camera parameters:", test_camera["params_undistorted"], end="\n\n"
    )
    print(pinhole_params_to_str(test_camera))

    ## RGB images
    rgb_images = undistort_images(
        provider=provider,
        output_dir_parent=output_dir,
        subfolder="rgb",
        camera_dict=camera_rgb,
        limit=1,
    )
    rgb_img = rgb_images[0]  # NOTE: testing on a single image
    rgb_imgs = get_images_pose(
        [rgb_img], camera_rgb, closed_trajectory, delete_wout_poses=False
    )

    # Closest left image
    left_images = undistort_images(
        provider=provider,
        output_dir_parent=output_dir,
        subfolder="left",
        camera_dict=camera_left_slam,
        limit=3,
    )
    closest_left_image = min(
        left_images, key=lambda x: abs(x["timestamp"] - rgb_img["timestamp"])
    )
    left_imgs = get_images_pose(
        [closest_left_image],
        camera_left_slam,
        closed_trajectory,
        delete_wout_poses=True,
    )
    closest_timestamp = closest_left_image["timestamp"]

    ## Get closest right image
    right_images = undistort_images(
        provider=provider,
        output_dir_parent=output_dir,
        subfolder="right",
        camera_dict=camera_right_slam,
        limit=3,
    )
    closest_right_image = min(
        right_images, key=lambda x: abs(x["timestamp"] - rgb_img["timestamp"])
    )
    right_imgs = get_images_pose(
        [closest_right_image],
        camera_right_slam,
        closed_trajectory,
        delete_wout_poses=True,
    )
    closest_timestamp = closest_right_image["timestamp"]

    print("Timestamps:")
    print(f"RGB: {rgb_imgs[0]['timestamp']}")
    print(f"Left: {left_imgs[0]['timestamp']}")
    print(
        f"Right: {right_imgs[0]['timestamp']} (Difference: {left_imgs[0]['timestamp'] - right_imgs[0]['timestamp']})"
    )

    ### POINTS
    point_positions = load_point_cloud(
        semidense_points,
        inverse_distance_std_threshold=0.001,
        distance_std_threshold=0.05,
    )
    observations = load_observations(
        semidense_observations,
        points=point_positions,
        timestamp_min=closest_timestamp - 1000,
        timestamp_max=closest_timestamp + 1000,
    )
    print(f"Observations loaded! ({len(observations)} filtered observations)")

    # Points seen by both cameras
    cam_1_observations = [
        x for x in observations if x["camera_serial"] == camera_serials[0]
    ]
    cam_2_observations = [
        x for x in observations if x["camera_serial"] == camera_serials[1]
    ]
    cam_1_observations, cam_2_observations = filter_by_column(
        cam_1_observations, cam_2_observations
    )

    point_positions_intersection, cam_1_observations = filter_by_column(
        point_positions, cam_1_observations, col="uid"
    )
    cam_1_observations, cam_2_observations = filter_by_column(
        cam_1_observations, cam_2_observations
    )

    # Points seen by at least one camera
    observations_uids = {x["uid"] for x in observations}
    point_positions_union = [
        x for x in point_positions if x["uid"] in observations_uids
    ]

    print(f"Filtered points ({len(point_positions_intersection)}) seen by both cameras")
    print(f"Filtered points ({len(point_positions_union)}) seen by at least one camera")
    print(f"Filtered observations 1 ({len(cam_1_observations)})")
    print(f"Filtered observations 2 ({len(cam_2_observations)})")

    NUM_SHOWN_POINTS = 50
    point_positions_intersection = sample(
        point_positions_intersection, NUM_SHOWN_POINTS
    )

    ## projection check for monochromatic cameras
    print("Projecting points to left undistorted")
    project_points_to_img(
        point_positions_intersection,
        left_imgs[0]["img_undistorted"],
        left_imgs[0]["pose_undistorted"],
        camera_left_slam["params_undistorted"],
        out_filename=output_dir + "/left_projected_aria_und.jpg",
        point_size=5,
    )

    print("Projecting points to left distorted")
    left_dist_projections = project_points_to_img(
        point_positions_intersection,
        left_imgs[0]["img_distorted"],
        left_imgs[0]["pose_distorted"],
        camera_left_slam["params_distorted"],
        out_filename=output_dir + "/left_projected_aria_dist.jpg",
        point_size=5,
    )

    print("Projecting points to right undistorted")
    project_points_to_img(
        point_positions_intersection,
        right_imgs[0]["img_undistorted"],
        right_imgs[0]["pose_undistorted"],
        camera_right_slam["params_undistorted"],
        out_filename=output_dir + "/right_projected_aria_und.jpg",
        point_size=5,
    )

    print("Projecting points to right distorted")
    right_dist_projections = project_points_to_img(
        point_positions_intersection,
        right_imgs[0]["img_distorted"],
        right_imgs[0]["pose_distorted"],
        camera_right_slam["params_distorted"],
        out_filename=output_dir + "/right_projected_aria_dist.jpg",
        point_size=5,
    )

    left_img_orig_obs = left_imgs[0]["img_distorted"]
    right_img_orig_obs = right_imgs[0]["img_distorted"]

    for i, chosen_point in enumerate(point_positions_intersection):
        chosen_cam_1_obs = next(
            x for x in cam_1_observations if x["uid"] == chosen_point["uid"]
        )  # left

        chosen_cam_2_obs = next(
            x for x in cam_2_observations if x["uid"] == chosen_point["uid"]
        )  # right

        print(
            f"UV left {chosen_cam_1_obs['uv']}, projected left: {left_dist_projections[i]}"
        )
        print(
            f"UV right {chosen_cam_2_obs['uv']}, projected right: {right_dist_projections[i]}"
        )

        left_img_orig_obs = plot_point(left_img_orig_obs, chosen_cam_1_obs["uv"])
        right_img_orig_obs = plot_point(right_img_orig_obs, chosen_cam_2_obs["uv"])

    cv2.imwrite(output_dir + "/left_uv.jpg", left_img_orig_obs)
    cv2.imwrite(output_dir + "/right_uv.jpg", right_img_orig_obs)

    # Projection onto RGB image
    print("Projecting points seen by both cameras to undistorted RGB:")
    rgb_und_projections = project_points_to_img(
        point_positions_union,
        rgb_imgs[0]["img_undistorted"],
        rgb_imgs[0]["pose_undistorted"],
        camera_rgb["params_undistorted"],
        out_filename=output_dir + "/rgb_projected_aria_und.jpg",
        point_color=(0, 0, 255),
    )

    print("Projecting points seen by both cameras to distorted RGB")
    rgb_dist_projections = project_points_to_img(
        point_positions_union,
        rgb_imgs[0]["img_distorted"],
        rgb_imgs[0]["pose_distorted"],
        camera_rgb["params_distorted"],
        out_filename=output_dir + "/rgb_projected_aria_distorted.jpg",
        point_color=(0, 0, 255),
    )

    rgbd_data = get_sparse_rgbd(
        point_positions_union,
        rgb_imgs[0]["img_undistorted"],
        rgb_imgs[0]["pose_undistorted"],
        camera_rgb["params_undistorted"],
    )
    print()
