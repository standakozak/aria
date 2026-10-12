"""Export Aria semidense point projections as per-frame sparse depth NPZ files.

The output pixel grid defaults to 504x504 to match ``da3_colmap.py``. Depth
values are camera-space z in metres; unobserved pixels are NaN. Each archive
contains ``depth``, ``mask``, ``count``, the source and resized intrinsics, and
the RGB frame timestamp. Resized RGB frames are also written for comparison.

Example:
    python aria/aria_depth_maps.py dataset.vrs closed_loop_trajectory.csv \
        semidense_points.csv.gz semidense_observations.csv.gz \
        --output-dir outputs/aria_sparse --width 504 --height 504
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import TimeDomain

from make_colmap_data import (
    default_camera_stream,
    get_camera_infos,
    get_camera_pose,
    load_closed_loop_trajectory,
    load_point_cloud,
    load_vrs,
    undistort_image,
)


LEFT_SERIAL = "0072510f1b2104050900000821100001"
RIGHT_SERIAL = "0072510f1b210005030000192d170001"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vrs", type=Path, help="Aria recording VRS file")
    parser.add_argument("trajectory", type=Path, help="MPS closed_loop_trajectory.csv")
    parser.add_argument("points", type=Path, help="MPS semidense_points.csv.gz")
    parser.add_argument("observations", type=Path, help="MPS semidense_observations.csv.gz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=504, help="Output DA3 grid width")
    parser.add_argument("--height", type=int, default=504, help="Output DA3 grid height")
    parser.add_argument("--max-observation-delta-ms", type=float, default=60.0,
                        help="Use left/right SLAM observation frames within this distance of each RGB timestamp")
    parser.add_argument("--left-serial", default=LEFT_SERIAL)
    parser.add_argument("--right-serial", default=RIGHT_SERIAL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def timestamp_ns(timestamp):
    return int(timestamp.seconds * 1e9 + timestamp.microseconds * 1e3)


def load_observations_by_camera(path, valid_uids):
    """Map (camera serial, capture timestamp) to the set of observed point IDs."""
    grouped = defaultdict(set)
    for observation in mps.read_point_observations(str(path)):
        uid = observation.point_uid
        if uid in valid_uids:
            grouped[(observation.camera_serial, timestamp_ns(observation.frame_capture_timestamp))].add(uid)
    timestamps = defaultdict(list)
    for serial, ts in grouped:
        timestamps[serial].append(ts)
    for serial in timestamps:
        timestamps[serial].sort()
    return grouped, timestamps


def nearest_observed_uids(grouped, timestamps, camera_serial, timestamp, max_delta_ns):
    camera_timestamps = timestamps.get(camera_serial, [])
    if not camera_timestamps:
        return set()
    at = bisect_left(camera_timestamps, timestamp)
    candidates = camera_timestamps[max(0, at - 1):min(len(camera_timestamps), at + 1)]
    nearest = min(candidates, key=lambda ts: abs(ts - timestamp))
    return grouped[(camera_serial, nearest)] if abs(nearest - timestamp) <= max_delta_ns else set()


def make_sparse_map(points, camera_pose, camera_params, output_size):
    """Project world points and z-buffer collisions into the requested pixel grid."""
    out_width, out_height = output_size
    src_width, src_height = map(int, camera_params.get_image_size())
    sx, sy = out_width / src_width, out_height / src_height
    depth = np.full((out_height, out_width), np.nan, dtype=np.float32)
    count = np.zeros((out_height, out_width), dtype=np.uint16)
    for point in points:
        p_cam = camera_pose @ point["pos"]
        z = float(p_cam.flatten()[2])
        if not np.isfinite(z) or z <= 0:
            continue
        uv = camera_params.project_no_checks(p_cam)
        x, y = int(round(float(uv[0]) * sx)), int(round(float(uv[1]) * sy))
        if 0 <= x < out_width and 0 <= y < out_height:
            count[y, x] = min(int(count[y, x]) + 1, np.iinfo(np.uint16).max)
            if not np.isfinite(depth[y, x]) or z < depth[y, x]:
                depth[y, x] = z
    K = np.asarray(camera_params.get_projection_params(), dtype=np.float64)
    fx, fy, cx, cy = K[:4]
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    resized_intrinsics = intrinsics.copy()
    resized_intrinsics[0, :] *= sx
    resized_intrinsics[1, :] *= sy
    return depth, count, intrinsics, resized_intrinsics


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.max_observation_delta_ms < 0:
        raise SystemExit("width/height must be positive and observation delta nonnegative")

    output = args.output_dir.expanduser().resolve()
    maps_dir, images_dir = output / "depth", output / "images"
    maps_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    provider = load_vrs(str(args.vrs))
    camera = get_camera_infos(provider, "camera-rgb", 1, downsample_rate=1,
                              convert_grayscale=False, crop_pixels=150)
    camera_params = camera["params_undistorted"]
    trajectory = load_closed_loop_trajectory(str(args.trajectory))
    points = load_point_cloud(str(args.points), inverse_distance_std_threshold=0.001,
                              distance_std_threshold=0.05)
    points_by_uid = {point["uid"]: point for point in points}
    observations, observation_timestamps = load_observations_by_camera(args.observations, set(points_by_uid))
    max_delta_ns = int(args.max_observation_delta_ms * 1e6)
    output_size = (args.width, args.height)

    frame_count, no_pose_count, no_obs_count = 0, 0, 0
    for data in default_camera_stream(provider, camera):
        ts = int(data.get_time_ns(TimeDomain.DEVICE_TIME))
        stem = f"camera-rgb_{ts}"
        npz_path, rgb_path = maps_dir / f"{stem}.npz", images_dir / f"{stem}.jpg"
        if npz_path.exists() and rgb_path.exists() and not args.overwrite:
            frame_count += 1
            continue
        pose = get_camera_pose(trajectory, camera_params, ts)
        if pose is None:
            no_pose_count += 1
            continue

        uids = set()
        for serial in (args.left_serial, args.right_serial):
            uids.update(nearest_observed_uids(observations, observation_timestamps, serial, ts, max_delta_ns))
        frame_points = [points_by_uid[uid] for uid in uids]
        if not frame_points:
            no_obs_count += 1

        raw = data.image_data_and_record()[0].to_numpy_array()
        undistorted = cv2.cvtColor(undistort_image(raw.copy(), camera), cv2.COLOR_RGB2BGR)
        resized = cv2.resize(undistorted, output_size, interpolation=cv2.INTER_AREA)
        depth, count, K, K_resized = make_sparse_map(
            frame_points, pose, camera_params, output_size
        )
        np.savez_compressed(
            npz_path,
            depth=depth,
            mask=np.isfinite(depth),
            count=count,
            intrinsics=K,
            intrinsics_resized=K_resized,
            timestamp_ns=np.int64(ts),
            source_size=np.array(camera_params.get_image_size(), dtype=np.int32),
        )
        cv2.imwrite(str(rgb_path), resized)
        frame_count += 1
        print(f"{stem}: {len(frame_points)} projected points, {np.isfinite(depth).sum()} occupied pixels")

    print(f"Finished: {frame_count} RGB frames; {no_pose_count} without poses; {no_obs_count} without nearby observations")
    print(f"Sparse NPZ maps: {maps_dir}\nResized RGB frames: {images_dir}")


if __name__ == "__main__":
    main()
