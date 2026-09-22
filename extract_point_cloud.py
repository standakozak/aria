import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence
import open3d
import numpy as np


def export_points(points, file):
    np_points = np.array([x.position_world for x in points], dtype=np.float64)

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(np_points)
    export_status = open3d.io.write_point_cloud(file, pcd)
    return export_status


if __name__ == "__main__":
    semidense_path = "/mnt/c/Users/kozak/aria_glasses/mps_First_floor_june_profile0_vrs/slam/semidense_points.csv.gz"

    points = mps.read_global_point_cloud(semidense_path)

    np_points = np.array([x.position_world for x in points], dtype=np.float64)

    ## Unfiltered points
    export_points(points, "outputs/first_floor_june_profile0/points.ply")

    # filter the point cloud using thresholds on the inverse depth and distance standard deviation
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15

    ## Filtered points
    filtered_points = filter_points_from_confidence(
        points, inverse_distance_std_threshold, distance_std_threshold
    )
    export_points(filtered_points, "outputs/first_floor_june_profile0/filtered_points.ply")

    print(filtered_points[0])
