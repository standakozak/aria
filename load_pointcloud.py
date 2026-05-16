import projectaria_tools.core.mps as mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence


if __name__ == "__main__":
    semidense_path = "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/semidense_points.csv.gz"
    semidense_observations = "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/semidense_observations.csv.gz"

    points = mps.read_global_point_cloud(semidense_path)

    # filter the point cloud using thresholds on the inverse depth and distance standard deviation
    inverse_distance_std_threshold = 0.001
    distance_std_threshold = 0.15

    filtered_points = filter_points_from_confidence(
        points, inverse_distance_std_threshold, distance_std_threshold
    )

    print(filtered_points[0])

    observations = mps.read_point_observations(semidense_observations)
    print(observations[0])
