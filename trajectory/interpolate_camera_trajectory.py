import numpy as np
from scipy.interpolate import CubicSpline

from scipy.spatial.transform import Rotation as R, Slerp


def export_interpolated_tum_trajectory(
    positions,
    orientations,
    durations,
    output_path,
    fps=30,
    boundary_condition="natural",
):
    """
    Generate a camera trajectory using cubic spline interpolation for
    position and SLERP interpolation for orientation, then export it
    in TUM format.

    Parameters
    ----------
    positions : list
        List of camera positions:

            [[x, y, z], ...]

    orientations : list
        List of camera orientations as quaternions:

    durations : list
        Duration in seconds for each movement segment.

        For N poses, this must contain N - 1 values:

            durations[0]: Pose 0 -> Pose 1
            durations[1]: Pose 1 -> Pose 2
            ...

    output_path : str
        Path to the output .tum file.

    fps : int or float
        Number of trajectory poses per second.

    boundary_condition : str
        Boundary condition for scipy.interpolate.CubicSpline.

        Common options:
            "natural"   - second derivative is zero at endpoints
            "clamped"   - first derivative is zero at endpoints

        Default: "natural"

    Output format
    -------------
    timestamp tx ty tz qx qy qz qw

    Quaternion order follows the TUM convention:

        qx qy qz qw

    Returns
    -------
    trajectory : numpy.ndarray
        Array of shape (N, 8) containing:

        [timestamp, x, y, z, qx, qy, qz, qw]
    """

    positions = np.asarray(positions, dtype=float)
    orientations = np.asarray(orientations, dtype=float)
    durations = np.asarray(durations, dtype=float)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    if len(positions) != len(orientations):
        raise ValueError("positions and orientations must have the same length.")

    if len(positions) < 2:
        raise ValueError("At least two poses are required.")

    if len(durations) != len(positions) - 1:
        raise ValueError("durations must contain len(positions) - 1 elements.")

    if positions.shape[1] != 3:
        raise ValueError("Each position must contain exactly [x, y, z].")

    if orientations.shape[1] != 4:
        raise ValueError("Each orientation must contain exactly three Euler angles.")

    if np.any(durations <= 0):
        raise ValueError("All durations must be greater than zero.")

    if fps <= 0:
        raise ValueError("fps must be greater than zero.")

    # ------------------------------------------------------------
    # Compute timestamps for control poses
    # ------------------------------------------------------------
    #
    # Example:
    #
    # durations = [2.0, 3.0, 1.0]
    #
    # control_times = [0.0, 2.0, 5.0, 6.0]
    #
    # These timestamps are used both for the position spline
    # and orientation interpolation.
    #
    control_times = np.concatenate(([0.0], np.cumsum(durations)))

    total_duration = control_times[-1]

    # ------------------------------------------------------------
    # Cubic spline interpolation for position
    # ------------------------------------------------------------
    #
    # Positions are interpolated independently for X, Y, and Z,
    # but all use the same time parameter.
    #
    position_spline = CubicSpline(
        control_times,
        positions,
        axis=0,
        bc_type=boundary_condition,
    )

    # ------------------------------------------------------------
    # Convert YXZ Euler angles to rotations
    # ------------------------------------------------------------
    #
    # SciPy:
    #
    #   uppercase "YXZ" = intrinsic rotations
    #
    # Quaternion output order is:
    #
    #   [qx, qy, qz, qw]
    #
    #rotations = R.from_euler(
    #    "YXZ",
    #    orientations,
    #    degrees=True,
    #)
    rotations = R.from_quat(
        orientations,
    )

    # ------------------------------------------------------------
    # SLERP interpolation for orientation
    # ------------------------------------------------------------

    slerp = Slerp(
        control_times,
        rotations,
    )

    # ------------------------------------------------------------
    # Generate frame timestamps
    # ------------------------------------------------------------
    #
    # We generate regular frame intervals and explicitly include
    # the final timestamp.
    #
    frame_times = np.arange(
        0.0,
        total_duration,
        1.0 / fps,
    )

    # Ensure that t = total_duration is included exactly once.
    frame_times = np.append(
        frame_times,
        total_duration,
    )

    # ------------------------------------------------------------
    # Interpolate positions and orientations
    # ------------------------------------------------------------

    interpolated_positions = position_spline(frame_times)

    interpolated_rotations = slerp(frame_times)

    interpolated_quaternions = interpolated_rotations.as_quat()

    # ------------------------------------------------------------
    # Combine into TUM trajectory
    # ------------------------------------------------------------

    trajectory = np.column_stack(
        (
            frame_times,
            interpolated_positions,
            interpolated_quaternions,
        )
    )

    # ------------------------------------------------------------
    # Export TUM trajectory
    # ------------------------------------------------------------

    np.savetxt(
        output_path,
        trajectory,
        fmt="%.9f",
        header="timestamp tx ty tz qx qy qz qw",
        comments="",
    )

    print(f"Saved {len(trajectory)} poses to: {output_path}")

    print(f"Total duration: {total_duration:.3f} s")

    return trajectory


if __name__ == "__main__":
    #  Generates a smooth trajectory with SLERP to visualize the scene "first_floor_aria" in nvidia isaac sim
    # xyz_positions = [
    #     (14.44, 7.79, 2.18),
    #     (8.97, 11.52, 1.22),
    #     (3.19, 2.12, 1.70),
    #     (-2.55, -3.24, 0.44),
    #     (3.69, -1.72, 0.15),  # Finish half-circle
    #     (1.38, -6.05, 0.03),  # start panning to show the side
    #     (5.25, -0.81, 0.03),
    #     (5.91, 4.22, -0.57),
    #     (10.28, 8.75, -0.12),
    # ]
    # yxz_rotations = [
    #     (68.90, 0.0, 108.13),
    #     (70.98, 0.0, 158.26),
    #     (68.90, 0.0, 131.96),
    #     (88.73, 0.0, -164.25),
    #     (79.55, 0.0, -2.03),
    #     (84.62, 0.0, 55.49),  # start panning to show the side
    #     (85.16, 0.0, 73.22),
    #     (87.68, 0.0, 24.75),
    #     (82.64, 0.0, 136.80),
    # ]

    xyz_positions = [
        (12.45, 6.86, 1.12),
        (6.18, 4.09, 0.58),
        (0.86, -0.87, -0.29),
        (-0.25, -2.23, -0.27),
        (0.40, -8.11, -0.31),
        (4.53, -0.08, 0.71),

    ]

    quat_rotations = [
        (0.33, 0.48, 0.67, 0.46),
        (0.33, 0.55, 0.66, 0.40),
        (0.29, 0.61, 0.66, 0.32),
        (0.12, 0.68, 0.72, 0.12),
        (0.70, -0.11, -0.11, 0.69),
        (0.70, -0.08, -0.08, 0.71)
    ]

    durations = [1, 1, 0.5, 0.5, 1]

    OUTPUT_PATH = "/home/stani/slam/aria/outputs/first_floor_3dgrut/trajectory.tum"

    export_interpolated_tum_trajectory(
        xyz_positions, quat_rotations, durations, output_path=OUTPUT_PATH
    )
