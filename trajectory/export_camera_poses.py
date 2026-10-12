# Run with /home/stani/isaacsim/python.sh ...
# After trajectory is created:
# ffmpeg -framerate 30 -i trajectory_watertight_100000/rgb_%04d.png -c:v libx264 -pix_fmt yuv420p trajectory.mp4

import argparse
import sys

import numpy as np

from isaacsim import SimulationApp


# ------------------------------------------------------------
# Start Isaac Sim
# ------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--scene",
    required=True,
    help="Path to USD scene",
)

parser.add_argument(
    "--output",
    default="camera_control_points.tum",
    help="Output TUM file",
)



args, unknown = parser.parse_known_args()

simulation_app = SimulationApp(
    {
        "headless": False,
    }
)
camera_path = "/World/TrajectoryCamera"


# ------------------------------------------------------------
# Imports that require Isaac Sim to be initialized
# ------------------------------------------------------------

import omni.usd
from omni.kit.viewport.utility import get_active_viewport

from pxr import UsdGeom


# ------------------------------------------------------------
# Open scene
# ------------------------------------------------------------

omni.usd.get_context().open_stage(args.scene)

simulation_app.update()


stage = omni.usd.get_context().get_stage()

# Create the camera prim if needed
if not stage.GetPrimAtPath(camera_path).IsValid():
    cam_prim = stage.DefinePrim(camera_path, "Camera")
else:
    cam_prim = stage.GetPrimAtPath(camera_path)

camera = UsdGeom.Camera(cam_prim)
camera.GetFocusDistanceAttr().Set(10.0)
camera.GetFocalLengthAttr().Set(15.0)

viewport = get_active_viewport()
if viewport is None:
    raise RuntimeError("No active viewport found")

viewport.camera_path = camera_path

# ------------------------------------------------------------
# Pose extraction
# ------------------------------------------------------------


def get_camera_world_pose(camera_prim):

    xform_cache = UsdGeom.XformCache()

    transform = xform_cache.GetLocalToWorldTransform(camera_prim)

    # Position
    translation = transform.ExtractTranslation()

    position = np.array(
        [
            float(translation[0]),
            float(translation[1]),
            float(translation[2]),
        ]
    )

    # Rotation
    rotation = transform.ExtractRotation()

    quaternion = rotation.GetQuat()

    qx = float(quaternion.GetImaginary()[0])

    qy = float(quaternion.GetImaginary()[1])

    qz = float(quaternion.GetImaginary()[2])

    qw = float(quaternion.GetReal())

    return position, np.array(
        [
            qx,
            qy,
            qz,
            qw,
        ]
    )


# ------------------------------------------------------------
# Interactive recording
# ------------------------------------------------------------

print()
print("=" * 60)
print("Camera trajectory recorder")
print("=" * 60)
print()
print(f"Camera: {camera_path}")
print()
print("Move the camera in the Isaac Sim viewport.")
print()
print("Press ENTER in this terminal to record the")
print("current camera pose.")
print()
print("Enter 'q' to finish.")
print("=" * 60)
print()


positions_xyz = []
rotations_quats = []
durations = []

import threading
import time

stop = {"value": False}


def input_thread():
    control_index = 0

    while not stop["value"]:
        try:
            if control_index == 0:
                cmd = input("Press Enter to record, q to quit: ")
            else:
                cmd = input("Press Enter to record (movement for 1 sec to this position), input a floating number to define duration, q to quit: ")
        except EOFError:
            break

        if cmd.lower() == "q":
            stop["value"] = True
            break
        elif control_index > 0:
            try:
                duration = float(cmd)
                if duration <= 0:
                    raise ValueError
            except ValueError:
                duration = 1.0
            durations.append(duration)

        # record pose here
        print("recording pose")

        # Update transform cache after camera movement
        position, quaternion = get_camera_world_pose(cam_prim)

        print()
        print(f"Pose {control_index}:")

        print(f"  Position: {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}")

        print(
            f"  Quaternion: "
            f"{quaternion[0]:.8f} "
            f"{quaternion[1]:.8f} "
            f"{quaternion[2]:.8f} "
            f"{quaternion[3]:.8f}"
        )

        if control_index > 0:
            print(f"Duration: {duration} s")

        positions_xyz.append([position[0], position[1], position[2]])
        rotations_quats.append(
            [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
        )

        control_index += 1


t = threading.Thread(target=input_thread, daemon=True)
t.start()

while simulation_app.is_running() and not stop["value"]:
    # Keep Isaac Sim responsive
    simulation_app.update()
    time.sleep(0.01)


# ------------------------------------------------------------
# Save control points
# ------------------------------------------------------------
from interpolate_camera_trajectory import export_interpolated_tum_trajectory

if len(positions_xyz) == 0:
    print("No poses recorded.")

else:
    export_interpolated_tum_trajectory(positions_xyz, rotations_quats, durations, output_path=args.output, fps=30)

simulation_app.close()
