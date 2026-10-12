# Standalone Isaac Sim trajectory renderer.
# Run with:
#   /home/stani/isaacsim/python.sh move_along_trajectory.py \
#       --tum-path /path/to/trajectory.tum \
#       --scene /path/to/scene.usdz \
#       --output-dir /path/to/output \
#       --resolution 1920 1080 \
#       --save-persistent

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from isaacsim import SimulationApp

import time
import threading
recording_activated = {"value": False}



parser = argparse.ArgumentParser(
    description="Render a TUM trajectory through a USDZ scene in Isaac Sim."
)
parser.add_argument(
    "--tum-path", required=True, help="Path to the TUM trajectory file."
)
parser.add_argument("--scene", required=True, help="Path to the input USDZ file.")
parser.add_argument(
    "--output-dir", required=True, help="Directory for rendered frames and MP4."
)
parser.add_argument(
    "--resolution",
    nargs=2,
    type=int,
    default=(1920, 1080),
    metavar=("WIDTH", "HEIGHT"),
    help="Render resolution as WIDTH HEIGHT (default: 1920 1080).",
)
parser.add_argument(
    "--save-persistent",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Whether rendered images should be saved persistently to disk (default: True).",
)
parser.add_argument(
    "--headless",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Run Isaac Sim in headless mode (default: True).",
)
args = parser.parse_args()

# Create the simulation app before importing any Omniverse / USD modules.
app = SimulationApp(
    {
        "headless": bool(args.headless),
        "renderer": "RealTimePathTracing",
    }
)

import omni.usd
import omni.replicator.core as rep
from pxr import Gf, UsdGeom


def load_tum_trajectory(tum_path):
    """Load a TUM trajectory of the form: timestamp tx ty tz qx qy qz qw."""
    data = np.loadtxt(tum_path, comments="#", skiprows=1)

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != 8:
        raise ValueError(f"Expected 8 columns in TUM file, got {data.shape[1]}")

    return data


def ensure_camera_prim(stage, camera_path):
    """Create a USD camera prim at the requested path if needed."""
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid():
        prim = stage.DefinePrim(camera_path, "Camera")

    camera = UsdGeom.Camera(prim)
    camera.CreateFocalLengthAttr(15.0)
    camera.GetFocusDistanceAttr().Set(10.0)
    # camera.CreateHorizontalApertureAttr(20.955)
    # camera.CreateVerticalApertureAttr(20.955 * 9.0 / 16.0)
    return prim


def set_camera_pose(camera_prim, position, quaternion):
    """Set a USD camera's transform from TUM position/quaternion."""
    xformable = UsdGeom.Xformable(camera_prim)

    translate_op = None
    orient_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        orient_op = xformable.AddOrientOp()

    tx, ty, tz = [float(v) for v in position]
    qx, qy, qz, qw = [float(v) for v in quaternion]

    translate_op.Set(Gf.Vec3d(tx, ty, tz))
    orient_op.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))

def input_thread():
    x = input("Press ENTER to continue")
    recording_activated["value"] = True


def render_tum_trajectory(tum_path, scene, output_dir, resolution, save_persistent):
    """Open a USDZ scene, walk the TUM trajectory, render each pose, and export a video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory = load_tum_trajectory(tum_path)
    print(f"Loaded {len(trajectory)} camera poses from {tum_path}")

    # Clean old image frames if present so ffmpeg output is deterministic.
    for path in output_dir.iterdir():
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            path.unlink()

    # Open stage.
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        omni.usd.get_context().open_stage(str(scene))
        for _ in range(10):
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                break
            app.update()

    if stage is None:
        raise RuntimeError(f"Failed to open USD stage: {scene}")

    camera_path = "/World/TrajectoryCamera"
    camera_prim = ensure_camera_prim(stage, camera_path)

    print("Stage opened")

    # Create render product and writer.
    render_product = rep.create.render_product(camera_path, resolution)

    writer = rep.writers.get("BasicWriter")
    try:
        writer.initialize(
            output_dir=str(output_dir), rgb=True, save_persistent=bool(save_persistent)
        )
    except TypeError:
        # Some Isaac Sim versions do not support the save_persistent kwarg.
        writer.initialize(output_dir=str(output_dir), rgb=True)

    writer.attach([render_product])
    print("Render product initialized and attached")

    rep.orchestrator.set_capture_on_play(False)

    if not args.headless:
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()

        while not recording_activated["value"]:
            app.update()
            time.sleep(0.01)


    for frame_idx, pose in enumerate(trajectory):
        position = pose[1:4]
        quaternion = pose[4:8]
        set_camera_pose(camera_prim, position, quaternion)

        # Use the synchronous Replicator step directly. In Isaac Sim, the async step
        # can deadlock when another event loop is already running.
        rep.orchestrator.step()
        app.update()

        if frame_idx % 10 == 0 or frame_idx == len(trajectory) - 1:
            print(f"Rendered {frame_idx + 1}/{len(trajectory)} frames")

    # Let the writer flush any pending captures without starting a new event loop.
    rep.orchestrator.wait_until_complete()

    # Create MP4.
    if not output_dir.exists():
        raise RuntimeError(f"Output directory does not exist: {output_dir}")

    frame_pattern = output_dir / "rgb_%04d.png"
    if not any(output_dir.glob("rgb_*.png")):
        raise RuntimeError(f"No rendered RGB frames found in {output_dir}")

    mp4_path = output_dir / "trajectory.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        "30",
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    print(f"Encoding MP4: {mp4_path}")
    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Finished. Frames saved in {output_dir}")
    print(f"Video saved to {mp4_path}")


def main():

    # If the user provides --resolution as a list of two ints, use that.
    resolution = tuple(args.resolution)

    try:
        omni.usd.get_context().open_stage(str(args.scene))
        app.update()
        render_tum_trajectory(
            tum_path=args.tum_path,
            scene=args.scene,
            output_dir=args.output_dir,
            resolution=resolution,
            save_persistent=args.save_persistent,
        )
    finally:
        app.close()


if __name__ == "__main__":
    main()
