import asyncio
import numpy as np
import omni.usd
import omni.replicator.core as rep

from pxr import UsdGeom, Gf


def load_tum_trajectory(tum_path):
    """
    Load a TUM trajectory.

    Format:
        timestamp tx ty tz qx qy qz qw
    """

    data = np.loadtxt(
        tum_path,
        comments="#",
        skiprows=1,
    )

    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] != 8:
        raise ValueError(
            f"Expected 8 columns, got {data.shape[1]}"
        )

    return data


def set_camera_pose(camera_prim, position, quaternion):
    """
    Set the pose of an existing USD camera.

    quaternion = [qx, qy, qz, qw]
    """

    xformable = UsdGeom.Xformable(camera_prim)

    translate_op = None
    orient_op = None

    for op in xformable.GetOrderedXformOps():

        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op

        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient_op = op

    if translate_op is None:
        raise RuntimeError(
            "Camera has no translation xformOp."
        )

    if orient_op is None:
        raise RuntimeError(
            "Camera has no orientation xformOp."
        )

    # ------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------

    tx, ty, tz = [
        float(v) for v in position
    ]

    if (
        translate_op.GetPrecision()
        == UsdGeom.XformOp.PrecisionDouble
    ):
        translate_op.Set(
            Gf.Vec3d(tx, ty, tz)
        )
    else:
        translate_op.Set(
            Gf.Vec3f(tx, ty, tz)
        )

    # ------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------

    qx, qy, qz, qw = [
        float(v) for v in quaternion
    ]

    if (
        orient_op.GetPrecision()
        == UsdGeom.XformOp.PrecisionDouble
    ):
        orient_op.Set(
            Gf.Quatd(
                qw,
                Gf.Vec3d(qx, qy, qz)
            )
        )
    else:
        orient_op.Set(
            Gf.Quatf(
                qw,
                Gf.Vec3f(qx, qy, qz)
            )
        )


async def render_tum_trajectory(
    tum_path,
    camera_path="/World/TrajectoryCamera",
    output_dir="/tmp/trajectory_render",
    resolution=(1920, 1080),
):
    """
    Render a TUM camera trajectory inside Isaac Sim / Kit.

    This version uses the asynchronous Replicator API and is
    therefore suitable for running from the Isaac Sim GUI.
    """

    # ------------------------------------------------------------
    # Load trajectory
    # ------------------------------------------------------------

    trajectory = load_tum_trajectory(
        tum_path
    )

    print(
        f"Loaded {len(trajectory)} poses."
    )

    # ------------------------------------------------------------
    # Get stage
    # ------------------------------------------------------------

    stage = omni.usd.get_context().get_stage()

    # ------------------------------------------------------------
    # Find camera
    # ------------------------------------------------------------

    camera_prim = stage.GetPrimAtPath(
        camera_path
    )

    if not camera_prim.IsValid():
        raise RuntimeError(
            f"Camera not found: {camera_path}\n"
            "Create the camera in Isaac Sim first."
        )

    if not camera_prim.IsA(
        UsdGeom.Camera
    ):
        raise RuntimeError(
            f"{camera_path} is not a USD Camera."
        )

    # ------------------------------------------------------------
    # Create render product
    # ------------------------------------------------------------

    render_product = rep.create.render_product(
        camera_path,
        resolution,
    )

    # ------------------------------------------------------------
    # Configure writer
    # ------------------------------------------------------------

    writer = rep.writers.get(
        "BasicWriter"
    )

    writer.initialize(
        output_dir=output_dir,
        rgb=True,
    )

    writer.attach(
        [render_product]
    )

    # ------------------------------------------------------------
    # Disable capture-on-play
    # ------------------------------------------------------------

    rep.orchestrator.set_capture_on_play(
        False
    )

    # ------------------------------------------------------------
    # Render trajectory
    # ------------------------------------------------------------

    for frame_idx, pose in enumerate(
        trajectory
    ):

        timestamp = pose[0]

        position = pose[1:4]

        quaternion = pose[4:8]

        # Set camera pose
        set_camera_pose(
            camera_prim,
            position,
            quaternion,
        )

        # --------------------------------------------------------
        # Render asynchronously
        # --------------------------------------------------------

        await rep.orchestrator.step_async(
            pause_timeline=True
        )

        if frame_idx % 30 == 0:

            print(
                f"Rendered "
                f"{frame_idx + 1}/{len(trajectory)} "
                f"(t={timestamp:.3f}s)"
            )

    # ------------------------------------------------------------
    # Wait for writer
    # ------------------------------------------------------------

    await rep.orchestrator.wait_until_complete_async()

    print(
        f"Finished rendering "
        f"{len(trajectory)} frames."
    )



asyncio.ensure_future(
    render_tum_trajectory(
        tum_path="/home/stani/slam/aria/outputs/first_floor_3dgrut/trajectory.tum",
        camera_path="/World/TrajectoryCamera",
        output_dir="/home/stani/slam/3dgrut/outputs/first_floor_3dgrut/first_floor_3dgut_mcmc_ppisp/first_floor_3dgrut-0309_094701/trajectory",
        resolution=(1920, 1080),
    )
)
