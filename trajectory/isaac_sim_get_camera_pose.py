import omni.usd
from pxr import UsdGeom


def get_camera_world_pose(camera_path):
    """
    Return the camera's world-space position and quaternion.

    Returns:
        position   = [x, y, z]
        quaternion = [qx, qy, qz, qw]
    """

    stage = omni.usd.get_context().get_stage()

    camera = stage.GetPrimAtPath(camera_path)

    if not camera.IsValid():
        raise RuntimeError(
            f"Camera not found: {camera_path}"
        )

    # Get the complete world transform
    xform_cache = UsdGeom.XformCache()
    world_transform = xform_cache.GetLocalToWorldTransform(
        camera
    )

    # Extract translation
    translation = world_transform.ExtractTranslation()

    position = [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
    ]

    # Extract rotation
    rotation = world_transform.ExtractRotation()

    # Convert Gf.Rotation -> quaternion
    quaternion = rotation.GetQuat()

    qx = float(quaternion.GetImaginary()[0])
    qy = float(quaternion.GetImaginary()[1])
    qz = float(quaternion.GetImaginary()[2])
    qw = float(quaternion.GetReal())

    return position, [qx, qy, qz, qw]
    
    
position, quaternion = get_camera_world_pose(
    "/World/Camera"
)

print(f"Position: \n[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
print(f"Quaternion: \n[{quaternion[0]:.2f}, {quaternion[1]:.2f}, {quaternion[2]:.2f}, {quaternion[3]:.2f}]" )