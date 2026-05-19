from projectaria_tools.core import data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
from projectaria_tools.core.stream_id import StreamId

vrsfile = "inputs/94439916-b1ab-404b-9b30-716d0a0978a6.vrs"

provider = data_provider.create_vrs_data_provider(vrsfile)
assert provider is not None, "Cannot open file"

# Get stream ID or stream label
stream_id = provider.get_stream_id_from_label("camera-slam-left")
# label = provider.get_label_from_stream_id(StreamId("1201-1"))


# Random access data by timestamp
for stream_id in provider.get_all_streams():
    t_first = provider.get_first_time_ns(stream_id, TimeDomain.DEVICE_TIME)
    t_last = provider.get_last_time_ns(stream_id, TimeDomain.DEVICE_TIME)
    query_timestamp = (t_first + t_last) // 2  # example query timestamp

    sensor_data = provider.get_sensor_data_by_time_ns(
        stream_id, query_timestamp, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST
    )


# Iterate over all sensor data
# for data in provider.deliver_queued_sensor_data():
#  print(data.get_time_ns(TimeDomain.DEVICE_TIME))


# Starts by default options which activates all sensors
deliver_option = provider.get_default_deliver_queued_options()

# Only play data from two cameras, also reduce framerate to half of vrs
deliver_option.deactivate_stream_all()
for label in ["camera-slam-left", "camera-slam-right"]:
    streamId = provider.get_stream_id_from_label(label)
    print(label, ":", stream_id)
    deliver_option.activate_stream(streamId)
    deliver_option.set_subsample_rate(streamId, 1)  # downsampling rate

# skip first 100ns
# deliver_option.set_truncate_first_device_time_ns(300_000_000_000)
# for data in provider.deliver_queued_sensor_data():
#  print(data.get_time_ns(TimeDomain.DEVICE_TIME))


## Section 2 - accessing image data
stream_id = provider.get_stream_id_from_label("camera-slam-left")
image_data = provider.get_image_data_by_index(stream_id, 0)
image = image_data[0].to_numpy_array()
print("Sample image:\n", image)

## Section 3 - calibration
device_calib = provider.get_device_calibration()
print("Device subtype:", device_calib.get_device_subtype())

# CPF = central-pupil-frame
label = "camera-slam-right"
transform_device_sensor = device_calib.get_transform_device_sensor(label)
transform_device_cpf = device_calib.get_transform_device_cpf()
transform_cpf_sensor = device_calib.get_transform_cpf_sensor(label)
print("SE3 Transform device -> sensor:", transform_device_sensor)

# Sensor calibration
label = "camera-slam-right"
sensor_calib = device_calib.get_camera_calib(label)
imu_calib = device_calib.get_imu_calib("imu-left")

print(dir(sensor_calib))
print(sensor_calib.get_projection_params())  # 15 params


## Section 4 - loading MPS output

import projectaria_tools.core.mps as mps

open_loop_path = (
    "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/open_loop_trajectory.csv"
)
open_loop_traj = mps.read_open_loop_trajectory(open_loop_path)

closed_loop_path = "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/closed_loop_trajectory.csv"
closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

# example: get transformation from this device to a world coordinate frame
for closed_loop_pose in closed_loop_traj:
    transform_world_device = closed_loop_pose.transform_world_device

# example: query to find the closest Timestamp device pose and move it to the Aria RGB camera pose
from projectaria_tools.core import data_provider
from projectaria_tools.core.mps.utils import get_nearest_pose

query_timestamp_ns = int(
    closed_loop_traj[1].tracking_timestamp.total_seconds() * 1e9
)  # to be updated with your VRS timestamps

pose_info = get_nearest_pose(closed_loop_traj, query_timestamp_ns)
if pose_info:
    T_world_device = pose_info.transform_world_device

    # Move this pose to the Project Aria RGB camera
    vrs_file = "inputs/94439916-b1ab-404b-9b30-716d0a0978a6.vrs"
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_file)
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
    T_world_rgb_camera = T_world_device @ T_device_rgb_camera

    print(T_world_rgb_camera)

    image_data = provider.get_image_data_by_time_ns(
        stream_id=rgb_stream_id,
        time_ns=query_timestamp_ns,
        time_domain=TimeDomain.DEVICE_TIME,
        time_query_options=TimeQueryOptions.CLOSEST,
    )
    print(image_data[0].to_numpy_array())
    print(dir(image_data[1]))


