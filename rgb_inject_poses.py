import os
from make_colmap_data import (
    load_vrs,
    get_camera_infos,
    pinhole_params_to_str,
    undistort_images,
    load_images,
    get_images_pose,
)


def save_data(path, camera_dict, image_list):
    camera_str = pinhole_params_to_str(camera_dict)

    camera_path = path + "/cameras.txt"
    with open(camera_path, "w") as cam_file:
        cam_file.write(camera_str)

    images_path = path + "/images.txt"
    with open(images_path, "r") as file:
        image_file_lines = file.readlines()

    index = 0
    while index < len(image_file_lines):
        line = image_file_lines[index]
        if line.strip().startswith("#"):
            index += 1
            continue

        line_elems = line.split(" ")
        assert line_elems[0].isdigit(), (
            f"Expects image index (or timestamp) on the line {index} in '{images_path}'"
        )

        ## COLMAP starts image indices from 1 (TODO: check with more data)
        img_index = int(line_elems[0]) - 1
        
        image_pose = image_list[img_index]["quat_trans"].split(" ")
        print(
            f"Updating image {line_elems[0]} (file {image_list[img_index]['file']}): ",
        )
        for i, pose_info in enumerate(image_pose):
            ## Rewriting the image pose
            print("\t", line_elems[i + 1] + " -> ", pose_info)
            line_elems[i + 1] = pose_info
        image_file_lines[index] = " ".join(line_elems)
        index += 2

    with open(images_path, "w") as file:
        file.writelines(image_file_lines)


if __name__ == "__main__":
    vrsfile = "inputs/94439916-b1ab-404b-9b30-716d0a0978a6.vrs"
    closed_trajectory = "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/closed_loop_trajectory.csv"
    semidense_points = "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/semidense_points.csv.gz"
    semidense_observations = "inputs/mps_94439916-b1ab-404b-9b30-716d0a0978a6_vrs/slam/semidense_observations.csv.gz"

    output_dir = "rgb2colmap"

    sparse_dir = output_dir + "/sparse_prior"
    os.makedirs(sparse_dir, exist_ok=True)

    #### CAMERA
    provider = load_vrs(vrsfile)

    camera_rgb = get_camera_infos(provider, "camera-rgb", 1)
    print("Distorted camera parameters:", camera_rgb["distorted_params"], end="\n\n")
    print(
        "Undistorted camera parameters:", camera_rgb["undistorted_params"], end="\n\n"
    )
    print(pinhole_params_to_str(camera_rgb))

    ### IMAGES
    images, _ = load_images(
        provider,
        output_dir,
        camera_dict=camera_rgb,
        downsample_rate=1,
    )

    images = get_images_pose(images, camera_rgb, closed_trajectory)

    ### OUTPUT
    save_data(sparse_dir, camera_rgb, images)
