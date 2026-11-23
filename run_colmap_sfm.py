"""Run COLMAP SfM with known poses on a dataset exported by preprocess_data.py.

Expected dataset layout (as written by preprocess_data.py):

    <dataset>/images/                 undistorted images (per-camera subfolders if >1 camera)
    <dataset>/sparse_prior/cameras.txt   PINHOLE intrinsics per camera
    <dataset>/sparse_prior/images.txt    MPS poses per image

Steps:
    1. colmap database_creator
    2. colmap feature_extractor  (once per camera, with the intrinsics from cameras.txt)
    3. colmap sequential_matcher (or exhaustive_matcher)
    4. Rewrite the prior model with the camera/image IDs assigned by the database
       (preprocess_data.py uses timestamp-based image IDs that do not fit COLMAP's
       32-bit IDs and point_triangulator requires IDs to match the database)
    5. colmap point_triangulator -> <dataset>/sparse/0
    6. colmap model_converter    -> TXT model in <dataset>/sparse/0
    7. (optional) colmap gui

Works with COLMAP 3.11 and newer: the GPU/thread option names that changed in
COLMAP 3.12 (SiftExtraction.* -> FeatureExtraction.*, SiftMatching.* ->
FeatureMatching.*) are detected from the installed binary.

Example:
    python run_colmap_sfm.py outputs/first_floor --gui
"""

import argparse
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def command_help(colmap, command):
    result = subprocess.run(
        [colmap, command, "-h"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return result.stdout


def option_prefix(help_text, new_prefix, old_prefix):
    """Pick the option group name supported by the installed COLMAP version."""
    if f"{new_prefix}.use_gpu" in help_text:
        return new_prefix
    if f"{old_prefix}.use_gpu" in help_text:
        return old_prefix
    raise RuntimeError(f"Neither {new_prefix}.use_gpu nor {old_prefix}.use_gpu found in COLMAP help")


def read_cameras(path):
    """cameras.txt -> {camera_id: (model, width, height, [params])}"""
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        elems = line.split()
        cameras[int(elems[0])] = (elems[1], int(elems[2]), int(elems[3]), elems[4:])
    return cameras


def read_images(path):
    """images.txt -> list of (qw qx qy qz tx ty tz as strings, camera_id, name).

    Each image has two lines; the second (2D points) line is ignored.
    """
    images = []
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    for i in range(0, len(lines), 2):
        elems = lines[i].split()
        if not elems:
            continue
        images.append((elems[1:8], int(elems[8]), " ".join(elems[9:])))
    return images


def read_database_images(database_path):
    """{image name: (image_id, camera_id)} as assigned by COLMAP."""
    with sqlite3.connect(database_path) as db:
        rows = db.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    return {name: (image_id, camera_id) for image_id, name, camera_id in rows}


def write_prior_with_db_ids(prior_cameras, prior_images, db_images, output_dir):
    """Write the prior model with camera and image IDs taken from the database."""
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_map = {}  # prior camera id -> database camera id
    image_lines = []
    missing = 0
    for pose, prior_cam_id, name in prior_images:
        if name not in db_images:
            missing += 1
            continue
        image_id, db_cam_id = db_images[name]
        if camera_map.setdefault(prior_cam_id, db_cam_id) != db_cam_id:
            raise RuntimeError(f"Images of prior camera {prior_cam_id} map to several database cameras")
        image_lines += [f"{image_id} {' '.join(pose)} {db_cam_id} {name}", ""]
    if missing:
        print(f"WARNING: {missing} images of the prior model are missing in the database")

    camera_lines = []
    for prior_cam_id, db_cam_id in sorted(camera_map.items(), key=lambda x: x[1]):
        model, width, height, params = prior_cameras[prior_cam_id]
        camera_lines.append(f"{db_cam_id} {model} {width} {height} {' '.join(params)}")

    (output_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n")
    (output_dir / "images.txt").write_text("\n".join(image_lines) + "\n")
    # The MPS semidense points have no 2D observations, point_triangulator
    # creates new points from the feature matches.
    (output_dir / "points3D.txt").write_text("")
    return len(image_lines) // 2


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path,
                        help="Dataset directory written by preprocess_data.py (contains images/ and sparse_prior/)")
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable (default: colmap)")
    parser.add_argument("--matcher", choices=["sequential", "exhaustive"], default="sequential",
                        help="Feature matcher (default: sequential)")
    parser.add_argument("--no-gpu", dest="use_gpu", action="store_false",
                        help="Run feature extraction and matching on the CPU")
    parser.add_argument("--num-threads", type=int, default=-1,
                        help="Threads for feature extraction (default: -1 = all cores)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete an existing database.db and sparse/ model before running")
    parser.add_argument("--gui", action="store_true",
                        help="Open the result in the COLMAP GUI when done")
    return parser.parse_args()


def main():
    args = parse_args()
    colmap = shutil.which(args.colmap)
    if colmap is None:
        sys.exit(f"COLMAP executable not found: {args.colmap}")

    dataset = args.dataset
    image_path = dataset / "images"
    prior_path = dataset / "sparse_prior"
    prior_db_path = dataset / "sparse_prior_db"
    sparse_root = dataset / "sparse"
    sparse_path = sparse_root / "0"  # standard COLMAP layout, expected by 3DGRUT
    database_path = dataset / "database.db"
    for path in (image_path, prior_path / "cameras.txt", prior_path / "images.txt"):
        if not path.exists():
            sys.exit(f"Not found: {path} (run preprocess_data.py first)")

    if database_path.exists() or sparse_root.exists():
        if not args.overwrite:
            sys.exit(f"{database_path} or {sparse_root} already exists, use --overwrite to replace them")
        database_path.unlink(missing_ok=True)
        shutil.rmtree(sparse_root, ignore_errors=True)
        shutil.rmtree(prior_db_path, ignore_errors=True)

    prior_cameras = read_cameras(prior_path / "cameras.txt")
    prior_images = read_images(prior_path / "images.txt")
    print(f"Prior model: {len(prior_cameras)} camera(s), {len(prior_images)} image(s)")

    extraction = option_prefix(command_help(colmap, "feature_extractor"), "FeatureExtraction", "SiftExtraction")
    matching = option_prefix(command_help(colmap, f"{args.matcher}_matcher"), "FeatureMatching", "SiftMatching")
    use_gpu = str(args.use_gpu).lower()

    # 1. Database
    run([colmap, "database_creator", "--database_path", database_path])

    # 2. Feature extraction, one run per camera so each gets its own intrinsics
    for camera_id, (model, _, _, params) in sorted(prior_cameras.items()):
        names = [name for _, cam_id, name in prior_images if cam_id == camera_id]
        if not names:
            continue
        image_list = dataset / f"image_list_camera{camera_id}.txt"
        image_list.write_text("\n".join(names) + "\n")
        run([
            colmap, "feature_extractor",
            "--database_path", database_path,
            "--image_path", image_path,
            "--image_list_path", image_list,
            f"--{extraction}.use_gpu", use_gpu,
            f"--{extraction}.num_threads", args.num_threads,
            "--ImageReader.camera_model", model,
            "--ImageReader.single_camera", "true",
            "--ImageReader.camera_params", ",".join(params),
        ])
        image_list.unlink()

    # 3. Matching
    run([
        colmap, f"{args.matcher}_matcher",
        "--database_path", database_path,
        f"--{matching}.use_gpu", use_gpu,
    ])

    # 4. Prior model with database IDs
    num_images = write_prior_with_db_ids(
        prior_cameras, prior_images, read_database_images(database_path), prior_db_path
    )
    print(f"Wrote prior model with database IDs ({num_images} images) to {prior_db_path}")

    # 5. Triangulation with fixed poses
    sparse_path.mkdir(parents=True, exist_ok=True)
    run([
        colmap, "point_triangulator",
        "--database_path", database_path,
        "--image_path", image_path,
        "--input_path", prior_db_path,
        "--output_path", sparse_path,
    ])

    # 6. Text model
    run([
        colmap, "model_converter",
        "--input_path", sparse_path,
        "--output_path", sparse_path,
        "--output_type", "TXT",
    ])
    print(f"\nSparse model written to {sparse_path}")

    # 7. GUI
    if args.gui:
        run([
            colmap, "gui",
            "--import_path", sparse_path,
            "--database_path", database_path,
            "--image_path", image_path,
        ])


if __name__ == "__main__":
    main()
