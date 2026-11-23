"""Fuse Depth Anything 3 depth maps with VDBFusion and export a triangle PLY.

For standalone .npy depth maps, camera poses/intrinsics are read from COLMAP
images.txt and cameras.txt. DA3/COLMAP extrinsics are world-to-camera.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import open3d as o3d

import vdbfusion

DEFAULT_INPUT = Path(
    "Depth-Anything-3/workspace/gallery/my_group/project"
)
DEFAULT_COLMAP = Path(
    "Depth-Anything-3/inputs/colmap_dataset/sparse"
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="DA3 output directory containing .npy or .npz depth maps",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mesh PLY (default: <input>/vdbfusion_mesh.ply)",
    )
    p.add_argument(
        "--images-txt",
        type=Path,
        default=None,
        help="COLMAP images.txt; defaults to the matching inputs/.../sparse file",
    )
    p.add_argument(
        "--cameras-txt",
        type=Path,
        default=None,
        help="COLMAP cameras.txt; defaults to the matching inputs/.../sparse file",
    )
    p.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Original COLMAP image directory (default: inferred next to sparse/)",
    )
    p.add_argument(
        "--voxel-size",
        type=float,
        default=0.04,
        help="TSDF voxel size in depth units, normally metres",
    )
    p.add_argument(
        "--sdf-trunc",
        type=float,
        default=0.12,
        help="TSDF truncation distance in depth units",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Depth multiplier; set to 0.001 if depth is in millimetres",
    )
    p.add_argument("--min-depth", type=float, default=0.1)
    p.add_argument("--max-depth", type=float, default=10.0)
    p.add_argument("--stride", type=int, default=2, help="Integrate every Nth pixel")
    p.add_argument(
        "--confidence-percentile",
        type=float,
        default=0.0,
        help="Drop pixels below per-frame confidence percentile; 0 disables. Using --confidence-threshold is more preferable.",
    )
    p.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.0,
            help="Drop pixels below per-frame absolute confidence threshold",
        )
    p.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit; 0 means all frames",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save depth, confidence, and confidence-masked depth plots for up to 3 frames",
    )
    p.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Debug plot directory (default: <input>/debug_confidence)",
    )
    return p.parse_args()


def save_debug_visualization(depth, conf, confidence_threshold, args, label, frame_index):
    """Save the maps and the per-frame mask used by fusion as a three-panel plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = np.asarray(conf, dtype=np.float32).squeeze()
    if conf.shape != depth.shape:
        raise ValueError(
            f"{label}: confidence shape {conf.shape} does not match depth shape {depth.shape}"
        )
    keep = np.isfinite(depth) & (depth >= args.min_depth) & (depth <= args.max_depth)
    if confidence_threshold is not None:
        keep &= np.isfinite(conf) & (conf >= confidence_threshold)
    masked_depth = np.where(keep, depth, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    depth_image = axes[0].imshow(
        depth, cmap="viridis", vmin=args.min_depth, vmax=args.max_depth
    )
    axes[0].set_title("Depth")
    fig.colorbar(depth_image, ax=axes[0], fraction=0.046, pad=0.04, label="depth")
    conf_image = axes[1].imshow(conf, cmap="magma")
    axes[1].set_title("Confidence")
    fig.colorbar(conf_image, ax=axes[1], fraction=0.046, pad=0.04, label="confidence")
    masked_image = axes[2].imshow(
        masked_depth, cmap="viridis", vmin=args.min_depth, vmax=args.max_depth
    )
    axes[2].set_title(f"Depth retained ({keep.mean():.1%})")
    fig.colorbar(masked_image, ax=axes[2], fraction=0.046, pad=0.04, label="depth")
    for axis in axes:
        axis.set_axis_off()

    output_dir = (args.debug_dir or args.input / "debug_confidence").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"frame_{frame_index:04d}.png"
    cutoff = f"{confidence_threshold:.5g}" if confidence_threshold is not None else "disabled"
    fig.suptitle(f"{label} | confidence cutoff: {cutoff}")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved debug visualization: {output_path}")


def read_cameras(path):
    result = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if not f or f[0].startswith("#"):
            continue
        cid, model, width, height = int(f[0]), f[1], int(f[2]), int(f[3])
        v = list(map(float, f[4:]))
        if model == "PINHOLE":
            fx, fy, cx, cy = v[:4]
        elif model == "SIMPLE_PINHOLE":
            fx, cx, cy = v[:3]
            fy = fx
        else:
            raise ValueError(
                f"Unsupported COLMAP camera model {model}; use PINHOLE or SIMPLE_PINHOLE calibration"
            )
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        result[cid] = (K, width, height)
    return result


def qvec_rotation(qw, qx, qy, qz):
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )


def read_images(path):
    # COLMAP text format alternates each image record with a keypoint line.
    records = {}
    lines = [
        s.strip()
        for s in path.read_text().splitlines()
        if s.strip() and not s.lstrip().startswith("#")
    ]
    for line in lines[::2]:
        f = line.split()
        if len(f) < 10:
            continue
        qw, qx, qy, qz = map(float, f[1:5])
        R = qvec_rotation(qw, qx, qy, qz)
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = np.array(list(map(float, f[5:8])))
        image_name = " ".join(f[9:])
        record = (int(f[8]), w2c, image_name)
        # Keep both the full basename and its stem: DA3 depth exports may use
        # the source image stem with a .npy extension.
        records[Path(image_name).name] = record
        records.setdefault(Path(image_name).stem, record)
    return records


def load_npy_frames(root, cameras_path, images_path, source_images_dir):
    from PIL import Image

    cameras, images = read_cameras(cameras_path), read_images(images_path)
    files = sorted(root.rglob("*.npy"))
    for path in files:
        depth = np.load(path, allow_pickle=False)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            continue
        stem = re.sub(r"\.(depth|npy)$", "", path.name)
        rec = images.get(path.name) or images.get(stem) or images.get(path.stem)
        if rec is None:
            raise ValueError(
                f"No COLMAP image pose matches {path.name}; check image/depth filenames"
            )
        cid, w2c, image_name = rec
        K, camera_width, camera_height = cameras[cid]

        # DA3 resizes each source image before inference (default: longest side
        # to 504, then each dimension to the nearest multiple of 14). Its depth
        # map is in that processed pixel grid; COLMAP remains at source size.
        # Use actual source-image dimensions when available, then scale fx/cx
        # and fy/cy independently to the exported depth-map dimensions.
        source_path = source_images_dir / image_name
        if source_path.is_file():
            with Image.open(source_path) as image:
                source_width, source_height = image.size
        else:
            # COLMAP camera dimensions are the correct fallback when image files
            # have been moved; warn because unusual image transforms may differ.
            source_width, source_height = camera_width, camera_height
            print(
                f"Warning: source image {source_path} not found; using camera.txt "
                f"dimensions {source_width}x{source_height}"
            )
        depth_height, depth_width = depth.shape
        sx, sy = depth_width / source_width, depth_height / source_height
        K = K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        print(
            f"{path.name}: source={source_width}x{source_height}, "
            f"depth={depth_width}x{depth_height}, K scales=({sx:.6f}, {sy:.6f})"
        )
        yield depth, w2c, K, None, path.name


def load_npz_frames(files):
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            dk = next((k for k in ("depth", "depths") if k in z.files), None)
            if dk is None:
                raise ValueError(f"{path}: no depth array, keys={z.files}")
            depths = np.asarray(z[dk])
            exts = next(
                (np.asarray(z[k]) for k in ("exts", "extrinsics") if k in z.files), None
            )
            ixts = next(
                (np.asarray(z[k]) for k in ("ixts", "intrinsics") if k in z.files), None
            )
            conf = np.asarray(z["conf"]) if "conf" in z.files else None
            if depths.ndim == 2:
                depths = depths[None]
            if exts is not None and exts.ndim == 2:
                exts = exts[None]
            if ixts is not None and ixts.ndim == 2:
                ixts = ixts[None]
            if conf is not None and conf.ndim == 2:
                conf = conf[None]
            for i, depth in enumerate(depths):
                yield (
                    depth,
                    (exts[i] if exts is not None else None),
                    (ixts[i] if ixts is not None else None),
                    (conf[i] if conf is not None else None),
                    f"{path.name}[{i}]",
                )


def main():
    a = arguments()
    if min(a.voxel_size, a.sdf_trunc, a.depth_scale, a.stride) <= 0:
        raise SystemExit(
            "voxel size, truncation, depth scale, and stride must be positive"
        )
    root = a.input.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Depth input directory not found: {root}")
    npzs = sorted(root.rglob("*.npz"))
    if npzs:
        frames = load_npz_frames(npzs)
    else:
        images_txt = a.images_txt or (DEFAULT_COLMAP / "images.txt")
        cameras_txt = a.cameras_txt or (DEFAULT_COLMAP / "cameras.txt")
        if not images_txt.is_file() or not cameras_txt.is_file():
            raise SystemExit(
                "No .npz found and COLMAP images.txt/cameras.txt unavailable; pass --images-txt and --cameras-txt"
            )
        # COLMAP image paths are relative to the dataset's images/ directory.
        sparse_dir = next(
            (parent for parent in images_txt.parents if parent.name == "sparse"),
            images_txt.parent,
        )
        images_dir = a.images_dir or (sparse_dir.parent / "images")
        frames = load_npy_frames(root, cameras_txt, images_txt, images_dir)
    print("Depth maps loaded")
    volume = vdbfusion.VDBVolume(a.voxel_size, a.sdf_trunc, space_carving=True)
    integrated = 0
    debug_visualized = 0
    for depth, w2c, K, conf, label in frames:
        if a.max_frames and integrated >= a.max_frames:
            break
        if w2c is None or K is None:
            raise ValueError(f"{label}: camera pose or intrinsics missing")
        depth = np.asarray(depth, dtype=np.float32).squeeze() * a.depth_scale
        if depth.ndim != 2:
            raise ValueError(
                f"{label}: expected a single 2D depth map, got {depth.shape}"
            )
        h, w = depth.shape
        yy, xx = np.mgrid[0 : h : a.stride, 0 : w : a.stride]
        z = depth[:: a.stride, :: a.stride]
        valid = np.isfinite(z) & (z >= a.min_depth) & (z <= a.max_depth)
        confidence_threshold = None
        if conf is not None:
            c = np.asarray(conf).squeeze()[:: a.stride, :: a.stride]
            finite = c[np.isfinite(c)]

            if a.confidence_percentile > 0:
                confidence_threshold = np.percentile(finite, a.confidence_percentile)
            if a.confidence_threshold > 0:
                confidence_threshold = a.confidence_threshold

            if finite.size:
                confidence_threshold = a.confidence_threshold
                valid &= np.isfinite(c) & (c >= confidence_threshold)
        if a.debug and conf is not None and debug_visualized < 50:
            save_debug_visualization(
                depth, conf, confidence_threshold, a, label, debug_visualized
            )
            debug_visualized += 1
        if not valid.any():
            print(f"Skip {label}: no valid depth samples")
            continue
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        # OpenCV camera coordinates: +x right, +y down, +z forward.
        points_cam = np.stack(
            ((xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z), axis=-1
        )[valid]
        w2c = np.asarray(w2c)
        if w2c.shape == (3, 4):
            M = np.eye(4)
            M[:3, :] = w2c
            w2c = M
        if w2c.shape != (4, 4):
            raise ValueError(
                f"{label}: expected 3x4 or 4x4 world-to-camera pose; got {w2c.shape}"
            )
        c2w = np.linalg.inv(w2c)
        points_world = points_cam @ c2w[:3, :3].T + c2w[:3, 3]
        volume.integrate(
            np.ascontiguousarray(points_world, dtype=np.float64),
            c2w[:3, 3].astype(np.float64),
        )
        integrated += 1
        print(f"Integrated {label}: {len(points_world):,} points ({integrated} frames)")
    if not integrated:
        raise SystemExit("No valid depth frames were integrated")
    vertices, triangles = volume.extract_triangle_mesh()
    if len(vertices) == 0 or len(triangles) == 0:
        raise SystemExit(
            "VDBFusion returned an empty mesh; verify depth units, poses and depth limits"
        )
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles)
    )
    mesh.compute_vertex_normals()
    output = (a.output or root / "vdbfusion_mesh.ply").expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise SystemExit(f"Could not write mesh: {output}")
    print(
        f"Saved {len(vertices):,} vertices and {len(triangles):,} triangles to {output}"
    )


if __name__ == "__main__":
    main()
