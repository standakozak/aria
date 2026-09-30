"""Run pose-conditioned Depth Anything 3 inference over a COLMAP scene in windows.

Each completed window is exported as per-frame NPZ files compatible with
``fuse_da3_vdb.py``. Inference is submitted to a running DA3 backend.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests


def _colmap_handler():
    """Import DA3's COLMAP reader, including from a neighboring source checkout."""
    try:
        from depth_anything_3.services.input_handlers import ColmapHandler
    except ImportError:
        source = Path(__file__).resolve().parents[2] / "Depth-Anything-3" / "src"
        if not source.is_dir():
            raise
        sys.path.insert(0, str(source))
        from depth_anything_3.services.input_handlers import ColmapHandler
    return ColmapHandler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("colmap_dir", type=Path, help="COLMAP directory with images/ and sparse/")
    p.add_argument("--sparse-subdir", default="", help="Sparse reconstruction subdirectory, e.g. 0")
    p.add_argument("--export-dir", type=Path, required=True, help="Output directory under the backend gallery")
    p.add_argument("--window-size", type=int, default=30, help="Maximum images per inference window")
    p.add_argument("--overlap", type=int, default=5, help="Shared context frames between neighboring windows")
    p.add_argument("--backend-url", default="http://localhost:8008", help="DA3 backend URL")
    p.add_argument("--gallery-dir", type=Path, default=Path("workspace/gallery"), help="Backend gallery root")
    p.add_argument("--task-timeout", type=float, default=1800, help="Maximum seconds to wait for each window")
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--process-res-method", default="upper_bound_resize")
    p.add_argument("--ref-view-strategy", default="saddle_balanced")
    p.add_argument("--no-align-to-input-ext-scale", action="store_true", help="Disable DA3 COLMAP scale alignment")
    p.add_argument("--conf-thresh-percentile", type=float, default=40.0)
    p.add_argument("--overwrite", action="store_true", help="Replace existing frame NPZ files")
    return p.parse_args()


def _relative_output(image_path: str, images_root: Path) -> Path:
    try:
        rel = Path(image_path).resolve().relative_to(images_root.resolve())
    except ValueError:
        rel = Path(image_path).name
    return rel.with_suffix(".npz")


def _wait_for_task(session, base_url: str, task_id: str, timeout: float):
    deadline = time.monotonic() + timeout
    while True:
        response = session.get(f"{base_url}/task/{task_id}", timeout=30)
        response.raise_for_status()
        task = response.json()
        state = task.get("status")
        if state == "completed":
            return
        if state == "failed":
            raise RuntimeError(f"DA3 backend task {task_id} failed: {task.get('message', task)}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for DA3 backend task {task_id}")
        progress = task.get("progress")
        suffix = f" ({progress:.0%})" if isinstance(progress, (float, int)) else ""
        print(f"  Backend status: {state}{suffix}")
        time.sleep(2)


def _load_result(path: Path, timeout: float):
    """Wait for the NPZ exporter thread to finish writing a readable archive."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as archive:
                    return {key: np.asarray(archive[key]) for key in archive.files}
            except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
                # DA3's exporter writes directly to results.npz on a background
                # thread. The file can appear before the ZIP central directory is
                # written, so an early read may raise BadZipFile.
                last_error = exc
        time.sleep(0.25)
    detail = f": {last_error}" if last_error else ""
    raise RuntimeError(f"DA3 did not finish writing a readable NPZ at {path}{detail}")


def main():
    args = parse_args()
    if args.window_size < 1:
        raise SystemExit("--window-size must be at least 1")
    if args.overlap < 0 or args.overlap >= args.window_size:
        raise SystemExit("--overlap must be nonnegative and smaller than --window-size")
    if args.task_timeout <= 0:
        raise SystemExit("--task-timeout must be positive")

    colmap_dir = args.colmap_dir.expanduser().resolve()
    export_dir = args.export_dir.expanduser().resolve()
    gallery_dir = args.gallery_dir.expanduser().resolve()
    try:
        export_dir.relative_to(gallery_dir)
    except ValueError as exc:
        raise SystemExit(f"Export directory must be inside backend gallery {gallery_dir}: {export_dir}") from exc
    try:
        # The backend validates export_dir relative to its process working directory.
        backend_export_root = export_dir.relative_to(Path.cwd().resolve())
    except ValueError as exc:
        raise SystemExit(
            "Export directory must also be representable relative to the backend working directory; "
            "pass a relative --export-dir such as workspace/gallery/my_group/dataset"
        ) from exc

    ColmapHandler = _colmap_handler()
    images, extrinsics, intrinsics = ColmapHandler.process(str(colmap_dir), args.sparse_subdir)
    export_dir.mkdir(parents=True, exist_ok=True)
    images_root = colmap_dir / "images"

    step = args.window_size - args.overlap
    starts = []
    start = 0
    while start < len(images):
        starts.append(start)
        if start + args.window_size >= len(images):
            break
        start += step

    session = requests.Session()
    api_key = os.environ.get("DA3_BACKEND_API_KEY")
    if api_key:
        session.headers["X-API-Key"] = api_key
    base_url = args.backend_url.rstrip("/")
    # Fail early with a useful message if the server cannot be reached.
    try:
        status = session.get(f"{base_url}/status", timeout=10)
        status.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Cannot reach DA3 backend at {base_url}: {exc}") from exc

    written = set()
    for window_no, start in enumerate(starts, 1):
        end = min(start + args.window_size, len(images))
        print(f"Window {window_no}/{len(starts)}: frames {start + 1}-{end} of {len(images)}")
        # Per-window temporary backend exports are removed after conversion, so
        # fuse_da3_vdb.py only sees the final one-file-per-frame NPZ collection.
        window_name = f".da3_window_{os.getpid()}_{window_no:05d}"
        window_relative = backend_export_root / window_name
        window_dir = Path.cwd() / window_relative
        if window_dir.exists():
            shutil.rmtree(window_dir)

        payload = {
            "image_paths": images[start:end],
            "export_dir": window_relative.as_posix(),
            "export_format": "npz",
            "process_res": args.process_res,
            "process_res_method": args.process_res_method,
            "align_to_input_ext_scale": not args.no_align_to_input_ext_scale,
            "ref_view_strategy": args.ref_view_strategy,
            "conf_thresh_percentile": args.conf_thresh_percentile,
            "extrinsics": extrinsics[start:end].astype(np.float64).tolist(),
            "intrinsics": intrinsics[start:end].astype(np.float64).tolist(),
        }
        try:
            response = session.post(f"{base_url}/inference", json=payload, timeout=60)
            response.raise_for_status()
            submission = response.json()
            if not submission.get("success") or not submission.get("task_id"):
                raise RuntimeError(f"Backend rejected window {window_no}: {submission}")
            _wait_for_task(session, base_url, submission["task_id"], args.task_timeout)
            result_path = window_dir / "exports" / "npz" / "results.npz"
            result = _load_result(result_path, min(args.task_timeout, 120))
            if "depth" not in result:
                raise ValueError(f"{result_path}: missing depth array")
            depths = result["depth"]
            conf = result.get("conf")
            exts = result.get("extrinsics", extrinsics[start:end])
            ixts = result.get("intrinsics", intrinsics[start:end])
            if len(depths) != end - start:
                raise ValueError(f"{result_path}: expected {end - start} depth maps, got {len(depths)}")
            for offset in range(end - start):
                index = start + offset
                target = export_dir / _relative_output(images[index], images_root)
                if index in written or (target.exists() and not args.overwrite):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                data = {
                    "depth": np.asarray(depths[offset], dtype=np.float32),
                    "extrinsics": np.asarray(exts[offset]),
                    "intrinsics": np.asarray(ixts[offset]),
                }
                if conf is not None:
                    data["conf"] = np.asarray(conf[offset])
                np.savez_compressed(target, **data)
                written.add(index)
                print(f"  Saved {target.relative_to(export_dir)}")
        finally:
            if window_dir.exists():
                shutil.rmtree(window_dir)

    if not written and not any(export_dir.rglob("*.npz")):
        raise SystemExit("No frame NPZ files were written")
    print(f"Finished: {len(written)} frame maps written under {export_dir}")
    if len(written) < len(images):
        print(f"{len(images) - len(written)} frames already existed and were kept (use --overwrite to replace)")


if __name__ == "__main__":
    main()
