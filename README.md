# Digital twin reconstruction with Project Aria Glasses
## Installation
Two conda environments are needed, because VDBFusion only provides wheels for Python <= 3.10 with numpy 1.x, while the conda-forge COLMAP 4.2.1 build requires Python >= 3.11:
```bash
conda env create -f environment.yml             # "aria": COLMAP 4.2.1 (CUDA), projectaria-tools, Depth Anything 3, usd-core
conda env create -f environment-vdbfusion.yml   # "aria-vdbfusion": VDBFusion, Open3D
./install_3dgrut.sh                             # adds NVIDIA 3DGRUT (cloned to ../3dgrut) to "aria"
```
| Script | Environment |
|---|---|
| `preprocess_data.py`, `run_colmap_sfm.py`, `run_da3_inference.py`, `da3 backend`, `ply_to_usd.py`, 3DGRUT `train.py` | `aria` |
| `run_tsdf_fusion.py` | `aria-vdbfusion` |

Both files use only the conda-forge channel. If conda stops with `CondaToSNonInteractiveError` because `defaults` is listed in your `.condarc`, accept the Anaconda Terms of Service (`conda tos accept`) or remove the channel (`conda config --remove channels defaults`).

`install_3dgrut.sh` installs 3DGRUT with PPISP, fused-ssim and slangc, but without Kaolin (only needed for the playground) and tiny-cuda-nn (only needed for NHT). It also adds conda activation hooks that set `CUDA_HOME=/usr` (CUDA 12.8 from Lambda Stack, `/usr/local/cuda` is 11.7), `CC`/`CXX` to GCC 11 (the default GCC 9 fails with the GCC 11 C++ headers) and `TORCH_CUDA_ARCH_LIST=8.9`, which 3DGRUT uses to compile its CUDA kernels on the first run. Override these variables when running the script on another machine.

Tested with an NVIDIA RTX 4090 and driver 580 (PyTorch with CUDA 12.8, COLMAP built with CUDA 12.9).

## Export an Aria recording to COLMAP
```powershell
python preprocess_data.py inputs/first_floor/first_floor.vrs \
--output-dir outputs/first_floor
```
Undistorts the camera images to a pinhole model and writes a COLMAP text model (`sparse_prior/cameras.txt`, `images.txt`, `points3D.txt`) with poses from the MPS closed-loop trajectory and the MPS semidense point cloud. The MPS files are looked up in `<vrs dir>/mps_<vrs name>_vrs/slam` (override with `--mps-dir`, `--trajectory`, `--points`).

By default, only the RGB camera is exported at full frame rate. Important parameters:
- `cameras`: any of `rgb slam-left slam-right`. With more than one camera, images go into per-camera subfolders.
- `downsample-rate`: keep every N-th frame
- `grayscale`: convert RGB images to grayscale

See all parameters by running `python preprocess_data.py --help`

## COLMAP SfM with known poses
Requires [COLMAP](https://colmap.github.io/install.html) 3.11 or newer on the `PATH` (included in the `aria` environment; only the Python standard library is needed).
```powershell
python run_colmap_sfm.py outputs/first_floor --gui
```
Runs on the output directory of `preprocess_data.py` and keeps the MPS poses fixed:
1. `database_creator` creates `database.db`
2. `feature_extractor` runs once per exported camera with the PINHOLE intrinsics from `sparse_prior/cameras.txt`
3. `sequential_matcher` matches features
4. The prior model is rewritten to `sparse_prior_db/` with the camera and image IDs assigned by the database (COLMAP requires matching IDs, and the timestamp-based IDs from `preprocess_data.py` are too large for COLMAP)
5. `point_triangulator` triangulates 3D points with the fixed poses into `sparse/0/` (standard COLMAP layout, as expected by 3DGRUT)
6. `model_converter` writes `sparse/0/` as a text model (`cameras.txt`, `images.txt`, `points3D.txt`)
7. (optional) the COLMAP GUI opens the result

Important parameters:
- `gui`: open the sparse model in the COLMAP GUI when done
- `matcher`: `sequential` (default) or `exhaustive`. Sequential matching follows the image name order, so with several cameras it mostly matches images of the same camera.
- `no-gpu`: run feature extraction and matching on the CPU
- `num-threads`: feature extraction threads (default: all cores)
- `overwrite`: delete an existing `database.db` and `sparse/` before running
- `colmap`: path to the COLMAP executable (default: `colmap`)

See all parameters by running `python run_colmap_sfm.py --help`

## Gaussian splatting with 3DGRUT (3DGUT + MCMC + PPISP)
Trains [3DGRUT](https://github.com/nv-tlabs/3dgrut) on the posed RGB images and the triangulated points from `run_colmap_sfm.py` with the 3DGUT renderer, the MCMC densification strategy and PPISP post-processing (learned exposure, color, vignetting and camera response correction). NHT is not used. Run from the 3DGRUT directory with `conda activate aria`:
```bash
cd ../3dgrut
python train.py --config-name apps/colmap_3dgut_mcmc.yaml \
path=../aria/outputs/first_floor \
out_dir=runs experiment_name=first_floor_3dgut_mcmc_ppisp \
post_processing.method=ppisp \
export_usd.enabled=true export_usd.format=nurec \
export_usd.apply_normalizing_transform=false
```
Important parameters:
- `path`: dataset with `images/` and `sparse/0/` (output directory of `preprocess_data.py` after `run_colmap_sfm.py`)
- `post_processing.method=ppisp`: enable PPISP. It uses the exposure stored in the image EXIF data by `preprocess_data.py`. After training, `post_processing.n_distillation_steps` (default 5000) extra steps train the PPISP controller for novel views.
- `dataset.downsample_factor`: train on downsampled images (e.g. `2`; the images are resized once into `images_2/`)
- `n_iterations`: training iterations (default 30000)
- `export_usd.enabled=true export_usd.format=nurec`: export a NuRec USDZ (`export_last_nurec.usdz` in the run directory) for Isaac Sim, which can be used as `--gaussian-usdz` in `ply_to_usd.py`. Without `export_usd.format=nurec`, a standard USD `ParticleField` is exported.
- `export_usd.apply_normalizing_transform=false`: keep the Gaussians in the COLMAP world frame. **Always set this for NuRec exports**, see below.

### NuRec export and the normalizing transform
By default (`export_usd.apply_normalizing_transform=true`), 3DGRUT moves the Gaussians into a frame normalized to the camera poses and stores the inverse transform on the Gaussian volume (`/World/gauss/gauss` in the USDZ). The NuRec exporter writes this 4x4 matrix in column-vector order (`threedgrut/export/usd/nurec/serializer.py`, `Gf.Matrix4d(*corrected_matrix.flatten())`), so the translation ends up in the last column instead of the last row. USD reads this as a projective transform: the scene is not placed correctly, and `ply_to_usd.py --gaussian-usdz` copies the same matrix onto the mesh, which distorts it. With `apply_normalizing_transform=false`, the transform is the identity and the Gaussians, the cameras and the TSDF mesh share the COLMAP/MPS world frame.

The `--no-transform` flag of 3DGRUT's standalone exporter (`python -m threedgrut.export.scripts.export_usd`) has no effect for `--format nurec`: the NuRec exporter reads `export_usd.apply_normalizing_transform` from the config stored in the checkpoint instead. To re-export an existing checkpoint without the normalizing transform, use `export_3dgrut_nurec.py`. It takes the same arguments as 3DGRUT's exporter and sets this value to `false` after loading the checkpoint when `--no-transform` is given:
```bash
cd ../3dgrut
python ../aria/export_3dgrut_nurec.py \
-c runs/first_floor_3dgut_mcmc_ppisp/<run>/ckpt_last.pt \
-o runs/first_floor_3dgut_mcmc_ppisp/<run>/export_nurec_world.usdz \
--format nurec --no-transform
```
The standalone exporter also exports the test cameras (`camera_0_val`) in addition to the training cameras. The warnings `Could not open asset @ppisp_usd_spg_auto.usda@` during the export are harmless: the file is added to the USDZ afterwards.

On the first run, 3DGRUT compiles its CUDA kernels (a few minutes), later runs use the cache in `~/.cache/torch_extensions`.

## TSDF fusion with DA3 + VDBFusion
### 1. Environments
[Depth Anything v3](https://github.com/ByteDance-Seed/Depth-Anything-3.git) is installed in the `aria` environment and [VDBFusion](https://github.com/PRBonn/vdbfusion.git) in the `aria-vdbfusion` environment (see [Installation](#installation)). Run steps 2 and 3 with `conda activate aria` and step 4 with `conda activate aria-vdbfusion`.
### 2. (optional) Start DA3 backend acceleration:
```powershell
da3 backend --host 0.0.0.0 \
--port 8008 \
--allow-unauthenticated \
--gallery-dir workspace/gallery
```
### 3. Run DA3 inference on a COLMAP dataset with sliding window
```powershell
python run_da3_inference.py \
/path/to/colmap/dataset --sparse-subdir 0 \
--export-dir workspace/gallery/group_name/project \
--window-size 30 --overlap 10 \
--gallery-dir workspace/gallery \
--backend-url http://localhost:8008
```
The script then runs DA3 inference with a simple sliding window for depth and uncertainty prediction. For frames that are in more than one window, it keeps only the last predicted depth map.
\
Important hyperparameters: 
- `window-size`: number of frames in the sliding window. Around 30-40 frames per window are supported on NVIDIA RTX 4090 with 24 GB VRAM. 
- `overlap`: increment between sliding windows

See all parameters by running `python run_da3_inference.py --help`

### 4. Run TSDF fusion on depth maps
```powershell
python run_tsdf_fusion.py \
--input workspace/gallery/group_name/project \
--output /path/to/output.ply \ 
--confidence-threshold 6.0 \
--voxel-size 0.04
```
Important hyperparameters: 
- `confidence-threshold`: minimum depth confidence to be used in TSDF fusion.
- `debug`: visualize confidence-based depth filtering for a few frames. This helps tuning the `confidence-threshold` parameter.
- `voxel-size`: voxel dimensions in the resulting mesh
- `sdf-trunc`: TSDF truncation distance in depth units

See all parameters by running `python run_tsdf_fusion.py --help`

The script reads the camera intrinsics, extrinsics and image dimensions from the COLMAP dataset and transforms the intrinsics to 504x504 resolution (used by DA3). Then, it performs TSDF integration and mesh extraction.
### 5. (optional) Convert .ply mesh to .usdz or .usda file for Isaac Sim
**Installation**: The *usd-core* Python library is included in the `aria` environment. Alternatively, you can run the script with the NVIDIA Isaac Sim interpreter (usually under /home/username/isaacsim/python.sh)
```powershell
python ply_to_usd.py /path/to/mesh.ply \
--gaussian-usdz /path/to/3dgrut_run/export_nurec_world.usdz \
-o output_mesh.usd \
--scene-output output_scene.usda
```

Important hyperparameters: 
- `gaussian-usdz`: (optional) - output file of the gaussian splatting scene (usdz file). If used, the script will apply the same transformation to the mesh, so that they are aligned in the .usda scene. Use a NuRec export without the normalizing transform (see [NuRec export and the normalizing transform](#nurec-export-and-the-normalizing-transform)), otherwise the mesh gets a projective transform.
- `-o`: output file of the converted .ply file (.usdz file)
- `scene-output` (optional, only when using `gaussian-usdz`) - combined .usda output scene with Gaussian .usdz and mesh .usdz with the same transformation