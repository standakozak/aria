# Digital twin reconstruction with Project Aria Glasses
## TSDF fusion with DA3 + VDBFusion
### 1. Install [Depth Anything v3](https://github.com/ByteDance-Seed/Depth-Anything-3.git) and [VDBFusion](https://github.com/PRBonn/vdbfusion.git).
Use the official installation guidelines. Please use separate environments because VDBFusion only supports Numpy==1.26.0
### 2. (optional) Start DA3 backend acceleration:
```powershell
da3 backend --host 0.0.0.0 \
--port 8008 \
--allow-unauthenticated \
--gallery-dir workspace/gallery
```
### 3. Run DA3 inference on a COLMAP dataset with sliding window
```powershell
python vdb_fusion/da3_colmap.py \
/path/to/colmap/dataset \
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

See all parameters by running `python vdb_fusion/da3_colmap.py --help`

### 4. Run TSDF fusion on depth maps
```powershell
python vdb_fusion/fuse_da3_vdb.py \
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

See all parameters by running `python vdb_fusion/fuse_da3_vdb.py --help`

The script reads the camera intrinsics, extrinsics and image dimensions from the COLMAP dataset and transforms the intrinsics to 504x504 resolution (used by DA3). Then, it performs TSDF integration and mesh extraction.
### 5. (optional) Convert .ply mesh to .usdz or .usda file for Isaac Sim
**Installation**: In order to run this script, the *usd-core* Python library has to be installed in the environment. Alternatively, you can run the script with the NVIDIA Isaac Sim interpreter (usually under /home/username/isaacsim/python.sh)
```powershell
python vdb_fusion/ply_to_usd.py /path/to/mesh.ply \
--gaussian-usdz /path/to/3dgrut_nurec_export.usdz \
-o output_mesh.usd \
--scene-output output_scene.usda
```

Important hyperparameters: 
- `gaussian-usdz`: (optional) - output file of the gaussian splatting scene (usdz file). If used, the script will apply the same transformation to the mesh, so that they are aligned in the .usda scene
- `-o`: output file of the converted .ply file (.usdz file)
- `scene-output` (optional, only when using `gaussian-usdz`) - combined .usda output scene with Gaussian .usdz and mesh .usdz with the same transformation