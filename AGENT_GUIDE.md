# PE-Field Repository Guide for AI Agent

## Overview
PE-Field (Positional Encoding Field) is a novel view synthesis system that uses depth-aware positional encodings with diffusion transformers (DiTs) to generate new camera viewpoints from a single image.

## Initial Setup Instructions

When setting up this repository from scratch, follow these steps:

### 1. Install gcsfuse
Install gcsfuse to mount GCS bucket:
```bash
curl -L https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v2.0.0/gcsfuse_2.0.0_amd64.deb -o /tmp/gcsfuse.deb
sudo dpkg -i /tmp/gcsfuse.deb
sudo apt-get install -f -y
```

### 2. Install and Authenticate gcloud CLI
```bash
sudo snap install google-cloud-cli --classic
gcloud auth login
gcloud auth application-default login
```

### 3. Mount GCS Directory
Mount the Google Cloud Storage bucket to `./mount`:
```bash
cd /home/ubuntu/PE-Field
mkdir -p mount
gcsfuse --implicit-dirs morphic-research ./mount
```

### 4. Restore Model Weights
Copy pre-downloaded model weights from mount directory instead of downloading (saves time, ~56GB total):

```bash
cd /home/ubuntu/PE-Field

# Copy FLUX.1-Kontext-dev weights (32GB)
cp -r ./mount/mrartemev/weights/FLUX.1-Kontext-dev ./

# Copy MoGe weights (1.3GB)
mkdir -p ./moge-2-vitl-normal
cp ./mount/mrartemev/weights/moge-2-vitl-normal/model.pt ./moge-2-vitl-normal/

# Copy transformer weights (23GB)
cp -r ./mount/mrartemev/weights/checkpoints/transformer/* ./checkpoints/transformer/
```

### 5. Install python3-venv
```bash
sudo apt install -y python3.10-venv
```

### 6. Create Virtual Environment
```bash
cd /home/ubuntu/PE-Field

# Create virtual environment
python3 -m venv ./envs/pe_field

# Activate environment
source ./envs/pe_field/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 7. Verify Setup
Check that all components are in place:
```bash
ls -lh FLUX.1-Kontext-dev/
ls -lh moge-2-vitl-normal/model.pt
ls -lh checkpoints/transformer/
which python  # Should point to ./envs/pe_field/bin/python
```

### 8. Configure Git (Optional)
```bash
git config --global user.email "maksim.artemev@morphic.com"
git config --global user.name "Max"
```

## Architecture

### Core Components
1. **MoGe** - Depth estimation model (v2-vitl-normal)
2. **FLUX.1-Kontext** - Base diffusion pipeline 
3. **Custom Transformer** - Fine-tuned transformer weights for view synthesis
4. **Position Encoding System** - Multi-scale 3D positional encodings

### Key Files

#### Main Inference
- `infer_viewchanger_single_v2.py` - Main inference script for single/batch image processing
- Takes input image → estimates depth → generates novel view

#### Utilities  
- `pvd_utils.py` - Camera pose manipulation, coordinate transforms, trajectory generation
- `read_write_model.py` - COLMAP model I/O utilities

#### Evaluation
- `eval/run_eval.py` - Batch evaluation on categories (real_world, anime)
- `eval/run_eval_ar.py` - Auto-regressive evaluation (chain multiple view changes)

#### Modified Diffusers
- `diffusers/` - Custom diffusers library with FluxKontextPipeline modifications

## Pipeline Flow

1. **Input Processing**
   - Load image (single file or directory)
   - Resize to nearest FLUX Kontext resolution (aspect ratio preserved)

2. **Depth Estimation** (MoGe)
   - Infer depth map from input image
   - Extract camera intrinsics (focal length, principal point)
   - Generate 3D point map in camera coordinates

3. **View Transform**
   - Define target camera pose using phi (azimuth) and theta (elevation) angles
   - Transform point cloud to object-centric coordinates
   - Generate new camera-to-world matrix for target view

4. **Position Encoding**
   - Project 3D points to target view pixel coordinates
   - Create normalized depth values
   - Generate multi-scale position encodings (3 levels: 1/16, 1/8, 1/4 resolution)
   - Split into 2x2 local grids for hierarchical encoding

5. **Forward Splatting**
   - Warp source image to target view using projected coordinates
   - Accumulate RGB values with depth-based weighting

6. **Diffusion Generation**
   - Feed warped image + position encodings to FluxKontextPipeline
   - Generate refined output with geometric consistency
   - Prompt: "Ensure that the picture does not change in any way"

## Key Parameters

### Camera Angles
- `phi` - Azimuth angle (horizontal rotation, degrees)
- `theta` - Elevation angle (vertical rotation, degrees)  
- `r` - Radius offset (default: 0)

### Coordinate Systems
- OpenCV: Right-Down-Forward (used internally)
- NeRF/OpenGL: Right-Up-Back (conversion functions in pvd_utils.py)
- COLMAP: Right-Down-Forward

### Resolution Levels
Preferred Kontext resolutions (aspect ratio matched):
- 672×1568, 688×1504, 720×1456, 752×1392, 800×1328
- 832×1248, 880×1184, 944×1104, 1024×1024
- 1104×944, 1184×880, 1248×832, 1328×800
- 1392×752, 1456×720, 1504×688, 1568×672

## Usage Examples

### Single Image
```bash
python infer_viewchanger_single_v2.py \
  --moge_checkpoint_path "./moge-2-vitl-normal/model.pt" \
  --transformer_checkpoint_path "./checkpoints" \
  --flux_kontext_path "./FLUX.1-Kontext-dev" \
  --input_image "path/to/image.jpg" \
  --output_dir "outputs" \
  --phi 30 --theta -15
```

### Batch Directory
```bash
python infer_viewchanger_single_v2.py \
  --moge_checkpoint_path "./moge-2-vitl-normal/model.pt" \
  --transformer_checkpoint_path "./checkpoints" \
  --flux_kontext_path "./FLUX.1-Kontext-dev" \
  --input_image "path/to/image_dir/" \
  --output_dir "outputs" \
  --phi 30 --theta 0
```

### Evaluation
```bash
cd eval
python run_eval.py  # Run on all test images with multiple views
python run_eval_ar.py  # Auto-regressive view synthesis
```

## Output Files

For each input, generates:
- `{name}_original.png` - Resized input image
- `{name}_warped.png` - Forward-splatted warped image  
- `{name}_output.png` - Final diffusion-refined output

## Dependencies

Key packages (see requirements.txt):
- torch, diffusers
- OpenEXR, Imath
- transformers
- PIL, opencv-python
- numpy, scipy

## Model Weights

Required downloads (not in git):
1. FLUX.1-Kontext-dev (32GB) → `./FLUX.1-Kontext-dev/`
2. MoGe weights (1.3GB) → `./moge-2-vitl-normal/model.pt`
3. PE-Field transformer (23GB) → `./checkpoints/transformer/`

## Important Functions

### pvd_utils.py
- `world_point_to_obj()` - Transform points to object-centric frame
- `random_sample_poses()` - Generate camera trajectory
- `sphere2pose()` - Convert spherical angles to camera pose
- `forward_splat()` - Image warping via splatting
- `split_into_2x2_local_grids()` - Hierarchical position encoding

### infer_viewchanger_single_v2.py
- `save_tensor_image()` - Save tensor as PNG
- `project_to_pixel()` - 3D to 2D projection
- `exr_loader_any()` - Load EXR files (if needed)

## Evaluation Setup

- Sample images in `eval/real_world/` and `eval/anime/`
- Output structure: `eval/outputs_{n}/{category}/{image_name}/`
- Flattened AR results: `eval/outputs_2/flat/`

## Notes for Agent

- All paths are hardcoded in eval scripts - adjust if needed
- Virtual env at `./envs/pe_field/`
- Large model files (~56GB total) should stay in .gitignore
- Output directories are generated at runtime
- MoGe uses resolution_level=9 by default
- FLUX uses bfloat16 precision
- Forward splatting handles occlusions via accumulation

