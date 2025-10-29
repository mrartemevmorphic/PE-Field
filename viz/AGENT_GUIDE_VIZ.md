# Visualization Tools Guide

## Overview

The `viz/` directory contains tools for creating HTML visualization reports of PE-Field evaluation results. These tools help compare novel view synthesis outputs across different camera angles and image categories by generating organized HTML grids with embedded images.

## Tools

### 1. viz.py - ImageGridVisualizer

Creates HTML tables displaying images in a grid format with descriptions. Each row contains a text description and multiple images/videos side-by-side for easy comparison.

### 2. viztools.py - GCS Upload & Combine CLI

Command-line tool for uploading HTML reports to Google Cloud Storage and merging multiple reports into unified visualizations.

## viz.py Usage

### Basic Example

```python
from viz import ImageGridVisualizer
from PIL import Image

viz = ImageGridVisualizer("results.html")

viz.add_row(
    description="Image 001 | phi=0°",
    images=[
        "path/to/original.png",
        "path/to/warped.png", 
        "path/to/output.png"
    ]
)

viz.render()
```

### ImageGridVisualizer Class

**Constructor:**
```python
viz = ImageGridVisualizer(filename: str)
```
- Creates HTML file and asset folder (e.g., `results.html` → `results/` folder)
- Asset folder stores embedded images

**Methods:**

`add_row(description: str, images: list[Image.Image | str | None])`
- Adds row with description and local images
- `images` can be PIL Image objects or file paths
- PIL images are saved to asset folder automatically

`add_row_urls(description: str, image_urls: list[str])`
- Adds row with GCS URLs
- URLs must start with `gs://`
- Automatically converts to `https://storage.cloud.google.com/` format

`add_row_video(description: str, video_urls: list[str])`
- Adds row with video URLs (mp4, avi, mkv, mov)
- Embeds HTML5 video player with controls

`add_set_column_names(column_names: list[str])`
- Sets custom column headers (default: "Image 1", "Image 2", ...)

`render()`
- Generates final HTML file
- Call once after adding all rows

### PE-Field Integration Example

```python
import os
from pathlib import Path
from viz import ImageGridVisualizer

output_base = "/home/ubuntu/PE-Field/eval/outputs"
categories = ["real_world", "anime"]
phi_angles = [-240, -180, -120, -60, 0, 60, 120, 180, 240]

for category in categories:
    viz = ImageGridVisualizer(f"eval/{category}_results.html")
    viz.add_set_column_names(["Original", "Warped", "Output"])
    
    category_dir = f"{output_base}/{category}"
    
    for img_folder in sorted(os.listdir(category_dir)):
        img_path = os.path.join(category_dir, img_folder)
        if not os.path.isdir(img_path):
            continue
            
        for phi in phi_angles:
            desc = f"{img_folder} | phi={phi}° theta=0°"
            
            images = [
                os.path.join(img_path, f"{img_folder}_original.png"),
                os.path.join(img_path, f"{img_folder}_warped.png"),
                os.path.join(img_path, f"{img_folder}_output.png")
            ]
            
            viz.add_row(desc, images)
    
    viz.render()
    print(f"Generated {category}_results.html")
```

### Multi-Angle Comparison

```python
viz = ImageGridVisualizer("multi_angle_comparison.html")

column_names = [f"φ={phi}°" for phi in [-240, -180, -120, -60, 0, 60, 120, 180, 240]]
viz.add_set_column_names(column_names)

for img_name in ["img001", "img002", "img003"]:
    outputs = [
        f"outputs/real_world/{img_name}/phi_{phi}_output.png" 
        for phi in [-240, -180, -120, -60, 0, 60, 120, 180, 240]
    ]
    viz.add_row(f"Image: {img_name}", outputs)

viz.render()
```

## viztools.py Usage

CLI tool with two commands: `upload_html_and_assets` and `combine`.

### Installation Requirements

```bash
pip install click beautifulsoup4 google-cloud-storage
```

Ensure GCS credentials are configured:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### Command 1: upload_html_and_assets

Uploads HTML file and its asset folder to GCS, converting relative links to absolute URLs.

**Syntax:**
```bash
python viz/viztools.py upload_html_and_assets \
    <html_file_path> \
    <bucket_name> \
    --destination-folder <folder> \
    --public  # or --private
```

**Example:**
```bash
python viz/viztools.py upload_html_and_assets \
    eval/real_world_results.html \
    ml-team-evaluation \
    --destination-folder pe_field_eval \
    --public
```

**What it does:**
1. Reads `real_world_results.html` and `real_world_results/` folder
2. Modifies HTML to use absolute GCS URLs
3. Uploads HTML to `gs://ml-team-evaluation/pe_field_eval/real_world_results.html`
4. Uploads assets to `gs://ml-team-evaluation/pe_field_eval/real_world_results/`
5. Makes files publicly accessible
6. Prints public URL for sharing

### Command 2: combine

Merges multiple HTML reports by combining their table rows.

**Syntax:**
```bash
python viz/viztools.py combine \
    <input_file_1> <input_file_2> ... \
    <output_file> \
    --destination-folder <folder> \
    --append "label1" --append "label2" \
    --public  # or --private
```

**Example:**
```bash
python viz/viztools.py combine \
    https://storage.cloud.google.com/ml-team-evaluation/pe_field_eval/real_world_results.html \
    https://storage.cloud.google.com/ml-team-evaluation/pe_field_eval/anime_results.html \
    combined_results.html \
    --destination-folder pe_field_eval \
    --append "Real World" --append "Anime" \
    --public
```

**What it does:**
1. Fetches HTML from GCS URLs (or local files)
2. Extracts table rows from each file
3. Prepends category labels to description column
4. Merges rows in round-robin fashion
5. Uploads combined HTML to GCS
6. Generates unique filename with UUID

## Complete Workflow Example

### Step 1: Run Evaluation
```bash
cd /home/ubuntu/PE-Field/eval
python run_eval.py
```

### Step 2: Generate HTML Reports
```python
import os
from viz.viz import ImageGridVisualizer

phi_angles = [-240, -180, -120, -60, 0, 60, 120, 180, 240]

def create_category_report(category):
    viz = ImageGridVisualizer(f"eval/{category}_eval.html")
    viz.add_set_column_names(["Original", "Warped", "Output"])
    
    output_dir = f"/home/ubuntu/PE-Field/eval/outputs/{category}"
    
    for img_folder in sorted(os.listdir(output_dir)):
        folder_path = os.path.join(output_dir, img_folder)
        if not os.path.isdir(folder_path):
            continue
        
        for phi in phi_angles:
            desc = f"{img_folder} | φ={phi}° θ=0°"
            viz.add_row(desc, [
                os.path.join(folder_path, f"{img_folder}_original.png"),
                os.path.join(folder_path, f"{img_folder}_warped.png"),
                os.path.join(folder_path, f"{img_folder}_output.png")
            ])
    
    viz.render()
    return f"eval/{category}_eval.html"

real_world_html = create_category_report("real_world")
anime_html = create_category_report("anime")
print(f"Generated: {real_world_html}, {anime_html}")
```

### Step 3: Upload to GCS
```bash
cd /home/ubuntu/PE-Field

python viz/viztools.py upload_html_and_assets \
    eval/real_world_eval.html \
    ml-team-evaluation \
    --destination-folder pe_field_results/2025_10_29 \
    --public

python viz/viztools.py upload_html_and_assets \
    eval/anime_eval.html \
    ml-team-evaluation \
    --destination-folder pe_field_results/2025_10_29 \
    --public
```

### Step 4: Combine Reports
```bash
python viz/viztools.py combine \
    https://storage.cloud.google.com/ml-team-evaluation/pe_field_results/2025_10_29/real_world_eval.html \
    https://storage.cloud.google.com/ml-team-evaluation/pe_field_results/2025_10_29/anime_eval.html \
    combined_all.html \
    --destination-folder pe_field_results/2025_10_29 \
    --append "Real World" --append "Anime" \
    --public
```

## Output Structure

After running visualization scripts:

```
eval/
├── real_world_eval.html
├── real_world_eval/           # Asset folder
│   ├── image_0.png
│   ├── image_1.png
│   └── ...
├── anime_eval.html
└── anime_eval/                # Asset folder
    ├── image_0.png
    └── ...
```

## Dependencies

Add to `requirements.txt` if not present:
```
click>=8.0.0
beautifulsoup4>=4.10.0
google-cloud-storage>=2.0.0
Pillow>=9.0.0
```

Install:
```bash
source ./envs/pe_field/bin/activate
pip install click beautifulsoup4 google-cloud-storage
```

## Notes

- HTML files auto-create asset folders with same base name
- GCS URLs require `gs://` prefix for `add_row_urls()`
- Images display at max 250px height (preserves aspect ratio)
- Tables use responsive styling (10px font, word-wrap enabled)
- Combined reports interleave rows from all inputs
- UUID appended to combined output filenames to prevent overwrites
- Video formats supported: mp4, avi, mkv, mov
- Empty cells rendered when row has fewer images than max columns

## Common Use Cases

**Compare warping quality across angles:**
```python
viz = ImageGridVisualizer("warp_quality.html")
for phi in range(-240, 241, 60):
    viz.add_row(f"φ={phi}°", [f"outputs/img001/warped_phi{phi}.png"])
viz.render()
```

**Show auto-regressive degradation:**
```python
viz = ImageGridVisualizer("ar_progression.html")
for step in range(10):
    viz.add_row(f"AR Step {step}", [f"outputs/ar/step{step}_output.png"])
viz.render()
```

**Upload batch results:**
```bash
for html in eval/*.html; do
    python viz/viztools.py upload_html_and_assets \
        "$html" ml-team-evaluation \
        --destination-folder batch_$(date +%Y%m%d) --public
done
```

End of Answer

