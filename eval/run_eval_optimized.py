import os
import sys
import subprocess
import logging
import math
import time
from pathlib import Path
from PIL import Image
import glob
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from infer_viewchanger_wrapper import ViewChangerInference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

output_prefix = "baseline"

moge_checkpoint = "/home/ubuntu/PE-Field/moge-2-vitl-normal/model.pt"
transformer_checkpoint = "/home/ubuntu/PE-Field/checkpoints"
flux_kontext = "/home/ubuntu/PE-Field/FLUX.1-Kontext-dev"
output_base = "/home/ubuntu/PE-Field/eval/outputs"
final_output_dir = "/home/ubuntu/PE-Field/mount/mrartemev/visualizations/pefields/baseline"

input_dir = "/home/ubuntu/PE-Field/eval/images"

def generate_trajectories() -> Dict[str, List[Tuple[int, int]]]:
    trajectories = {}
    
    trajectories["shift_left"] = [(phi, 0) for phi in [-10, -20, -30]]
    trajectories["shift_right"] = [(phi, 0) for phi in [10, 20, 30]]
    trajectories["up"] = [(0, theta) for theta in [10, 20, 30]]
    trajectories["down"] = [(0, theta) for theta in [-10, -20, -30]]
    
    circle_20 = []
    for angle_deg in range(0, 360, 30):
        angle_rad = math.radians(angle_deg)
        phi = int(round(20 * math.cos(angle_rad)))
        theta = int(round(20 * math.sin(angle_rad)))
        if (phi, theta) != (0, 0):
            circle_20.append((phi, theta))
    trajectories["circle_20"] = circle_20
    
    circle_10 = []
    for angle_deg in range(0, 360, 30):
        angle_rad = math.radians(angle_deg)
        phi = int(round(10 * math.cos(angle_rad)))
        theta = int(round(10 * math.sin(angle_rad)))
        if (phi, theta) != (0, 0):
            circle_10.append((phi, theta))
    trajectories["circle_10"] = circle_10
    
    return trajectories

os.makedirs(final_output_dir, exist_ok=True)

trajectories = generate_trajectories()
total_steps = sum(len(traj) for traj in trajectories.values())
images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
total_expected_runs = len(images) * total_steps

logger.info(f"Starting evaluation with {len(trajectories)} trajectories and {len(images)} images")
logger.info(f"Expected total runs: {total_expected_runs}")
logger.info(f"Loading models once...")

inferencer = ViewChangerInference(moge_checkpoint, transformer_checkpoint, flux_kontext)

logger.info(f"Models loaded successfully. Starting inference...")

total_runs = 0
for img in images:
    img_path = os.path.join(input_dir, img)
    img_name = Path(img).stem
    
    for traj_name, traj_points in trajectories.items():
        for step_idx, (phi, theta) in enumerate(traj_points):
            combined_filename = f"{output_prefix}_{img_name}_{traj_name}_{step_idx:02d}_phi{phi:+04d}_theta{theta:+03d}"
            final_output_path = os.path.join(final_output_dir, f"{combined_filename}.png")
            
            if os.path.exists(final_output_path):
                logger.info(f"Skipping run {total_runs + 1}/{total_expected_runs}: {img} - {traj_name} step {step_idx+1}/{len(traj_points)} (already exists)")
                total_runs += 1
                continue
            
            output_dir = os.path.join(output_base, combined_filename)
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Run {total_runs + 1}/{total_expected_runs}: {img} - {traj_name} step {step_idx+1}/{len(traj_points)} (phi={phi:+d}°, theta={theta:+d}°)")
            
            start_time = time.time()
            success = inferencer.process_single(img_path, phi, theta, 0, output_dir)
            elapsed_time = time.time() - start_time
            
            if not success:
                logger.error(f"Failed to process {img} at phi={phi}, theta={theta} after {elapsed_time:.2f}s")
            else:
                original_imgs = glob.glob(os.path.join(output_dir, "*_original.png"))
                warped_imgs = glob.glob(os.path.join(output_dir, "*_warped.png"))
                output_imgs = glob.glob(os.path.join(output_dir, "*_output.png"))
                
                if original_imgs and warped_imgs and output_imgs:
                    img1 = Image.open(original_imgs[0])
                    img2 = Image.open(warped_imgs[0])
                    img3 = Image.open(output_imgs[0])
                    
                    total_width = img1.width + img2.width + img3.width
                    max_height = max(img1.height, img2.height, img3.height)
                    
                    combined = Image.new('RGB', (total_width, max_height))
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))
                    combined.paste(img3, (img1.width + img2.width, 0))
                    
                    img3.save(final_output_path)
                    logger.info(f"Successfully processed in {elapsed_time:.2f}s")
            
            total_runs += 1

logger.info(f"Completed {total_runs} inference runs!")
logger.info(f"Results saved in: {final_output_dir}")

