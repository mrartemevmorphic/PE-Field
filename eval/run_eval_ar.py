import os
import subprocess
from pathlib import Path
from PIL import Image
import glob
import shutil

moge_checkpoint = "/home/ubuntu/PE-Field/moge-2-vitl-normal/model.pt"
transformer_checkpoint = "/home/ubuntu/PE-Field/checkpoints"
flux_kontext = "/home/ubuntu/PE-Field/FLUX.1-Kontext-dev"
output_base = "/home/ubuntu/PE-Field/eval/outputs_2"
output_flat_base = "/home/ubuntu/PE-Field/eval/outputs_2/flat"
os.makedirs(output_flat_base, exist_ok=True)

input_dir = "/home/ubuntu/PE-Field/eval/real_world"
input_name = "real_world"

prefix = "AR"

num_phi_angles = 5
theta = 0

total_runs_needed = num_phi_angles * 10 #len(images)
total_runs = 0

images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])

for img in images:
    img_path = os.path.join(input_dir, img)
    img_name = Path(img).stem
    
    last_result_path = None
        
    for i in range(num_phi_angles):
        phi = 6
        actual_phi = phi * (i + 1)
        output_dir = f"{output_base}/{prefix}_{input_name}_phi{actual_phi:03d}_theta{theta:03d}"
        os.makedirs(output_dir, exist_ok=True)
        
        if last_result_path is None:
            last_result_path = img_path
            
        ext = Path(last_result_path).suffix
        new_input_path = os.path.join(output_dir, f"input_image{ext}")
        shutil.copy(last_result_path, new_input_path)
        
        cmd = [
            "/home/ubuntu/PE-Field/envs/pe_field/bin/python", "/home/ubuntu/PE-Field/infer_viewchanger_single_v2.py",
            "--moge_checkpoint_path", moge_checkpoint,
            "--transformer_checkpoint_path", transformer_checkpoint,
            "--flux_kontext_path", flux_kontext,
            "--input_image", new_input_path,
            "--output_dir", output_dir,
            "--phi", str(phi),
            "--theta", str(theta)
        ]        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode != 0:
            print(f"ERROR: Failed to process {img} at phi={phi}")
            continue
        
        original_img = glob.glob(os.path.join(output_dir, "*_original.png"))[0]
        warped_img = glob.glob(os.path.join(output_dir, "*_warped.png"))[0]
        output_img = glob.glob(os.path.join(output_dir, "*_output.png"))[0]
        
        # since we are doing AR, we replace the original image with the output image
        last_result_path = output_img
        
        img1 = Image.open(original_img)
        img2 = Image.open(warped_img)
        img3 = Image.open(output_img)
            
        total_width = img1.width + img2.width + img3.width
        max_height = max(img1.height, img2.height, img3.height)
        
        combined = Image.new('RGB', (total_width, max_height))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (img1.width, 0))
        combined.paste(img3, (img1.width + img2.width, 0))

        combined_filename = f"{prefix}_{img_name}_phi{actual_phi:03d}_theta{theta:03d}.png"
        combined_path = os.path.join(output_flat_base, combined_filename)                
        combined.save(combined_path)
                    
        total_runs += 1

print(f"\n\nCompleted {total_runs} inference runs!")
print(f"Results saved in: {output_base}")

