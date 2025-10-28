import os
import subprocess
from pathlib import Path
from PIL import Image
import glob

moge_checkpoint = "/home/ubuntu/PE-Field/moge-2-vitl-normal/model.pt"
transformer_checkpoint = "/home/ubuntu/PE-Field/checkpoints"
flux_kontext = "/home/ubuntu/PE-Field/FLUX.1-Kontext-dev"
output_base = "/home/ubuntu/PE-Field/eval/outputs"

categories = {
    "real_world": "/home/ubuntu/PE-Field/eval/real_world",
    "anime": "/home/ubuntu/PE-Field/eval/anime"
}

phi_angles = [-240, -180, -120, -60, 0, 60, 120, 180, 240]
theta = 0

total_runs = 0
for category, input_dir in categories.items():
    images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    for img in images:
        img_path = os.path.join(input_dir, img)
        img_name = Path(img).stem
        
        for i, phi in enumerate(phi_angles):
            output_dir = f"{output_base}/{category}/_temp"
            os.makedirs(output_dir, exist_ok=True)
            
            cmd = [
                "/home/ubuntu/PE-Field/envs/pe_field/bin/python", "/home/ubuntu/PE-Field/infer_viewchanger_single_v2.py",
                "--moge_checkpoint_path", moge_checkpoint,
                "--transformer_checkpoint_path", transformer_checkpoint,
                "--flux_kontext_path", flux_kontext,
                "--input_image", img_path,
                "--output_dir", output_dir,
                "--phi", str(phi),
                "--theta", str(theta)
            ]
            
            print(f"\n{'='*80}")
            print(f"Run {total_runs + 1}/200: {category}/{img} - View {i+1}/10 (phi={phi}°, theta={theta}°)")
            print(f"{'='*80}")
            
            result = subprocess.run(cmd, capture_output=False)
            
            if result.returncode != 0:
                print(f"ERROR: Failed to process {img} at phi={phi}")
            else:
                original_imgs = glob.glob(os.path.join(output_dir, "*_original.png"))
                warped_imgs = glob.glob(os.path.join(output_dir, "*_warped.png"))
                output_imgs = glob.glob(os.path.join(output_dir, "*_output.png"))
                
                if original_imgs and warped_imgs and output_imgs:
                    original_img = original_imgs[0]
                    warped_img = warped_imgs[0]
                    output_img = output_imgs[0]
                    img1 = Image.open(original_img)
                    img2 = Image.open(warped_img)
                    img3 = Image.open(output_img)
                    
                    total_width = img1.width + img2.width + img3.width
                    max_height = max(img1.height, img2.height, img3.height)
                    
                    combined = Image.new('RGB', (total_width, max_height))
                    combined.paste(img1, (0, 0))
                    combined.paste(img2, (img1.width, 0))
                    combined.paste(img3, (img1.width + img2.width, 0))
                    
                    combined_filename = f"harsh_{i:02d}_phi{phi:+04d}_theta{theta:+03d}.png"
                    combined_path = os.path.join(output_base, category, img_name)
                    os.makedirs(combined_path, exist_ok=True)
                    
                    combined.save(os.path.join(combined_path, combined_filename))
                    
                    os.remove(original_img)
                    os.remove(warped_img)
                    os.remove(output_img)
            
            total_runs += 1

print(f"\n\nCompleted {total_runs} inference runs!")
print(f"Results saved in: {output_base}")

