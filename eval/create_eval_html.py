import os
import sys
import math
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz.viz import ImageGridVisualizer

def generate_trajectories():
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

input_dir = "/home/ubuntu/PE-Field/eval/images"
htmls_dir = "/home/ubuntu/PE-Field/mount/mrartemev/visualizations/htmls"
os.makedirs(htmls_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

trajectories = generate_trajectories()
images = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')], 
                key=lambda x: int(Path(x).stem))

directional_html = os.path.join(htmls_dir, f"baseline_eval_directional_{timestamp}.html")
viz_dir = ImageGridVisualizer(directional_html)

directional_trajs = ["shift_left", "shift_right", "up", "down"]
column_names = ["Input (0,0)", "Step 1", "Step 2", "Step 3"]
viz_dir.add_set_column_names(column_names)

for traj_name in directional_trajs:
    traj_points = trajectories[traj_name]
    
    for img in images:
        img_name = Path(img).stem
        description = f"Image {img_name} - {traj_name}"
        
        image_urls = []
        image_urls.append(f"gs://morphic-research/mrartemev/visualizations/pefields/inputs_29/{img}")
        
        for step_idx, (phi, theta) in enumerate(traj_points):
            url = f"gs://morphic-research/mrartemev/visualizations/pefields/baseline/baseline_{img_name}_{traj_name}_{step_idx:02d}_phi{phi:+04d}_theta{theta:+03d}.png"
            image_urls.append(url)
        
        viz_dir.add_row_urls(description, image_urls)

viz_dir.render()
print(f"Directional HTML created: {directional_html}")

circular_html = os.path.join(htmls_dir, f"baseline_eval_circular_{timestamp}.html")
viz_circ = ImageGridVisualizer(circular_html)

circular_trajs = ["circle_10", "circle_20"]
column_names = ["Input (0,0)"] + [f"Step {i+1}" for i in range(12)]
viz_circ.add_set_column_names(column_names)

for traj_name in circular_trajs:
    traj_points = trajectories[traj_name]
    
    for img in images:
        img_name = Path(img).stem
        description = f"Image {img_name} - {traj_name}"
        
        image_urls = []
        image_urls.append(f"gs://morphic-research/mrartemev/visualizations/pefields/inputs_29/{img}")
        
        for step_idx, (phi, theta) in enumerate(traj_points):
            url = f"gs://morphic-research/mrartemev/visualizations/pefields/baseline/baseline_{img_name}_{traj_name}_{step_idx:02d}_phi{phi:+04d}_theta{theta:+03d}.png"
            image_urls.append(url)
        
        viz_circ.add_row_urls(description, image_urls)

viz_circ.render()
print(f"Circular HTML created: {circular_html}")

