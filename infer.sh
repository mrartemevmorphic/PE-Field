python ./infer_viewchanger_single_v2.py \
  --moge_checkpoint_path "./moge-2-vitl-normal/model.pt" \
  --transformer_checkpoint_path "./checkpoints" \
  --flux_kontext_path "./FLUX.1-Kontext-dev" \
  --input_image "image_path_or_dir" \
  --output_dir "outputs" \
  --phi -5 --theta 5