python \
 test.py \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --vae_model_path="./model/vae" \
  --imageEncoder_model_path="model/dinov2-G" \
  --scheduler_path="./model/scheduler" \
  --data_root_path="dataset/VITON-HD" \
  --model_train_choice="all_trained/" \
  --model_trained_steps=12500 \
  --num_samples=4 \
  --guidance_scale=2.0 \
  --image_reserve_path="./result" 