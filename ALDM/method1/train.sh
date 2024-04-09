accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --use_deepspeed --mixed_precision "fp16" \
 train.py \
  --pretrained_model_name_or_path="./model/unet/" \
  --train_batch_size=4 \
  --vae_model_path="./model/vae/" \
  --data_root_path="./dataset/VITON-HD" \
  --image_encoder_path="./model/dinov2-G/" \
  --mixed_precision="fp16" \
  --ratio=4 \
  --learning_rate=1e-5 \
  --output_dir="./simpleOnUnet" \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs=6000 \
  --checkpointing_steps=10000 \
  --adam_weight_decay=0.01 
