 accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --num_processes 8 --use_deepspeed --mixed_precision "fp16" \
  my_train.py \
  --pretrained_model_name_or_path="./model/" \
  --train_batch_size=8 \
  --vae_model_path="./model/vae/" \
  --data_root_path="./dataset/VITON-HD" \
  --image_encoder_dinov_path="./model/dinov2-G/" \
  --mixed_precision="fp16" \
  --ratio=2 \
  --output_dir="checkpoint" \
  --learning_rate=1e-4 \
  --enable_xformers_memory_efficient_attention \
  --num_train_epochs=50000 \
  --checkpointing_steps=10000 \
  --adam_weight_decay=0.01 \
  --max_train_steps=200000 \


 
