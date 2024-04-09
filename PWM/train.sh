accelerate launch --gpu_ids 0,1,2,3,4,5,6,7 --use_deepspeed --num_processes 8   \
  stage1_train_prior_model.py \
  --pretrained_model_name_or_path="./pretrained_model" \
  --output_dir="checkpoints" \
  --data_root_path="../try_on/dataset/VITON-HD" \
  --train_batch_size=128 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=100000 \
  --noise_offset=0.1 \
  --enable_xformers_memory_efficient_attention \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=2000 \
  --checkpointing_steps=1000 \
  --num_layers=10