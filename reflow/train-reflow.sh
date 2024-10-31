export MODEL_NAME="/root/autodl-tmp/FLUX-dev"
export OUTPUT_DIR="/root/autodl-tmp/lora_ckpt/2rf-various-prompts-full-finetune-journeydb-v3-alpha=4.0-ema"

accelerate launch train_reflow_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --reflow_data_root="/root/autodl-tmp/data/1rf-journeydb_v3-guidance=3.5" \
  --training_t_dist="u_shape" \
  --mixed_precision="bf16" \
  --resolution="1024*1024" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="Adafactor" \
  --learning_rate=1e-5 \
  --use_ema \
  --adam_weight_decay=0.01 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=20000 \
  --validation_prompt="A cat holding a sign 'hello world'" \
  --num_validation_inference_steps=8 \
  --validation_epochs=1 \
  --seed="0" \
  --gradient_checkpointing \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=2