export MODEL_NAME="/root/autodl-tmp/FLUX-dev"
export OUTPUT_DIR="/root/autodl-tmp/lora_ckpt/2rf-general-cont"

accelerate launch train_reflow_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --reflow_data_dir="/root/autodl-tmp/data/1rf_dev_various_prompts" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1.0 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=30000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=16 \
  --resume_from_checkpoint="/root/autodl-tmp/lora_ckpt/2rf-general-cont/checkpoint-5000" \
  --validation_prompt="Oriental Pearl Tower standing tall against a stormy Shanghai skyline. Dark clouds swirl above, illuminated by flashes of lightning cutting through the sky. The tower’s futuristic spheres glow faintly, reflecting off the rain-slicked streets below. Wind sweeps across the city, bending trees and causing ripples on the river. The lightning casts sharp shadows, adding depth and intensity to the scene. Use cinematic lighting and a high level of detail in the architecture and storm clouds. Dramatic and moody atmosphere." \
  --validation_inference_steps=8 \
  --lora_warmup_steps=0 \
  --validation_epochs=1 \
  --seed="0" \
  --rank=128 