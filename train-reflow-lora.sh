export MODEL_NAME="/root/autodl-tmp/data/FLUX-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="/root/autodl-tmp/flux-reflow-lora"

accelerate launch train_reflow_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --img_root="/root/autodl-tmp/reflow/imgs"\
  --prompt_root="/root/autodl-tmp/reflow/prompt"\
  --prior_latent_root="/root/autodl-tmp/reflow/z_0"\
  --img_latent_root="/root/autodl-tmp/reflow/z_1"\
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=2000 \
  --validation_prompt="A photo of chihuahua dog running on the beach under the sunshine" \
  --validation_epochs=1 \
  --seed="0" \
  --rank=1024
  --push_to_hub