export MODEL_NAME="/root/autodl-tmp/FLUX-dev"
export OUTPUT_DIR="/root/autodl-tmp/lora_ckpt/2rf-tarot"

accelerate launch train_reflow_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --reflow_data_dir="/root/autodl-tmp/data/1rf_tarot" \
  --mixed_precision="bf16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=5000 \
  --validation_prompt="a person giving a ted talk on a TED stage with the TED logo, 'the speaker' in the style of TOK a trtcrd, tarot style" \
  --lora_warmup_steps=1000 \
  --validation_epochs=1 \
  --seed="0" \
  --rank=64
  --push_to_hub