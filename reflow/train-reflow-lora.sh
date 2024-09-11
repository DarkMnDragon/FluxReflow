export MODEL_NAME="/root/autodl-tmp/FLUX-dev"
export OUTPUT_DIR="/root/autodl-tmp/lora_ckpt/3reflow-dev-various"

accelerate launch train_reflow_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --reflow_data_dir="/root/autodl-tmp/data/2reflow_various_prompts" \
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
  --validation_prompt="A gothic queen vampiress with dark blue hair and crimson red eyes: Her sensuous white skin gleams in the atmospheri
c, dense fog, creating an epic and dramatic mood. This hyper-realistic portrait is filled with morbid beauty, from her gothic
 attire to the intense lighting that highlights every intricate detail. The scene combines glamour with dark, mysterious elem
ents, blending fantasy and horror in a visually stunning way." \
  --validation_epochs=1 \
  --seed="0" \
  --rank=256
  --push_to_hub