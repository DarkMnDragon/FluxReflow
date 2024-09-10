export MODEL_NAME="/root/autodl-tmp/data/FLUX-dev"
export INSTANCE_DIR="dog"
export OUTPUT_DIR="/root/autodl-tmp/flux-lora-dreambooth"
export CLASS_DIR="/root/autodl-tmp/data/dog-prior"

accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --mixed_precision="bf16" \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --prior_generation_precision "bf16" \
  --instance_prompt="A high-resolution photo of sks dog" \
  --class_prompt="A high-resolution photo of a dog" \
  --resolution=512 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --validation_prompt="A photo of sks dog running on the beach, oil painting" \
  --validation_epochs=30 \
  --num_class_images=200 \
  --seed="0" \
  --rank=1024
  --push_to_hub