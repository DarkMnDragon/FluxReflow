export MODEL_NAME="black-forest-labs/FLUX.1-dev"
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
<<<<<<< HEAD
  --resolution=1024 \
=======
  --resolution=512 \
>>>>>>> 994d251 (Optimize prior generation VRAM usage)
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-5 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1500 \
  --validation_prompt="A photo of sks dog running on the beach, oil painting" \
  --validation_epochs=30 \
  --num_class_images=200 \
  --seed="0" \
<<<<<<< HEAD
  --rank=512
=======
  --rank=8
>>>>>>> 994d251 (Optimize prior generation VRAM usage)
  --push_to_hub