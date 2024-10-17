export MODEL_NAME="/root/autodl-tmp/FLUX-dev"
export OUTPUT_DIR="/root/autodl-tmp/lora_ckpt/dreambooth-dog-dynamic-backward"

accelerate launch train_dreamreflow_lora_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --prior_reflow_data_root="/root/autodl-tmp/data/1rf_dog" \
  --instance_data_root="/root/autodl-tmp/data/dog-instance" \
  --use_reflow_prior_loss \
  --backward_reflow_threshold=1000 \
  --backward_update_steps=100 \
  --num_inversion_steps=100 \
  --mixed_precision="bf16" \
  --resolution="1024*1024" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --prior_loss_weight=2. \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --validation_prompt="A photo of a ohwx sitting in a blue basket, high resolution" \
  --num_validation_inference_steps=10 \
  --validation_epochs=1 \
  --seed="0" \
  --rank=128