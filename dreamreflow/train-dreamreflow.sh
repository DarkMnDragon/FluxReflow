export MODEL_NAME="/root/autodl-tmp/FLUX-dev"
export OUTPUT_DIR="/root/autodl-tmp/lora_ckpt/dreambooth-dog-prior_reflow-no_backward-finetune"

accelerate launch train_dreamreflow_flux.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --prior_reflow_data_root="/root/autodl-tmp/data/1rf_dog" \
  --instance_data_root="/root/autodl-tmp/data/dog-instance" \
  --use_reflow_prior_loss \
  --prior_loss_weight=2. \
  --gaussian_pair_t_dist="logit_normal" \
  --reflow_pair_t_dist="u_shape" \
  --mixed_precision="bf16" \
  --resolution="1024*1024" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --optimizer="Adafactor" \
  --learning_rate=4e-6 \
  --adam_weight_decay=0.01 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=3000 \
  --validation_prompt="A photo of a sks corgi puppy sitting in a blue basket, high resolution" \
  --num_validation_inference_steps=10 \
  --validation_epochs=1 \
  --seed="0" \
  --gradient_checkpointing