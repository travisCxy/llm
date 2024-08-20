deepspeed vary/train/train_qwen_box.py --deepspeed zero_config/zero2.json \
          --model_name_or_path /mnt/ceph/pretrained/hfl/chinese-llama-2-7b/ \
          --vision_tower /mnt/ceph/pretrained/Ucas/vit-large-patch14/ \
          --freeze_vision_tower True \
          --freeze_lm_model False \
          --vision_select_layer -2 \
          --use_im_start_end True \
          --bf16 True \
          --per_device_eval_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 5000 \
          --save_total_limit 2 \
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 1 --tf32 True \
          --model_max_length 4096 \
          --gradient_checkpointing True \
          --dataloader_num_workers 4 \
          --report_to tensorboard \
          --per_device_train_batch_size 4 \
          --num_train_epochs 3 \
          --learning_rate 2e-5\
          --datasets tiku_next_box0_train+tiku_next_box1_train \
          --output_dir ../runs/03242
