#deepspeed vary/train/train_qwen.py --deepspeed zero_config/zero2.json \
#          --model_name_or_path /mnt/ceph2/pretrained/Qwen/Qwen1.5-0.5B-Chat/ \
#          --vision_tower /mnt/ceph2/pretrained/Ucas/vit-large-patch14/ \
#          --freeze_vision_tower False \
#          --freeze_lm_model False \
#          --vision_select_layer -2 \
#          --use_im_start_end True \
#          --bf16 True \
#          --per_device_eval_batch_size 1 \
#          --gradient_accumulation_steps 4 \
#          --evaluation_strategy "no" \
#          --save_strategy "steps" \
#          --save_steps 1000 \
#          --save_total_limit 2 \
#          --weight_decay 0. \
#          --warmup_ratio 0.03 \
#          --lr_scheduler_type "cosine" \
#          --logging_steps 1 --tf32 True \
#          --model_max_length 4096 \
#          --gradient_checkpointing True \
#          --dataloader_num_workers 16 \
#          --report_to tensorboard \
#          --per_device_train_batch_size 4 \
#          --num_train_epochs 5 \
#          --learning_rate 5e-5 \
#          --datasets det_train+qms_train+tiku_ocr_1M_train+conversations_caption+conversations_rec+conversations_reg \
#          --output_dir ../runs/04143

deepspeed vary/train/train_qwen.py --deepspeed zero_config/zero2.json \
          --model_name_or_path /mnt/ceph2/pretrained/Qwen/Qwen1.5-0.5B-Chat/ \
          --vision_tower /mnt/ceph2/pretrained/Ucas/vit-large-patch14/ \
          --freeze_vision_tower False \
          --freeze_lm_model False \
          --vision_select_layer -2 \
          --use_im_start_end True \
          --bf16 True \
          --per_device_eval_batch_size 1 \
          --gradient_accumulation_steps 4 \
          --evaluation_strategy "no" \
          --save_strategy "steps" \
          --save_steps 1000 \
          --save_total_limit 2 \
          --weight_decay 0.1 \
          --adam_beta2 0.95 \
          --warmup_ratio 0. \
          --lr_scheduler_type "cosine" \
          --logging_steps 1 --tf32 True \
          --model_max_length 4096 \
          --gradient_checkpointing True \
          --dataloader_num_workers 16 \
          --report_to tensorboard \
          --per_device_train_batch_size 4 \
          --num_train_epochs 1 \
          --learning_rate 1e-5 \
          --datasets det_train+qms_train+tiku_ocr_1M_train+conversations_caption+conversations_rec+conversations_reg \
          --output_dir ../runs/0416