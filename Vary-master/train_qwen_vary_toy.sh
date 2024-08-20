#deepspeed --include localhost:4,5,6,7 vary/train/train_qwen_vary.py --deepspeed zero_config/zero2.json \
#          --model_name_or_path /mnt/ceph/pretrained/Ucas/vary-toy/ \
#          --vision_tower /mnt/ceph/pretrained/Ucas/vit-large-patch14/ \
#          --freeze_vision_tower False \
#          --freeze_lm_model False \
#          --vision_select_layer -2 \
#          --use_im_start_end True \
#          --bf16 True \
#          --per_device_eval_batch_size 1 \
#          --gradient_accumulation_steps 2 \
#          --evaluation_strategy "no" \
#          --save_strategy "steps" \
#          --save_steps 5000 \
#          --save_total_limit 2 \
#          --weight_decay 0. \
#          --warmup_ratio 0.03 \
#          --lr_scheduler_type "cosine" \
#          --logging_steps 1 --tf32 True \
#          --model_max_length 4096 \
#          --gradient_checkpointing True \
#          --dataloader_num_workers 4 \
#          --report_to tensorboard \
#          --per_device_train_batch_size 8\
#          --num_train_epochs 5 \
#          --learning_rate 1e-5 \
#          --datasets det_train \
#          --output_dir ../runs/0402


#deepspeed vary/train/train_qwen_vary.py --deepspeed zero_config/zero2.json \
#          --model_name_or_path /mnt/ceph2/pretrained/Ucas/vary-toy/ \
#          --vision_tower /mnt/ceph2/pretrained/Ucas/vit-large-patch14/ \
#          --freeze_vision_tower True \
#          --freeze_lm_model False \
#          --vision_select_layer -2 \
#          --use_im_start_end True \
#          --bf16 True \
#          --per_device_eval_batch_size 1 \
#          --gradient_accumulation_steps 4 \
#          --evaluation_strategy "no" \
#          --save_strategy "steps" \
#          --save_steps 5000 \
#          --save_total_limit 2 \
#          --weight_decay 0. \
#          --warmup_ratio 0.03 \
#          --lr_scheduler_type "cosine" \
#          --logging_steps 1 --tf32 True \
#          --model_max_length 4096 \
#          --gradient_checkpointing True \
#          --dataloader_num_workers 16 \
#          --report_to tensorboard \
#          --per_device_train_batch_size 4\
#          --num_train_epochs 10 \
#          --learning_rate 1e-6 \
#          --datasets sjb_point_train+tiku_point_train+data_train \
#          --output_dir ../runs/04163


deepspeed vary/train/train_qwen_vary.py --deepspeed zero_config/zero2.json \
          --model_name_or_path /mnt/ceph2/pretrained/Ucas/vary-toy/ \
          --vision_tower /mnt/ceph2/pretrained/Ucas/vit-large-patch14/ \
          --freeze_vision_tower True \
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
          --weight_decay 0. \
          --warmup_ratio 0.03 \
          --lr_scheduler_type "cosine" \
          --logging_steps 1 --tf32 True \
          --model_max_length 4096 \
          --gradient_checkpointing True \
          --dataloader_num_workers 16 \
          --report_to tensorboard \
          --per_device_train_batch_size 4\
          --num_train_epochs 5 \
          --learning_rate 1e-5 \
          --datasets tiku_font_train \
          --output_dir ../runs/0424