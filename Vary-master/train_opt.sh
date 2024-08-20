# zero2 batch 4 100/544400 [01:08<98:19:32,  1.54it/s] 34874MiB
#             4x2 100/272200 [02:07<90:41:25,  1.20s/it] 31202MiB
#             4x4 100/136000 [04:09<90:18:24,  2.39s/it] 33158MiB
# zero3 batch 4 100/544400 [01:47<139:15:17,  1.09it/s] 30412MiB
#             8 100/272200 [02:58<127:44:38,  1.69s/it] 39544MiB
# zero3_offload batch 4 100/544400 [02:46<232:11:07,  1.54s/it] 31854MiB
#                     8 100/272200 [04:01<168:58:36,  2.24s/it] 39896MiB
#                     8x2 100/136000 [06:41<149:12:28,  3.95s/it] 38864MiB
deepspeed vary/train/train_opt.py --deepspeed zero_config/zero3.json \
          --model_name_or_path /mnt/ceph2/pretrained/facebook/opt-125m/ \
          --conversation_version opt \
          --freeze_vision_tower False \
          --freeze_lm_model False \
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
          --logging_steps 10 --tf32 True \
          --model_max_length 4096 \
          --gradient_checkpointing True \
          --dataloader_num_workers 16 \
          --report_to tensorboard \
          --per_device_train_batch_size 4 \
          --num_train_epochs 3 \
          --learning_rate 1e-4 \
          --datasets opt_html_1M_train \
          --output_dir ../runs/0529