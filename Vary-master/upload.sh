#rsync -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku7 root@10.204.53.71:/mnt/data1/wcq/datasets/tiku7
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/vary/conversations_tiku_id2.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/vary/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/vary/conversations_sjb_id2.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/vary/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/vary/conversations_camera_id2.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/vary/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/vary/conversations_yyt_ocr2.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/vary/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/vary/conversations_tiku_id3.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/vary/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/pretrain/conversations_box_det.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/pretrain/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/datasets/tiku/pretrain/conversations_box_qms.json root@10.204.53.71:/mnt/data1/wcq/datasets/tiku/pretrain/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/train_qwen_vary_A800.sh root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/train_qwen_vary.sh
rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/train_qwen2.sh root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/train_qwen2_vary.sh root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/.
rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/vary/train/train_qwen2.py root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/vary/train/.
rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/vary/model/vary_qwen2.py root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/vary/model/.
#rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/vary/utils/constants_A800.py root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/vary/utils/constants.py
rsync --progress -avz -e "ssh -i /mnt/ceph2/key/stage_rsa_2048" /mnt/ceph2/Vary/Vary-master/check.py root@10.204.53.71:/mnt/data1/wcq/Vary/Vary-master/.