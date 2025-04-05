CUDA_VISIBLE_DEVICES=4,5 python3 tools/run_net.py \
--cfg "./configs/SSv2/TimeSformer_divST_8_224.yaml" \
--init_method "tcp://localhost:9990" \
--start_task 1 \
--init_task  84 \
--nb_class 10 \
--end_task 10 \
--casual_loss_weight 0.1 \
--casual_logits_weight 0.1 \
--mix_loss_weight 0.1 \
--loss_weight_decay 0.005 \
--adapter 1 \
--casual  1 \
--supervise_loss 1 \
--compensate_effect 1 \
--casual_enhance 1 \
--balance_training 1 \
--val_epoch 1 \
--path_weight 0.08 \
--output_dir "./output/ssv2" \


