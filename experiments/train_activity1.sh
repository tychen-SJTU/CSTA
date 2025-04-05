CUDA_VISIBLE_DEVICES=3 python3 tools/run_net.py \
--cfg "./configs/ActivityNet/TimeSformer_divST_8x32_224.yaml" \
--start_task 1 \
--init_task  20 \
--nb_class 20 \
--end_task 10 \
--casual_loss_weight 0.4 \
--casual_logits_weight 0.1 \
--mix_loss_weight 0.1 \
--loss_weight_decay 0.005 \
--val_epoch 1 \
--path_weight 0.08 \
--output_dir "./output/activitynet" \


