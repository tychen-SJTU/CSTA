CUDA_VISIBLE_DEVICES=2,3 python3 tools/run_net.py \
--cfg "./configs/UCF101/TimeSformer_divST_8x32_224.yaml" \
--start_task 1 \
--init_task  51 \
--nb_class 10 \
--end_task 6 \
--casual_loss_weight 0.2 \
--casual_logits_weight 0.15 \
--mix_loss_weight 0.1 \
--loss_weight_decay 0.0025 \
--val_epoch 1 \
--path_weight 0.1 \
--output_dir "./output/ucf" \


