CUDA_VISIBLE_DEVICES=2,3 python3 tools/run_net.py \
--cfg "./configs/HMDB51/TimeSformer_divST_8x32_224.yaml" \
--start_task 1 \
--init_task  26 \
--nb_class 5 \
--end_task 4 \
--casual_loss_weight 0.2 \
--casual_logits_weight 0.15 \
--mix_loss_weight 0.1 \
--loss_weight_decay 0.1 \
--val_epoch 1 \
--path_weight 0.1 \
--output_dir "./output/hmdb" \


