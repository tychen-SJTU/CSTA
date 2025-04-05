CUDA_VISIBLE_DEVICES=0,1,2,3 python3 tools/run_net.py \
--cfg "./configs/Kinetics/TimeSformer_divST_8x32_224.yaml" \
--start_task 1 \
--init_task  40 \
--nb_class 40 \
--end_task 10 \
--casual_loss_weight 0.1 \
--casual_logits_weight 0.1 \
--mix_loss_weight 0.1 \
--loss_weight_decay 0.0025 \
--adapter 1 \
--casual 1 \
--supervise_loss 1 \
--compensate_effect 1 \
--casual_enhance 1 \
--val_epoch 3 \
--path_weight 0.1 \
--output_dir "./output/kinetics" \


