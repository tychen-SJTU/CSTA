# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Train a video classification model."""
import copy
import numpy as np
import itertools
import json
from collections import Counter
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
import torch
import torch.nn.functional as F
from fvcore.nn.precise_bn import get_bn_modules, update_bn_stats
import os
from cl_utils import set_lambda_0, lf_dist_tcd, feat_dist, init_cosine_classifier
import timesformer.models.losses as losses
import timesformer.models.optimizer as optim
import timesformer.utils.checkpoint as cu
import timesformer.utils.distributed as du
import timesformer.utils.logging as logging
import timesformer.utils.metrics as metrics
import timesformer.utils.misc as misc
# import timesformer.visualization.tensorboard_vis as tb
from timesformer.datasets import loader
from timesformer.models import build_model
from timesformer.utils.meters import TrainMeter, ValMeter
from timesformer.utils.multigrid import MultigridSchedule
from timesformer.models.vit import Classifier
from torch.nn.functional import kl_div
from timm.data import Mixup
from torch import nn
import torch.distributed as dist
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

logger = logging.get_logger(__name__)


def train_epoch(
        train_loader, model, optimizer, train_meter, cur_epoch, cfg, replay_loader, score_resultss=None,
        index_resultss=None,
        score_resultst=None, index_resultst=None, score_delta_s=None, score_delta_t=None, index_delta_s=None,
        index_delta_t=None, inputs_list=None,
        args=None, writer=None
):

    global preds, pred1, pred2, pred3, cur_iter, data_cos1, data_cos2, data_cos3, data_cos4, data_spatial_grad_average, \
        data_temporal_grad_average, data_learning_temporal_grad_average, data_learning_spatial_grad_average, grad1, grad2, \
        grad3, grad4, spatial_grad_sum, temporal_grad_sum, learning_temporal_grad_sum, learning_spatial_grad_sum, cos_sum, \
        cos_sum2, cos_sum3, cos_sum4, hard_labels, loss_s, loss_t, kl_spatial, kl_temporal, spatial_mix, temporal_mix, \
        loss_t_mix, loss_s_mix, loss_s_mix_ex, loss_t_mix_ex, logits_spatial0, logits_temporal0, pred_s, pred_t, \
        preds_glipmse, loss_ce_glimpse
    model.train()
    train_meter.iter_tic()
    data_size = len(train_loader)
    if cfg.HEAD_TYPE == 'cosine':
        if cfg.NUM_GPUS > 1:
            model.module.model.training = True
        else:
            model.training = True
    cfg.TRAINING = True

    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
    batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
    cos_list_tisi, cos_list_tmsm, cos_list_tmsi, cos_list_tism = [], [], [], []

    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):

        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        # Update the learning rate.
        lr = optim.get_epoch_lr(cur_epoch + float(cur_iter) / data_size, cfg)
        optim.set_lr(optimizer, lr)

        train_meter.data_toc()

        # Explicitly declare reduction to mean.
        if not cfg.MIXUP.ENABLED:
            loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
        else:
            mixup_fn = Mixup(
                mixup_alpha=cfg.MIXUP.ALPHA, cutmix_alpha=cfg.MIXUP.CUTMIX_ALPHA,
                cutmix_minmax=cfg.MIXUP.CUTMIX_MINMAX, prob=cfg.MIXUP.PROB,
                switch_prob=cfg.MIXUP.SWITCH_PROB, mode=cfg.MIXUP.MODE,
                label_smoothing=0.1, num_classes=cfg.MODEL.NUM_CLASSES)
            hard_labels = labels
            inputs, labels = mixup_fn(inputs, labels)
            loss_fun = SoftTargetCrossEntropy()
        # test baseline
        if cfg.AGE > 0:
            if not args.adapter:
                if not args.glimpse:
                    model.eval()
                    model.module.model.freeze_parameters()
                    for blk in model.module.model.blocks:
                        blk.mlp.train()
                        blk.mlp.activite_parameters()
                    model.module.model.new_head.train()
                    model.module.model.new_head.activite_parameters()
            # model.module.model.head.train()
            # model.module.model.head.activite_parameters()
        if args.adapter:
            if cfg.AGE > 0:
                model.eval()
                if cfg.NUM_GPUS == 1:
                    # single gpu
                    model.model.freeze_parameters()
                    for blk in model.model.blocks:
                        # freeze former adapter to save key message
                        for i in range(0, cfg.AGE - 1):
                            blk.Spatial_adapter[i].eval()
                            blk.Temporal_adapter[i].eval()
                            blk.Spatial_adapter[i].freeze_parameters()
                            blk.Temporal_adapter[i].freeze_parameters()
                        # train new adapter and att via tasks
                        blk.Spatial_adapter[cfg.AGE - 1].train()
                        blk.Temporal_adapter[cfg.AGE - 1].train()
                        blk.Spatial_attn_via_tasks.train()
                        blk.Temporal_attn_via_tasks.train()
                        blk.Spatial_adapter[cfg.AGE - 1].activite_parameters()
                        blk.Temporal_adapter[cfg.AGE - 1].activite_parameters()
                        blk.Spatial_attn_via_tasks.activite_parameters()
                        blk.Temporal_attn_via_tasks.activite_parameters()
                    # freeze former head, tran new head
                    model.model.new_head.train()
                    model.model.new_head.activite_parameters()
                else:
                    # multi gpus
                    model.module.model.freeze_parameters()
                    for blk in model.module.model.blocks:
                        for i in range(0, cfg.AGE - 1):
                            blk.Spatial_adapter[i].eval()
                            blk.Temporal_adapter[i].eval()
                            blk.Spatial_adapter[i].freeze_parameters()
                            blk.Temporal_adapter[i].freeze_parameters()
                        blk.Spatial_adapter[cfg.AGE - 1].train()
                        blk.Temporal_adapter[cfg.AGE - 1].train()
                        blk.Spatial_attn_via_tasks.train()
                        blk.Temporal_attn_via_tasks.train()
                        blk.Spatial_adapter[cfg.AGE - 1].activite_parameters()
                        blk.Temporal_adapter[cfg.AGE - 1].activite_parameters()
                        blk.Spatial_attn_via_tasks.activite_parameters()
                        blk.Temporal_attn_via_tasks.activite_parameters()
                    model.module.model.new_head.train()
                    model.module.model.new_head.activite_parameters()

        if cfg.DETECTION.ENABLE:
            _, feat, _, _, _ = model(inputs, meta["boxes"])
        else:
            if cfg.AGE > 0:
                if args.adapter:
                    # with adapter
                    preds, feat, pred3, logits_spatial0, logits_temporal0 = model(inputs, 0)
                    pred_s, pred_t = model(inputs, 2)
                else:
                    # baseline
                    if args.glimpse:
                        inputs_glipmse = inputs[:, 3, :, :, :].unsqueeze(1).repeat(1, 8, 1, 1, 1)
                        preds_glipmse, _ = model(inputs_glipmse, 1)
                        preds, _ = model(inputs, 1)
                (output_list, output_list1, output_list2, output_list_st, output_list_st2, spatial_mix_list,
                 temporal_mix_list) = [], [], [], [], [], [], []

                if args.casual:
                    for bs in range(0, batch_size):
                        output, output1, output2, output_st, output_st2, spatial_mix, temporal_mix = 0, 0, 0, 0, 0, 0, 0

                        if args.casual_enhance:
                            for i in range(0, 5):
                                inputs_s = inputs_list[index_resultss[cur_iter * batch_size + bs][i]]
                                inputs_s = inputs_s.cuda()
                                logits_spatial, logits_temporal = model(inputs_s, 2)
                                logits_spatial = logits_spatial.detach()
                                logits_temporal = logits_temporal.detach()
                                torch.cuda.empty_cache()
                                if args.compensate_effect:
                                    output += score_resultss[cur_iter * batch_size + bs][i] * (logits_spatial +
                                                                                               0.3 * logits_temporal)
                                    # mix loss spatial
                                    spatial_mix += score_resultss[cur_iter * batch_size + bs][i] * logits_spatial
                                else:
                                    output += score_resultss[cur_iter * batch_size + bs][i] * logits_spatial

                                del logits_spatial, inputs_s, logits_temporal
                                torch.cuda.empty_cache()

                                inputs_t = inputs_list[index_resultst[cur_iter * batch_size + bs][i]]
                                inputs_t = inputs_t.cuda()
                                _, logits_temporal = model(inputs_t, 2)
                                logits_temporal = logits_temporal.detach()
                                torch.cuda.empty_cache()

                                output += score_resultst[cur_iter * batch_size + bs][i] * logits_temporal
                                # mix loss temporal
                                if args.compensate_effect:
                                    temporal_mix += score_resultst[cur_iter * batch_size + bs][i] * logits_temporal

                                del logits_temporal, inputs_t
                                torch.cuda.empty_cache()

                        if args.supervise_loss:
                            for i in range(0, 3):
                                inputs_s = inputs_list[index_delta_s[cur_iter * batch_size + bs][i]]
                                inputs_s = inputs_s.cuda()
                                _, _, logits_spatial, logits_temporal = model(inputs_s, 4)
                                logits_spatial = logits_spatial.detach()
                                logits_temporal = logits_temporal.detach()
                                torch.cuda.empty_cache()
                                output1 += score_delta_s[cur_iter * batch_size + bs][i] * logits_spatial
                                output_st += score_delta_s[cur_iter * batch_size + bs][i] * \
                                             (logits_spatial + logits_temporal)
                                del logits_spatial, logits_temporal, inputs_s
                                torch.cuda.empty_cache()

                                inputs_t = inputs_list[index_delta_t[cur_iter * batch_size + bs][i]]
                                inputs_t = inputs_t.cuda()
                                _, _, logits_spatial, logits_temporal = model(inputs_t, 4)
                                logits_spatial = logits_spatial.detach()
                                logits_temporal = logits_temporal.detach()
                                torch.cuda.empty_cache()
                                output2 += score_delta_t[cur_iter * batch_size + bs][i] * logits_temporal
                                output_st2 += score_delta_t[cur_iter * batch_size + bs][i] * \
                                              (logits_spatial + logits_temporal)
                                del logits_spatial, logits_temporal, inputs_t
                                torch.cuda.empty_cache()

                        if args.casual_enhance:
                            # casual enhance
                            output_list.append(output)
                        if args.supervise_loss:
                            # supervise
                            output_list1.append(output1[:, :cfg.NUM_OLD_HEAD])
                            output_list2.append(output2[:, :cfg.NUM_OLD_HEAD])
                            output_list_st.append(output_st[:, :cfg.NUM_OLD_HEAD])
                            output_list_st2.append(output_st2[:, :cfg.NUM_OLD_HEAD])
                            # compensate
                        if args.compensate_effect:
                            spatial_mix_list.append(spatial_mix)
                            temporal_mix_list.append(temporal_mix)

                    if args.casual_enhance:
                        output = torch.stack(output_list)
                        preds = preds + cfg.LOGITS_WEIGHT * torch.squeeze(output, dim=1)

                    if args.supervise_loss:
                        output1 = torch.stack(output_list1)
                        output2 = torch.stack(output_list2)
                        output_st = torch.stack(output_list_st)
                        output_st2 = torch.stack(output_list_st2)

                        s_st_lable = (output1 / output_st).squeeze(1)
                        t_st_lable = (output2 / output_st2).squeeze(1)
                        current_s_st = logits_spatial0[:, :cfg.NUM_OLD_HEAD] / \
                                       (logits_spatial0[:, :cfg.NUM_OLD_HEAD] + logits_temporal0[:, :cfg.NUM_OLD_HEAD])
                        current_t_st = logits_temporal0[:, :cfg.NUM_OLD_HEAD] / \
                                       (logits_spatial0[:, :cfg.NUM_OLD_HEAD] + logits_temporal0[:, :cfg.NUM_OLD_HEAD])

                        loss_s = 1 - torch.abs(F.cosine_similarity(current_s_st, s_st_lable, dim=1).mean())
                        loss_t = 1 - torch.abs(F.cosine_similarity(current_t_st, t_st_lable, dim=1).mean())

                    if args.compensate_effect:
                        # mix loss
                        output_spatial = torch.stack(spatial_mix_list).squeeze(1)
                        output_temporal = torch.stack(temporal_mix_list).squeeze(1)
                        loss_s_mix_ex = 1 - torch.abs(F.cosine_similarity(logits_spatial0[:, :cfg.NUM_OLD_HEAD],
                                                                          output_spatial[:, :cfg.NUM_OLD_HEAD],
                                                                          dim=1).mean())
                        loss_t_mix_ex = 1 - torch.abs(F.cosine_similarity(logits_temporal0[:, :cfg.NUM_OLD_HEAD],
                                                                          output_temporal[:, :cfg.NUM_OLD_HEAD],
                                                                          dim=1).mean())
            else:
                # while in age 0
                preds, _ = model(inputs, 1)
                if args.glimpse:
                    inputs_glipmse = inputs[:, 3, :, :, :].unsqueeze(1)
                    inputs_glipmse = inputs_glipmse.repeat(1, 8, 1, 1, 1)
                    preds_glipmse, _ = model(inputs_glipmse, 1)
        loss_ce = 0
        if cfg.AGE > 0:
            for i in range(0, labels.shape[0]):
                if labels[i] not in cfg.EX_TASK:
                    loss_ce += loss_fun(preds[i], labels[i])
        else:
            loss_ce = loss_fun(preds, labels)

        if args.glimpse:
            loss_ce_glimpse = loss_fun(preds_glipmse, labels)
        if cfg.AGE > 0:
            if args.adapter:
                if cfg.NUM_GPUS == 1:
                    torch.save(model.model.blocks[0].Spatial_adapter[cfg.AGE - 1].state_dict(),
                               os.path.join(args.output_dir, 'model_Spatial_adapter{}.pt'.format(cfg.AGE - 1)))
                    torch.save(model.model.blocks[0].Temporal_adapter[cfg.AGE - 1].state_dict(),
                               os.path.join(args.output_dir, 'model_Temporal_adapter{}.pt'.format(cfg.AGE - 1)))
                    torch.save(model.model.blocks[0].Spatial_attn_via_tasks.state_dict(),
                               os.path.join(args.output_dir, 'model_Spatial_attn_via_tasks{}.pt'.format(cfg.AGE - 1)))
                    torch.save(model.model.blocks[0].Temporal_attn_via_tasks.state_dict(),
                               os.path.join(args.output_dir, 'model_Temporal_attn_via_tasks{}.pt'.format(cfg.AGE - 1)))
                    torch.save(model.model.new_head.state_dict(),
                               os.path.join(args.output_dir, 'model_head{}.pt'.format(cfg.AGE)))
                else:
                    if dist.get_rank() == 0:
                        torch.save(model.module.model.blocks[0].Spatial_adapter[cfg.AGE - 1].state_dict(),
                                   os.path.join(args.output_dir, 'model_Spatial_adapter{}.pt'.format(cfg.AGE - 1)))
                        torch.save(model.module.model.blocks[0].Temporal_adapter[cfg.AGE - 1].state_dict(),
                                   os.path.join(args.output_dir, 'model_Temporal_adapter{}.pt'.format(cfg.AGE - 1)))
                        torch.save(model.module.model.blocks[0].Spatial_attn_via_tasks.state_dict(),
                                   os.path.join(args.output_dir,
                                                'model_Spatial_attn_via_tasks{}.pt'.format(cfg.AGE - 1)))
                        torch.save(model.module.model.blocks[0].Temporal_attn_via_tasks.state_dict(),
                                   os.path.join(args.output_dir,
                                                'model_Temporal_attn_via_tasks{}.pt'.format(cfg.AGE - 1)))
                        torch.save(model.module.model.new_head.state_dict(),
                                   os.path.join(args.output_dir, 'model_head{}.pt'.format(cfg.AGE)))
                loss_ce5 = 0
                # for i in range(0, labels.shape[0]):
                if labels[0] not in cfg.EX_TASK:
                    loss_ce5 = F.kl_div(preds.log_softmax(dim=1), pred3.softmax(dim=1), reduction='batchmean')

                if args.casual and args.supervise_loss and args.casual_enhance and args.compensate_effect:
                    #  all ideas
                    loss = (loss_ce + 20 * loss_ce5 +
                            (args.casual_loss_weight - args.loss_weight_decay * (cfg.AGE - 1)) *
                            (loss_s + loss_t) +
                            (args.mix_loss_weight - args.loss_weight_decay * (cfg.AGE - 1)) *
                            (loss_s_mix_ex + loss_t_mix_ex))

                elif args.casual and args.supervise_loss and not args.compensate_effect:
                    # only casual_enhance
                    loss = loss_ce + 20 * loss_ce5 + cfg.LOSS_WEIGHT * (loss_s + loss_t)
                else:
                    loss = loss_ce + 20 * loss_ce5

            # baseline
            else:
                if args.glimpse:
                    loss = loss_ce + loss_ce_glimpse
                else:
                    loss = loss_ce
                if not args.glimpse:
                    if cfg.NUM_GPUS == 1:
                        torch.save(model.model.new_head.state_dict(),
                                   os.path.join(args.output_dir, 'model_head{}.pt'.format(cfg.AGE)))
                    else:
                        if dist.get_rank() == 0:
                            torch.save(model.module.model.new_head.state_dict(),
                                       os.path.join(args.output_dir, 'model_head{}.pt'.format(cfg.AGE)))
        else:
            if args.glimpse:
                loss = loss_ce + loss_ce_glimpse
            else:
                loss = loss_ce

        if cfg.MIXUP.ENABLED:
            labels = hard_labels

        # check Nan Loss.
        misc.check_nan_losses(loss)

        if cur_global_batch_size >= cfg.GLOBAL_BATCH_SIZE:
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward(retain_graph=False)
            # Update the parameters.
            optimizer.step()
        else:
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward(retain_graph=False)

            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    if p.grad != None:
                        p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()

        if cfg.DETECTION.ENABLE:
            if cfg.NUM_GPUS > 1:
                loss = du.all_reduce([loss])[0]
            loss = loss.item()

            # Update and log stats.
            train_meter.update_stats(None, None, None, loss, lr)
            # write to tensorboard format if available.
            if writer is not None:
                if model.local_rank == 0:
                    writer.add_scalars(
                        {"Train/loss": loss, "Train/lr": lr},
                        global_step=data_size * cur_epoch + cur_iter,
                    )

        else:
            top1_err, top5_err = None, None
            if cfg.DATA.MULTI_LABEL:
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    [loss] = du.all_reduce([loss])
                loss = loss.item()
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                # Gather all the predictions across all the devices.
                if cfg.NUM_GPUS > 1:
                    loss, top1_err, top5_err = du.all_reduce(
                        [loss, top1_err, top5_err]
                    )

                # Copy the stats from GPU to CPU (sync point).
                loss, top1_err, top5_err = (
                    loss.item(),
                    top1_err.item(),
                    top5_err.item(),
                )

            # Update and log stats.
            train_meter.update_stats(
                top1_err,
                top5_err,
                loss,
                lr,
                inputs[0].size(0)
                * max(
                    cfg.NUM_GPUS, 1
                ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
            )
            # write to tensorboard format if available.
            if writer is not None:
                if model.local_rank == 0:
                    writer.add_scalars(
                        {
                            "Train/loss": loss,
                            "Train/lr": lr,
                            "Train/Top1_err": top1_err,
                            "Train/Top5_err": top5_err,
                        },
                        global_step=data_size * cur_epoch + cur_iter,
                    )

        train_meter.iter_toc()  # measure allreduce for this meter
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats.
    file_path = 'cos_list_tisi5.txt'
    with open(file_path, 'a') as file:
        for item in cos_list_tisi:
            file.write(str(item) + '\n')

    file_path = 'cos_list_tmsm5.txt'
    with open(file_path, 'a') as file:
        for item in cos_list_tmsm:
            file.write(str(item) + '\n')

    file_path = 'cos_list_tmsi5.txt'
    with open(file_path, 'a') as file:
        for item in cos_list_tmsi:
            file.write(str(item) + '\n')

    file_path = 'cos_list_tism5.txt'
    with open(file_path, 'a') as file:
        for item in cos_list_tism:
            file.write(str(item) + '\n')
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


def balance_fintuning(balance_loader, model, cfg, optimizer, args):
    #  fine-tuning head
    print("balance training")
    model.eval()
    model.module.model.freeze_parameters()
    model.module.model.new_head.train()
    model.module.model.new_head.activite_parameters()
    model.module.model.head.train()
    model.module.model.head.activite_parameters()

    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    cur_global_batch_size = cfg.NUM_SHARDS * cfg.TRAIN.BATCH_SIZE
    num_iters = cfg.GLOBAL_BATCH_SIZE // cur_global_batch_size
    for i in range(0, 3):
        for cur_iter, (inputs, labels, _, meta) in enumerate(balance_loader):
            if cfg.NUM_GPUS:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda()
                for key, val in meta.items():
                    if isinstance(val, (list,)):
                        for i in range(len(val)):
                            val[i] = val[i].cuda(non_blocking=True)
                    else:
                        meta[key] = val.cuda(non_blocking=True)

            # Update the learning rate.
            data_size = len(balance_loader)
            lr = optim.get_epoch_lr(0 + float(cur_iter) / data_size, cfg)

            optim.set_lr(optimizer, lr)
            preds, _, _, _, _ = model(inputs, 0)
            loss = loss_fun(preds, labels)
            if cur_iter == 0:
                optimizer.zero_grad()
            loss.backward(retain_graph=False)
            if (cur_iter + 1) % num_iters == 0:
                for p in model.parameters():
                    if p.grad != None:
                        p.grad /= num_iters
                optimizer.step()
                optimizer.zero_grad()
    save_head = Classifier(768, cfg.MODEL.NUM_CLASSES)
    save_head.head.weight.data[:cfg.NUM_OLD_HEAD] = model.module.model.head.head.weight.data.clone()
    save_head.head.bias.data[:cfg.NUM_OLD_HEAD] = model.module.model.head.head.bias.data.clone()
    save_head.head.weight.data[
    cfg.NUM_OLD_HEAD:cfg.MODEL.NUM_CLASSES] = model.module.model.new_head.head.weight.data.clone()
    save_head.head.bias.data[
    cfg.NUM_OLD_HEAD:cfg.MODEL.NUM_CLASSES] = model.module.model.new_head.head.bias.data.clone()
    torch.save(save_head.state_dict(), os.path.join(args.output_dir, 'complete_head{}.pt'.format(cfg.AGE)))


def grad_calculate(train_loader, model, cfg):
    """
    grad calculate sim calculate
    """
    model.eval()
    grad_tensor_list_S = []
    grad_tensor_list_T = []
    grad_tensor_list_S_ST = []
    grad_tensor_list_T_ST = []
    print("calculate grad ")
    for cur_iter, (inputs, labels, _, meta) in tqdm(enumerate(train_loader)):
        inputs = inputs.cuda()
        preds, _, logits_spatial, logits_temporal = model(inputs, 5)

        pred_tensor_S = logits_spatial.detach()
        pred_tensor_S.requires_grad = False
        pred_tensor_T = logits_temporal.detach()
        pred_tensor_T.requires_grad = False
        pred_tensor = preds.detach()
        pred_tensor.requires_grad = False
        for_norm0 = torch.norm(pred_tensor_S)
        for_norm1 = torch.norm(pred_tensor_T)
        dot_product = torch.sum(pred_tensor_S * pred_tensor_T, dim=1, keepdim=True)
        cosine_sim = dot_product / (for_norm0 * for_norm1)

        torch.cuda.empty_cache()
        grad_tensor_list_S.append(pred_tensor_S)
        grad_tensor_list_T.append(pred_tensor_T)

        grad_tensor_list_S_ST.append(torch.cat((pred_tensor_S / (pred_tensor_S + pred_tensor_T), cosine_sim), dim=1))
        grad_tensor_list_T_ST.append(torch.cat((pred_tensor_T / (pred_tensor_S + pred_tensor_T), cosine_sim), dim=1))
        torch.cuda.empty_cache()

    print("calculate sim ")
    # calculate relation <S,ST>
    length = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)) * len(train_loader)
    tensors = torch.stack(grad_tensor_list_S_ST).view(length, -1)
    distances = F.cosine_similarity(tensors.unsqueeze(1), tensors.unsqueeze(0), dim=2)
    top_k = 3
    score_results_S_ST = []
    index_results_S_ST = []
    for i in range(len(grad_tensor_list_S_ST * int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)))):
        distance_scores = distances[i]
        topk_scores, topk_indices = torch.topk(distance_scores, k=top_k, largest=True, sorted=True)
        topk_scores = torch.softmax(topk_scores, dim=0)
        combined = zip(topk_indices, topk_scores)
        sorted_combined = sorted(combined, key=lambda x: x[0])
        topk_indices, topk_scores = zip(*sorted_combined)
        topk_scores = [x.item() for x in topk_scores]
        topk_indices = [x.item() for x in topk_indices]
        score_results_S_ST.append(topk_scores)
        index_results_S_ST.append(topk_indices)

    # calculate relation <T,ST>
    tensors = torch.stack(grad_tensor_list_T_ST).view(length, -1)
    distances = F.cosine_similarity(tensors.unsqueeze(1), tensors.unsqueeze(0), dim=2)
    score_results_T_ST = []
    index_results_T_ST = []
    for i in range(len(grad_tensor_list_T_ST * int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)))):
        distance_scores = distances[i]
        topk_scores, topk_indices = torch.topk(distance_scores, k=top_k, largest=True, sorted=True)
        topk_scores = torch.softmax(topk_scores, dim=0)
        combined = zip(topk_indices, topk_scores)
        sorted_combined = sorted(combined, key=lambda x: x[0])
        topk_indices, topk_scores = zip(*sorted_combined)
        topk_scores = [x.item() for x in topk_scores]
        topk_indices = [x.item() for x in topk_indices]
        score_results_T_ST.append(topk_scores)
        index_results_T_ST.append(topk_indices)

    top_k = 5
    # Calculate S
    tensors = torch.stack(grad_tensor_list_S).view(length, -1)
    distances = F.cosine_similarity(tensors.unsqueeze(1), tensors.unsqueeze(0), dim=2)
    score_results_S = []
    index_results_S = []
    for i in range(len(grad_tensor_list_S * int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)))):
        distance_scores = distances[i]
        topk_scores, topk_indices = torch.topk(distance_scores, k=top_k, largest=True, sorted=True)
        topk_scores = torch.softmax(topk_scores, dim=0)
        combined = zip(topk_indices, topk_scores)
        sorted_combined = sorted(combined, key=lambda x: x[0])
        topk_indices, topk_scores = zip(*sorted_combined)
        topk_scores = [x.item() for x in topk_scores]
        topk_indices = [x.item() for x in topk_indices]
        score_results_S.append(topk_scores)
        index_results_S.append(topk_indices)

    # Calculate T
    tensors = torch.stack(grad_tensor_list_T).view(length, -1)
    distances = F.cosine_similarity(tensors.unsqueeze(1), tensors.unsqueeze(0), dim=2)
    score_results_T = []
    index_results_T = []
    for i in range(len(grad_tensor_list_T * int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS)))):
        distance_scores = distances[i]
        topk_scores, topk_indices = torch.topk(distance_scores, k=top_k, largest=True, sorted=True)
        topk_scores = torch.softmax(topk_scores, dim=0)
        combined = zip(topk_indices, topk_scores)
        sorted_combined = sorted(combined, key=lambda x: x[0])
        topk_indices, topk_scores = zip(*sorted_combined)
        topk_scores = [x.item() for x in topk_scores]
        topk_indices = [x.item() for x in topk_indices]
        score_results_T.append(topk_scores)
        index_results_T.append(topk_indices)

    return score_results_S_ST, index_results_S_ST, score_results_T_ST, index_results_T_ST, \
        score_results_S, index_results_S, score_results_T, index_results_T


def get_input(train_loader):
    inputs_list = []
    for cur_iter, (inputs, labels, _, meta) in enumerate(train_loader):
        inputs_list.append(inputs)
        if cur_iter % 2000 == 0:
            print(cur_iter / len(train_loader))
    # stacked_tensor = torch.stack(inputs_list, dim=0)
    # stacked_tensor = stacked_tensor.view(-1, 8, 3, 224, 224)
    # inputs_list = [tensor.view(1, 8, 3, 224, 224) for tensor in torch.split(stacked_tensor, 1, dim=0)]
    return inputs_list


def load_single_score(file_path):
    score_resultss = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip() + ']'
        data = [round(float(num), 5) for num in line.strip().lstrip('[').rstrip(']').split(',')]
        score_resultss.append(data)
    return score_resultss


def load_single_index(file_path):
    index_resultss = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip() + ']'
        data = [int(num) for num in line.strip().lstrip('[').rstrip(']').split(',')]
        index_resultss.append(data)
    return index_resultss


@torch.no_grad()
def eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer=None):
    """
    Evaluate the model on the val set.
    Args:
        val_loader (loader): data loader to provide validation data.
        model (model): model to evaluate the performance.
        val_meter (ValMeter): meter instance to record and calculate the metrics.
        cur_epoch (int): number of the current epoch of training.
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        writer (TensorboardWriter, optional): TensorboardWriter object
            to writer Tensorboard log.
    """

    # Evaluation mode enabled. The running stats would not be updated.
    global top1_err
    model.eval()
    val_meter.iter_tic()
    cfg.TRAINING = False
    if cfg.HEAD_TYPE == 'cosine':
        if cfg.NUM_GPUS > 1:
            model.module.model.training = False
        else:
            model.training = False
    total_counts = torch.zeros(51, dtype=torch.long)  # 用于累积类别计数
    for cur_iter, (inputs, labels, _, meta) in enumerate(val_loader):
        if cfg.NUM_GPUS:
            # Transferthe data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda()
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)
        val_meter.data_toc()
        # model.module.model.training = False
        if cfg.DETECTION.ENABLE:
            # Compute the predictions.
            preds, feat, _ = model(inputs, meta["boxes"])
            ori_boxes = meta["ori_boxes"]
            metadata = meta["metadata"]

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                ori_boxes = ori_boxes.cpu()
                metadata = metadata.cpu()

            if cfg.NUM_GPUS > 1:
                preds = torch.cat(du.all_gather_unaligned(preds), dim=0)
                ori_boxes = torch.cat(du.all_gather_unaligned(ori_boxes), dim=0)
                metadata = torch.cat(du.all_gather_unaligned(metadata), dim=0)

            val_meter.iter_toc()
            # Update and log stats.
            val_meter.update_stats(preds, ori_boxes, metadata)

        else:
            preds, feat = model(inputs, 1)
            if cfg.DATA.MULTI_LABEL:
                if cfg.NUM_GPUS > 1:
                    preds, labels = du.all_gather([preds, labels])
            else:
                # Compute the errors.
                num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))

                # Combine the errors across the GPUs.
                top1_err, top5_err = [
                    (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
                ]
                if cfg.NUM_GPUS > 1:
                    top1_err, top5_err = du.all_reduce([top1_err, top5_err])

                # Copy the errors from GPU to CPU (sync point).
                top1_err, top5_err = top1_err.item(), top5_err.item()

                val_meter.iter_toc()
                # Update and log stats.
                val_meter.update_stats(
                    top1_err,
                    top5_err,
                    inputs[0].size(0)
                    * max(
                        cfg.NUM_GPUS, 1
                    ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
                )
                # write to tensorboard format if available.
                if writer is not None:
                    if model.local_rank == 0:
                        writer.add_scalars(
                            {"Val/Top1_err": top1_err, "Val/Top5_err": top5_err},
                            global_step=len(val_loader) * cur_epoch + cur_iter,
                        )

            val_meter.update_predictions(preds, labels)
            # print("top1_err", top1_err)
            # print("in age:",cfg.AGE)

        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()
    # Log epoch stats.
    val_meter.log_epoch_stats(cur_epoch)
    # write to tensorboard format if available.
    if writer is not None:
        if cfg.DETECTION.ENABLE:
            writer.add_scalars(
                {"Val/mAP": val_meter.full_map}, global_step=cur_epoch
            )
        else:
            all_preds = [pred.clone().detach() for pred in val_meter.all_preds]
            all_labels = [
                label.clone().detach() for label in val_meter.all_labels
            ]
            if cfg.NUM_GPUS:
                all_preds = [pred.cpu() for pred in all_preds]
                all_labels = [label.cpu() for label in all_labels]
            writer.plot_eval(
                preds=all_preds, labels=all_labels, global_step=cur_epoch
            )

    val_meter.reset()


def calculate_and_update_precise_bn(loader, model, num_iters=200, use_gpu=True):
    """
    Update the stats in bn layers by calculate the precise stats.
    Args:
        loader (loader): data loader to provide training data.
        model (model): model to update the bn stats.
        num_iters (int): number of iterations to compute and update the bn stats.
        use_gpu (bool): whether to use GPU or not.
    """

    def _gen_loader():
        for inputs, *_ in loader:
            if use_gpu:
                if isinstance(inputs, (list,)):
                    for i in range(len(inputs)):
                        inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs = inputs.cuda(non_blocking=True)
            yield inputs

    # Update the bn stats.
    update_bn_stats(model, _gen_loader(), num_iters)


def build_trainer(cfg):
    """
    Build training model and its associated tools, including optimizer,
    dataloaders and meters.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
    Returns:
        model (nn.Module): training model.
        optimizer (Optimizer): optimizer.
        train_loader (DataLoader): training data loader.
        val_loader (DataLoader): validatoin data loader.
        precise_bn_loader (DataLoader): training data loader for computing
            precise BN.
        train_meter (TrainMeter): tool for measuring training stats.
        val_meter (ValMeter): tool for measuring validation stats.
    """
    # Build the video model and print model statistics.
    cfg.TRAINING = True
    model = build_model(cfg)
    if cfg.HEAD_TYPE == 'cosine':
        if cfg.NUM_GPUS > 1:
            model.module.model.training = True
        else:
            model.training = True
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = loader.construct_loader(
        cfg, "train", is_precise_bn=True
    )
    # Create meters.
    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)

    return (
        model,
        optimizer,
        train_loader,
        val_loader,
        precise_bn_loader,
        train_meter,
        val_meter,
    )


def train(cfg, args):
    global lambda_0, model_old, score_resultss, index_resultss, index_delta_t, index_delta_s, \
        score_delta_t, score_delta_s, index_resultst, score_resultst, inputs_list, balance_loader

    du.init_distributed_training(cfg)
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging(cfg.OUTPUT_DIR)

    # Init multigrid.
    multigrid = None
    if cfg.MULTIGRID.LONG_CYCLE or cfg.MULTIGRID.SHORT_CYCLE:
        multigrid = MultigridSchedule()
        cfg = multigrid.init_multigrid(cfg)
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, _ = multigrid.update_long_cycle(cfg, cur_epoch=0)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc() and cfg.LOG_MODEL_INFO:
        misc.log_model_info(model, cfg, use_train_input=True)

    # Construct the optimizer.
    optimizer = optim.construct_optimizer(model, cfg)

    # load checkpoint when age>0
    if cfg.AGE > 0:
        age = cfg.AGE - 1
        path = os.path.join(args.output_dir,
                            "checkpoints/checkpoint_age{:02d}.pyth".format(age))
        cu.load_checkpoint(path, model)
        print("load model from age:", age)
        # if cfg.DIST:
        #     model_old = copy.deepcopy(model)
        #     model_old.eval()
        if cfg.HEAD_TYPE == 'cosine':
            cfg.NUM_HEAD = cfg.MODEL.NUM_CLASSES
            if cfg.NUM_GPUS > 1:
                model.module.increment_head(cfg.NUM_HEAD, cfg.AGE)
            else:
                model.increment_head(cfg.NUM_HEAD, cfg.AGE)
            cfg.EMBEDDING = True
            emdding_loader = loader.construct_loader(cfg, "train")
            class_indexer = dict((i, n) for n, i in enumerate(cfg.CLASS_INDEXER))
            init_cosine_classifier(model, cfg.CURRENT_TASK, class_indexer, emdding_loader)
            cfg.EMBEDDING = False
        else:
            if cfg.NUM_GPUS:
                model.module.model.head.load_state_dict(torch.load(os.path.join(cfg.DIR,
                                                                                'complete_head{}.pt'.format(
                                                                                    cfg.AGE - 1))))
            else:
                model.model.head.load_state_dict(torch.load(os.path.join(cfg.DIR,
                                                                         'complete_head{}.pt'.format(cfg.AGE - 1))))

    start_epoch = 0
    # Create the video train and val loaders. train only present task ,test all tasks
    cfg.EMBEDDING = False
    train_loader = loader.construct_loader(cfg, "train")
    list_loader = loader.construct_loader(cfg, "list")
    if cfg.AGE > 0:
        balance_loader = loader.construct_loader(cfg, "balance")
    val_loader = loader.construct_loader(cfg, "val")

    precise_bn_loader = (
        loader.construct_loader(cfg, "train", is_precise_bn=True)
        if cfg.BN.USE_PRECISE_STATS
        else None
    )

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)
    writer = None
    print(cfg.SOLVER.MAX_EPOCH)
    # Perform the training loop.
    print("---------------------------------------------------------")
    logger.info("Start epoch: {}".format(start_epoch + 1))
    print("start testing in age", cfg.AGE)
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        if cfg.MULTIGRID.LONG_CYCLE:
            cfg, changed = multigrid.update_long_cycle(cfg, cur_epoch)
            if changed:
                (
                    model,
                    optimizer,
                    train_loader,
                    val_loader,
                    precise_bn_loader,
                    train_meter,
                    val_meter,
                ) = build_trainer(cfg)

                # Load checkpoint.
                if cu.has_checkpoint(cfg.OUTPUT_DIR):
                    last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
                    assert "{:05d}.pyth".format(cur_epoch) in last_checkpoint
                else:
                    last_checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH
                logger.info("Load from {}".format(last_checkpoint))
                cu.load_checkpoint(
                    last_checkpoint, model, cfg.NUM_GPUS > 1, optimizer
                )
        # Train for one epoch.

        if not args.casual or cfg.AGE == 0:
            (
            score_resultss, score_resultst, score_delta_s, score_delta_t, index_resultss, index_resultst, index_delta_s,
            index_delta_t, inputs_list) = [], [], [], [], [], [], [], [], []
        else:
            inputs_list = get_input(list_loader)
        if args.casual:
            if cfg.AGE > 0 and cur_epoch == 0:
                score_delta_s, index_delta_s, score_delta_t, index_delta_t, \
                    score_resultss, index_resultss, score_resultst, index_resultst = grad_calculate(train_loader, model,
                                                                                                    cfg)
        if cfg.AGE == 0:
            balance_loader = []
        train_epoch(
            train_loader, model, optimizer, train_meter, cur_epoch, cfg, balance_loader,
            score_resultss, index_resultss,
            score_resultst, index_resultst, score_delta_s, score_delta_t, index_delta_s,
            index_delta_t, inputs_list, args, writer)

        is_checkp_epoch = cu.is_checkpoint_epoch(
            cfg,
            cur_epoch,
            None if multigrid is None else multigrid.schedule,
        )
        is_eval_epoch = misc.is_eval_epoch(
            cfg, cur_epoch, None if multigrid is None else multigrid.schedule
        )

        # Compute precise BN stats.
        if (
                (is_checkp_epoch or is_eval_epoch)
                and cfg.BN.USE_PRECISE_STATS
                and len(get_bn_modules(model)) > 0
        ):
            calculate_and_update_precise_bn(
                precise_bn_loader,
                model,
                min(cfg.BN.NUM_BATCHES_PRECISE, len(precise_bn_loader)),
                cfg.NUM_GPUS > 0,
            )
        _ = misc.aggregate_sub_bn_stats(model)
        if not args.glimpse:
            if args.balance_training:
                if cfg.AGE > 0 and cur_epoch == cfg.SOLVER.MAX_EPOCH-1:
                    balance_fintuning(balance_loader, model, cfg, optimizer, args)
                else:
                    torch.save(model.module.model.head.state_dict(),
                               os.path.join(args.output_dir, 'complete_head{}.pt'.format(cfg.AGE)))
            else:
                if cfg.AGE == 0:
                    if cfg.NUM_GPUS == 1:
                        torch.save(model.model.head.state_dict(),
                                   os.path.join(args.output_dir, 'complete_head{}.pt'.format(cfg.AGE)))
                    else:
                        torch.save(model.module.model.head.state_dict(),
                                   os.path.join(args.output_dir, 'complete_head{}.pt'.format(cfg.AGE)))
                else:
                    save_head = Classifier(768, cfg.MODEL.NUM_CLASSES)
                    save_head.head.weight.data[:cfg.NUM_OLD_HEAD] = model.module.model.head.head.weight.data.clone()
                    save_head.head.bias.data[:cfg.NUM_OLD_HEAD] = model.module.model.head.head.bias.data.clone()
                    save_head.head.weight.data[
                    cfg.NUM_OLD_HEAD:cfg.MODEL.NUM_CLASSES] = model.module.model.new_head.head.weight.data.clone()
                    save_head.head.bias.data[
                    cfg.NUM_OLD_HEAD:cfg.MODEL.NUM_CLASSES] = model.module.model.new_head.head.bias.data.clone()
                    torch.save(save_head.state_dict(),
                               os.path.join(args.output_dir, 'complete_head{}.pt'.format(cfg.AGE)))

        print("finish training in age", cfg.AGE)
        print("------------------------------------------")
        print("start testing in age", cfg.AGE)
        if cur_epoch > 0:
            print("final test in age:", cfg.AGE)
        cfg.TRAIN.CHECKPOINT_FILE_PATH = cu.save_final_checkpoint(args.output_dir, model, optimizer, cfg, cfg.AGE)
        if cfg.AGE > 0:
            if cur_epoch > (args.val_epoch - 2):
                cfg.EVAL = True
                eval_epoch(val_loader, model, val_meter, cur_epoch, cfg, writer)
                cfg.EVAL = False
        print("finish testing in age", cfg.AGE)
        print("------------------------------------------")
    if writer is not None:
        writer.close()
