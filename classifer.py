import torch
import torch.nn as nn
import torch.nn.functional as F
from timesformer.utils import *
import numpy as np
import time

### https://github.com/imyzx2017/icarl.pytorch/blob/master/icarl.py


def _get_closest(means, features, test_mode, num_crop, target, topk=5):
    num_class = means.size(0)
    if len(features.shape)==3:
        features = features.permute(1,0,2)
        means = means.permute(1,0,2)
    distances = torch.cdist(features,means,p=2)
    if len(distances.shape)==3:
        distances = distances.sum(0)

    if test_mode:
        distances = distances.reshape(-1,num_crop,num_class).mean(1)

    return -distances


def compute_class_mean(model,current_task,exemplar_loader,class_indexer=None):
    print("Computing the class mean vectors...")
    model.eval()
    ex_dict = {}
    std = []
    norm_std = []
    if class_indexer:
        loop_criteria = current_task
        for i in current_task:
            ex_dict[class_indexer[i]] = {}
    else:
        loop_criteria = range(current_task)
        for i in range(current_task):
            ex_dict[i] = {}
    with torch.no_grad():
        for i, (input, target, props, _) in enumerate(exemplar_loader):
            input = input.cuda()
            target = target.cuda()
            model = model.cuda()
            base_out, feat = model(input, True)
            feat = feat.data.cpu()
            del base_out
            # feat = feat.mean(1)

            for j in range(target.size(0)):
                k = props[0][j]
                v = feat[j]
                if int(target[j]) in ex_dict.keys():
                    ex_dict[int(target[j])].update({k:v})

    class_mean_list = []
    feature_list = []

    for i in loop_criteria:
        if class_indexer:
            temp_dict = ex_dict[class_indexer[i]]
        else:
            temp_dict = ex_dict[i]
        features = []

        for k, v in enumerate(temp_dict.items()):
            f_path = v[0]
            feat = v[1]
            feat = feat/torch.norm(feat,p=2)
            features.append(feat)
        if features!= []:
            features = torch.stack(features)
            class_mean = torch.mean(features,axis=0)
            class_mean = class_mean / torch.norm(class_mean,p=2)

            feature_list.append(features)
            class_mean_list.append(class_mean)

    class_means = torch.stack(class_mean_list)

    return class_means, feature_list


