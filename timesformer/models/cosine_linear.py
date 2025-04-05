# From Official LUCIR Code
import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module


def stable_cosine_distance(a, b, squared=True):
    # From PODNet
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]

def _reduce_proxies(similarities, num_proxy):
    # shape (batch_size, n_classes * proxy_per_class)
    n_classes = similarities.shape[1] / num_proxy
    assert n_classes.is_integer(), (similarities.shape[1], num_proxy)
    n_classes = int(n_classes)
    bs = similarities.shape[0]

    simi_per_class = similarities.view(bs, n_classes, num_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)
    return (simi_per_class * attentions).sum(-1)

class CosineLinear(Module):
    def __init__(self, in_features, out_features, num_proxy=1, sigma_learnable=True, sigma=1.0,
            eta_learnable=True, eta=1.0, version='lsc', nca_margin=0.6, is_train=True):
        super(CosineLinear, self).__init__()
        self.version = version
        self.num_proxy = num_proxy
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.num_proxy * out_features, in_features))

        self.sigma_learnable = sigma_learnable
        self.sigma = sigma
        if self.sigma_learnable:
            self.sigma = Parameter(torch.ones(1))

        self.eta_learnable = eta_learnable
        self.eta = eta

        if self.eta_learnable:
            self.eta = Parameter(torch.ones(1))
        self.nca_margin = nca_margin
        self.reset_parameters()
        self.is_train = is_train

    def reset_parameters(self):
        if self.version == 'lsc':
            nn.init.kaiming_normal_(self.weight, nonlinearity='linear')
        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        if isinstance(self.sigma, Parameter):
            self.sigma.data.fill_(1) #for initializaiton of sigma
        if self.eta and isinstance(self.eta, Parameter):
            self.eta.data.fill_(1)

    def forward(self, input, training, sigma_from_wrapper=None):
        if self.version == 'cc':
            out = F.linear(F.normalize(input, p=2,dim=1), \
                    F.normalize(self.weight, p=2, dim=1))
            out = self.sigma * out
        elif self.version == 'lsc':
            if sigma_from_wrapper is None:
                features = self.sigma * F.normalize(input,p=2,dim=1).cuda()
                weights = self.sigma * F.normalize(self.weight,p=2,dim=1).cuda()
                out = - stable_cosine_distance(features, weights)
                out = _reduce_proxies(out, self.num_proxy)
                if self.is_train:
                    out = self.eta * (out - self.nca_margin)
            else:
                features = sigma_from_wrapper * F.normalize(input,p=2,dim=1)
                weights = sigma_from_wrapper * F.normalize(self.weight,p=2,dim=1)
                out = - stable_cosine_distance(features, weights)
                out = _reduce_proxies(out, self.num_proxy)
        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, sigma={}, eta={}'.format(
            self.in_features, self.num_proxy*self.out_features,
            self.sigma.data if self.sigma_learnable else self.sigma,
            self.eta.data if self.eta_learnable else self.eta
        )

class SplitCosineLinear(Module):
    #consists of two fc layers and concatenate their outputs
    def __init__(self, in_features, out_features1, out_features2, num_proxy=1, sigma_learnable=True, sigma=1.0,
            eta_learnable=False, version='lsc', nca_margin=0.6, is_train=True):
        super(SplitCosineLinear, self).__init__()
        self.version = version
        self.in_features = in_features
        self.out_features = out_features1 + out_features2
        self.num_proxy = num_proxy
        self.fc1 = CosineLinear(in_features, out_features1, self.num_proxy,
                False, 1.0, False, 1.0, version=self.version)
        self.fc2 = CosineLinear(in_features, out_features2, self.num_proxy,
                False, 1.0, False, 1.0, version=self.version)
        self.sigma_learnable = sigma_learnable
        self.sigma = sigma
        self.eta_learnable = eta_learnable
        # self.eta = eta

        self.sigma = Parameter(torch.Tensor(1))
        self.sigma.data.fill_(1)

        self.eta = Parameter(torch.Tensor(1))

        self.nca_margin = nca_margin
        self.is_train = is_train

    def activite_parameters(m, requires_grad=True):
        if m is None:
            return

        if isinstance(m, nn.Parameter):
            m.requires_grad = requires_grad
        else:
            for p in m.parameters():
                p.requires_grad = requires_grad

    def activite(self):
        return self.freeze_parameters(self)

    def forward(self, x, training):
        if self.version == 'cc':
            out1 = self.fc1(x)
            out2 = self.fc2(x)
            out = torch.cat((out1, out2), dim=1) #concatenate along the channel
            if self.sigma is not None:
                out = self.sigma * out
        elif self.version == 'lsc': # for pod (nca_loss)...
            out1 = self.fc1(x, 0.1)
            out2 = self.fc2(x, 0.1)
            out = torch.cat((out1,out2),dim=1)
            if training:
                out = self.eta * (out - self.nca_margin)

        return out

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, sigma={}, eta={}'.format(
            self.in_features, self.num_proxy * self.out_features,
            self.sigma.data if self.sigma_learnable else self.sigma,
            self.eta.data if self.eta_learnable else self.eta
        )

