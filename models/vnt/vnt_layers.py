import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6

class classproperty(object):
    def __init__(self, f):
        self.f = f
    def __get__(self, obj, owner):
        return self.f(owner)


class VNTLinear(nn.Module):
    def __init__(self, in_channels, out_channels, which_norm_VNT='softmax'):
        super(VNTLinear, self).__init__()
        self.in_channels = in_channels
        self.which_norm = which_norm_VNT
        self.weight = torch.nn.Parameter(torch.rand(out_channels, in_channels))
        if self.which_norm == 'softmax':
            self.weight.data = F.softmax(self.weight.data, dim=1)
        else:
            self.weight.data = self.weight.data / (torch.sum(self.weight.data, dim=1, keepdim=True) + EPS)

        #Legacy parameter, exist to avoid re-saving the weights
        self.amplification = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        in_channels = x.size(1)
        if in_channels != self.in_channels:
            sys.exit(f'In channels  {in_channels} must equal {self.in_channels}')

        if self.which_norm == 'softmax':
            weight = F.softmax(self.weight, dim=1)
        else:
            weight = self.weight / (torch.sum(self.weight, dim=1, keepdim=True) + EPS)
        x_out = torch.matmul(x.transpose(1,-1), weight.t()).transpose(1,-1)
        return x_out

class VNTLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.2, use_batchnorm=True, which_norm_VNT='softmax'):
        super(VNTLinearLeakyReLU, self).__init__()
        self.dim = dim
        self.negative_slope = negative_slope
        
        self.map_to_feat = VNTLinear(in_channels, out_channels, which_norm_VNT)

        self.use_batchnorm = use_batchnorm
        if use_batchnorm == True:
            self.batchnorm = VNTBatchNorm(out_channels, dim=dim)

        if share_nonlinearity == True:
            self.map_to_dir = VNTLinear(in_channels, 1, which_norm_VNT)
        else:
            self.map_to_dir = VNTLinear(in_channels, out_channels, which_norm_VNT)

        self.map_to_src = VNTLinear(in_channels, 1, which_norm_VNT)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # Linear
        p = self.map_to_feat(x)
        # BatchNorm
        if self.use_batchnorm == True:
            p = self.batchnorm(p)
        # LeakyReLU
        d = self.map_to_dir(x)
        o = self.map_to_src(x)
        d = d - o
        dotprod = ((p - o) * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d*d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1-self.negative_slope) * (mask*p + (1-mask)*(p-(dotprod/(d_norm_sq+EPS))*d))
        return x_out


class VNTBatchNorm(nn.Module):
    def __init__(self, num_features, dim):
        super(VNTBatchNorm, self).__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        # norm = torch.sqrt((x*x).sum(2))
        if  self.dim > 3:
            mean_val = torch.mean(x, dim=3, keepdim=True)
            x = x - mean_val
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        x = x / norm * norm_bn
        if self.dim > 3:
            x = x + mean_val
        return x


class VNTMaxPool(nn.Module):
    def __init__(self, in_channels, which_norm_VNT='softmax'):
        super(VNTMaxPool, self).__init__()
        self.map_to_dir = VNTLinear(in_channels, in_channels, which_norm_VNT)
        self.map_to_src = VNTLinear(in_channels, 1, which_norm_VNT)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        d = self.map_to_dir(x)
        o = self.map_to_src(x)
        dotprod = ((x-o) * (d-o)).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]]) + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNTStdFeature(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2, use_batchnorm=True, which_norm_VNT='softmax'):
        super(VNTStdFeature, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        
        self.vn1 = VNTLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm, which_norm_VNT=which_norm_VNT)
        self.vn2 = VNTLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm, which_norm_VNT=which_norm_VNT)
        if normalize_frame:
            self.vn_lin = VNTLinear(in_channels//4, 2, which_norm_VNT)
        else:
            self.vn_lin = VNTLinear(in_channels//4, 3, which_norm_VNT)

        self.mlp_to_src = VNTLinearLeakyReLU(in_channels, 1, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope, use_batchnorm=use_batchnorm, which_norm_VNT=which_norm_VNT)

    def forward(self, x):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        x = x - torch.mean(x, dim=-1, keepdim=True)
        z0 = x
        z0 = self.vn1(z0)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0)
        #o = self.mlp_to_src(x)

        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        #x = x - o
        #z0 = z0 - o
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)


        return x_std, z0#, o