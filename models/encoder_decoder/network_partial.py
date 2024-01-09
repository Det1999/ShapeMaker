import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

from models.atlas.model import mlpAdj, patchDeformationMLP
import models.atlas.utils as atlas_utils

from models.vnn.vn_pointnet import *
from models.vnn.vn_layers import *
from models.vnt.utils.vn_dgcnn_util import get_graph_feature_cross

import models.vnt.vnt_layers as vnt_layers
import utils.pc_utils.pc_utils as pc_utils


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out


class BaseRot(nn.Module):
    def __init__(self, which_strict_rot):
        super(BaseRot, self).__init__()
        self.which_strict_rot = which_strict_rot

    def constraint_rot(self, rot_mat):
        if self.which_strict_rot == 'None':
            return rot_mat
        else:
            return pc_utils.to_rotation_mat(rot_mat)


class SimpleRot_Partial(BaseRot):
    def __init__(self, in_ch, which_strict_rot, rot_samples=5):
        super(SimpleRot_Partial, self).__init__(which_strict_rot)
        self.model = nn.ModuleList()
        self.rot_samples = rot_samples
        for i in range(rot_samples):
            self.model.append(VNLinear(in_ch, 3))

    def forward(self, x):
        rot_mat = []
        for i in range(self.rot_samples):
            rot_mat_now = self.model[i](x).squeeze(-1)
            rot_mat.append(self.constraint_rot(rot_mat_now))
        rot_mat = torch.stack(rot_mat)
        return rot_mat
