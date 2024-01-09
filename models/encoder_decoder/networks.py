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


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                if hasattr(m, 'no_init_bias') and m.no_init_bias:
                    return
                else:
                    init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def net_to_device(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    return net


def define_encoder(opt):
    net = None
    if opt.net_e == 'simple':
        net = VNTSimpleEncoder(opt, global_feat=True, feature_transform=True)
    net = net_to_device(net, opt.gpu_ids)
    return net


def define_decoder(opt):
    net = None
    if opt.net_d == 'point':
        net = PointTransMLPAdjDecoder(opt)
        net.apply(atlas_utils.weights_init)
    elif opt.net_d == 'patch':
        net = PatchDeformMLPAdjDecoder(opt)
        net.apply(atlas_utils.weights_init)

    net = net_to_device(net, opt.gpu_ids)
    return net


def define_rot_module(net_rot, nlatent, which_strict_rot, gpu_ids, rot_samples=None):
    net = None
    if net_rot == 'simple':
        net = SimpleRot(nlatent // 2 // 3, which_strict_rot)
    elif net_rot == 'complex':
        net = ComplexRot((nlatent // 3), which_strict_rot)
    elif net_rot == 'simple_partial':
        net = SimpleRot_Partial(nlatent // 2 // 3, which_strict_rot, rot_samples)
    else:
        raise NotImplementedError(f"rot module {net_rot} not supported")

    net = net_to_device(net, gpu_ids)
    return net


##############################################################################
# Classes
##############################################################################
class RotationLoss(nn.Module):
    """Define Rotation loss between non-orthogonal matrices.
    """

    def __init__(self, device, which_metric='MSE'):
        super(RotationLoss, self).__init__()
        self.device = device
        self.indentity_rot = torch.eye(3, device=self.device).unsqueeze(0)

        self.which_metric = which_metric
        assert self.which_metric in ['cosine', 'angular', 'orthogonal', 'MSE'], f'{self.which_metric} invalid rot loss'

        if self.which_metric == 'cosine':
            self.metric = torch.nn.CosineSimilarity(dim=2)
        else:
            self.metric = torch.nn.MSELoss()

    def batched_trace(self, mat):
        return mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

    def cosine_loss(self, R1, R2):
        return torch.mean(1 - self.metric(R1, R2))

    def angular_loss(self, R1, R2):
        M = torch.matmul(R1, R2.transpose(1, 2))
        return torch.mean(torch.acos(torch.clamp((self.batched_trace(M) - 1) / 2., -1 + EPS, 1 - EPS)))

    def __call__(self, R1, R2):
        # Input:
        #    R1, R2 - Bx3x3 (dim=1 - channels, dim=2 - xyz)
        # Output:
        #    loss - torch tensor

        if self.which_metric == 'cosine':
            return self.cosine_loss(R1, R2)
        elif self.which_metric == 'angular':
            return self.angular_loss(R1, R2)
        elif self.which_metric == 'orthogonal':
            return self.criterionMSE(torch.matmul(R1, R2.transpose(1, 2)),
                                     self.indentity_rot.expand(R1.size(0), 3, 3))
        else:
            return self.metric(R1, R2)


class DirichletLoss(nn.Module):
    """Define symmetric dirichlet loss.
    """

    def __init__(self, device):
        super(DirichletLoss, self).__init__()
        self.device = device

    def __call__(self, R):
        RTR = torch.matmul(R.transpose(1, 2), R)
        RTRinv = torch.inverse(RTR)

        # Explicit:
        # a = RTR[:, 0, 0]
        # b = RTR[:, 0, 1]
        # c = RTR[:, 0, 2]
        # d = RTR[:, 1, 0]
        # e = RTR[:, 1, 1]
        # f = RTR[:, 1, 2]
        # g = RTR[:, 2, 0]
        # h = RTR[:, 2, 1]
        # i = RTR[:, 2, 2]
        #
        # RTRinv = torch.zeros_like(RTR, device=self.device)
        # RTRinv[:, 0, 0] = e * i - f * h
        # RTRinv[:, 0, 1] = -1 * (d * i -f * g)
        # RTRinv[:, 0, 2] = d * h - g * e
        # RTRinv[:, 1, 0] = -1 * (b * i - c * h)
        # RTRinv[:, 1, 1] = a * i - c * g
        # RTRinv[:, 1, 2] = -1 * (a * h - b * g)
        # RTRinv[:, 2, 0] = f * b - c * e
        # RTRinv[:, 2, 1] = -1 * (a * f - c * d)
        # RTRinv[:, 2, 2] = a * e - b * d
        #
        # det = a * RTRinv[:, 0, 0] + b * RTRinv[:, 0, 1] + c * RTRinv[:, 0, 2] + 1e-15
        # RTRinv = RTRinv.transpose(1,2) / det.unsqueeze(-1).unsqueeze(-1)

        rigidity_loss = (RTR ** 2).sum(1).sum(1).sqrt() + (RTRinv ** 2).sum(1).sum(1).sqrt()

        return rigidity_loss.mean()


class OrthogonalLoss(nn.Module):
    """Define orthogonal loss for non-orthogonal matrix.
    """

    def __init__(self, device, which_metric='MSE'):
        super(OrthogonalLoss, self).__init__()
        self.device = device
        self.indentity_rot = torch.eye(3, device=self.device).unsqueeze(0)

        self.which_metric = which_metric
        assert self.which_metric in ['dirichlet', 'svd', 'MSE'], f'{self.which_metric} invalid ortho loss'

        if self.which_metric == 'dirichlet':
            self.metric = self.DirichletLoss(self.device)
        else:
            self.metric = torch.nn.MSELoss()

    def __call__(self, R1):
        # Input:
        #    R1, R2 - Bx3x3 (dim=1 - channels, dim=2 - xyz)
        # Output:
        #    loss - torch tensor

        if self.which_metric == 'dirichlet':
            return self.metric(R1)
        elif self.which_metric == 'svd':
            u, s, v = torch.svd(R1)
            return self.metric(R1, torch.matmul(u, v.transpose(1, 2)))
        else:
            return self.metric(torch.matmul(R1, R1.transpose(1, 2)),
                               self.indentity_rot.expand(R1.size(0), 3,
                                                         3))


class PatchDeformMLPAdjDecoder(nn.Module):
    """Atlas net auto decoder"""

    def __init__(self, options):
        super(PatchDeformMLPAdjDecoder, self).__init__()

        self.npoint = options.npoint
        self.npatch = options.npatch
        self.nlatent = options.nlatent
        self.patchDim = options.patchDim
        self.patchDeformDim = options.patchDeformDim

        # encoder_decoder decoder and patch deformation module
        # ==============================================================================
        self.decoder = nn.ModuleList(
            [mlpAdj(nlatent=self.patchDeformDim + self.nlatent) for i in range(0, self.npatch)])
        self.patchDeformation = nn.ModuleList(
            patchDeformationMLP(patchDim=self.patchDim, patchDeformDim=self.patchDeformDim) for i in
            range(0, self.npatch))
        # ==============================================================================

    def forward(self, x):
        outs = []
        patches = []
        for i in range(0, self.npatch):
            # random planar patch
            # ==========================================================================
            rand_grid = torch.FloatTensor(x.size(0), self.patchDim, self.npoint // self.npatch).cuda()
            rand_grid.data.uniform_(0, 1)
            rand_grid[:, 2:, :] = 0
            rand_grid = self.patchDeformation[i](rand_grid.contiguous())
            patches.append(rand_grid[0].transpose(1, 0))
            # ==========================================================================

            # cat with latent vector and decode
            # ==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            # ==========================================================================

        return torch.cat(outs, 2).transpose(2, 1).contiguous(), patches


class PointTransMLPAdjDecoder(nn.Module):
    """Atlas net decoder"""

    def __init__(self, options, mode='full'):

        super(PointTransMLPAdjDecoder, self).__init__()

        if mode == 'base' or mode == 'partial':
            self.npoint = options.base_decoder_npoint
        elif mode == 'full':
            self.npoint = options.full_resample_npoints
        elif mode == 'cross':
            self.npoint = options.partial_resample_npoints
        else:
            self.npoint = 1024
        self.npatch = options.base_decoder_npatch
        self.nlatent = options.base_decoder_nlatent
        self.dim = options.base_decoder_patchDim

        # encoder_decoder and decoder modules
        # ==============================================================================
        self.decoder = nn.ModuleList([mlpAdj(nlatent=self.dim + self.nlatent) for i in range(0, self.npatch)])
        # ==============================================================================

        # patch
        # ==============================================================================
        self.grid = []
        for patchIndex in range(self.npatch):
            patch = torch.nn.Parameter(torch.FloatTensor(1, self.dim, self.npoint // self.npatch))
            patch.data.uniform_(0, 1)
            patch.data[:, 2:, :] = 0
            self.register_parameter("patch%d" % patchIndex, patch)
            self.grid.append(patch)
        # ==============================================================================

    def forward(self, x):
        outs = []
        patches = []

        for i in range(0, self.npatch):
            # random planar patch
            # ==========================================================================
            rand_grid = self.grid[i].expand(x.size(0), -1, -1)
            patches.append(rand_grid[0].transpose(1, 0))
            # ==========================================================================

            # cat with latent vector and decode
            # ==========================================================================
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
            # ==========================================================================

        return torch.cat(outs, 2).transpose(2, 1).contiguous(), patches


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


class SimpleRot(BaseRot):
    def __init__(self, in_ch, which_strict_rot):
        super(SimpleRot, self).__init__(which_strict_rot)
        self.model = VNLinear(in_ch, 3)

    def forward(self, x):
        rot_mat = self.model(x).squeeze(-1)
        return self.constraint_rot(rot_mat)


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


class ComplexRot(BaseRot):
    def __init__(self, in_ch, which_strict_rot):
        super(ComplexRot, self).__init__(which_strict_rot)
        self.linear1 = VNLinearLeakyReLU(in_ch, in_ch, dim=4, negative_slope=0.0)
        self.linear2 = VNLinearLeakyReLU(in_ch, in_ch // 2, dim=4, negative_slope=0.0)
        self.linear3 = VNLinearLeakyReLU(in_ch // 2, in_ch // 2, dim=4, negative_slope=0.0)
        self.linearR = VNLinear(in_ch // 2, 3)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        R = self.linearR(x)
        rot_mat = torch.mean(R, dim=-1)
        return self.constraint_rot(rot_mat)


class VNTSimpleEncoder(nn.Module):
    def __init__(self, args, global_feat=True, feature_transform=False):
        super(VNTSimpleEncoder, self).__init__()
        self.args = args
        self.n_knn = args.base_encoder_n_knn
        which_norm_VNT = args.base_encoder_which_norm_VNT
        base_ch = args.base_encoder_base_ch
        output_ch = args.base_encoder_nlatent // 2
        self.conv_pos = vnt_layers.VNTLinearLeakyReLU(3, base_ch // 3, dim=5, negative_slope=0.0,
                                                      which_norm_VNT=which_norm_VNT)
        self.conv_center_ = vnt_layers.VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0,
                                                          which_norm_VNT=which_norm_VNT)
        self.conv_center = vnt_layers.VNTLinearLeakyReLU(base_ch // 3, 1, dim=4, negative_slope=0.0,
                                                         which_norm_VNT=which_norm_VNT)

        self.conv1 = VNLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(base_ch // 3 * 2, (2 * base_ch) // 3, dim=4, negative_slope=0.0)

        self.conv3 = VNLinear((2 * base_ch) // 3, output_ch // 3)
        self.bn3 = VNBatchNorm(output_ch // 3, dim=4)

        self.std_feature = VNStdFeature(output_ch // 3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)

        if args.base_encoder_pooling == 'max':
            self.pool = vnt_layers.VNTMaxPool(base_ch // 3, which_norm_VNT=which_norm_VNT)
        elif args.base_encoder_pooling == 'mean':
            self.pool = mean_pool

        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(args, d=base_ch // 3)

    def forward(self, x, mode='normal'):
        B, D, N = x.size()

        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.n_knn, use_global=self.args.base_encoder_global_bias)
        x = self.conv_pos(feat)
        x = self.pool(x)
        x_center = self.conv_center(self.conv_center_(x))
        center_loc = torch.mean(x_center, dim=-1, keepdim=True)

        x = x - x_center
        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        x_mean_out = x.mean(dim=-1, keepdim=True)
        x_mean = x_mean_out.expand(x.size())
        x = torch.cat((x, x_mean), 1)

        x, trans = self.std_feature(x)

        x = x.view(B, -1, N)
        x_bef = x.clone()
        x = torch.max(x, -1, keepdim=False)[0]
        if mode == 'normal':
            return x, None, x_mean_out, center_loc.squeeze(-1)
        else:
            return x, x_bef.permute(0, 2, 1), x_mean_out, center_loc.squeeze(-1)
