import time
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from tensorboardX import SummaryWriter
from dataset.shapenetv1 import Shapes
from models.I_RED_full import I_RED_full
from models.I_RED_partial import I_RED_partial
from models.I_RED_rd import I_RED_rd
from models.I_RED_rd_cage import I_RED_rd_cage
import itertools

import argparse
from options.base_options import BaseOptions, combine_flags
from options.full_options import FullOptions
from options.partial_options import PartialOptions
from options.cross_options import CrossOptions
from options.dataset_options import ShapeNetV1_options
import os

from losses.partial_loss import Partial_loss

import torch.nn.functional as F
from pytorch3d import loss
from losses.supp_loss_funcs import RotationLoss, OrthogonalLoss, u_chamfer_distance
from pytorch3d.ops.knn import knn_points
from visualization.vis_pc import show_pc,show_pc_kp,show_pc_seg


torch.autograd.set_detect_anomaly(False)


def init_data_opt_augment(opt):
    ######## full ########
    opt.full_downsample = False
    opt.full_add_noise = False
    opt.full_remove_knn = False
    opt.full_fps = False
    opt.full_resample = False
    opt.full_randRT = False
    opt.full_apply_can_rot = False
    
    ######## partial #######
    opt.partial_add_noise = True
    opt.partial_fps = False
    opt.partial_remove_knn = False
    opt.partial_resample = False
    opt.partial_randRT = True
    opt.partial_dual = False
    opt.partial_apply_can_rot = True
    
    return opt

def occu_select(opt):
    N_points = 2500
    Now_points = int(N_points * 0.75)
    opt.base_decoder_npoint = Now_points
    opt.base_resample_npoints = Now_points
    opt.partial_resample_npoints = Now_points
    
    return opt

def train():
    arg_parser = argparse.ArgumentParser()
    base_options = BaseOptions()
    full_options = FullOptions()
    partial_options = PartialOptions()
    cross_options = CrossOptions()
    dataset_options = ShapeNetV1_options()
    arg_parser = base_options.initialize(arg_parser)
    arg_parser = full_options.initialize(arg_parser)
    arg_parser = partial_options.initialize(arg_parser)
    arg_parser = cross_options.initialize(arg_parser)

    opt = arg_parser.parse_args()
    
    opt = occu_select(opt)
    opt.config = 'configs/Tables.yaml'
    opt = combine_flags(opt, config_path=opt.config)
    log_dir = opt.log_dir+'_partial_rd_0.75' + '/' + opt.name
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(logdir=log_dir)
    opt_dataset = dataset_options.combine_configs(file_path=opt.config)
    
    ###  create shape dataset
    dataset = Shapes(opt_dataset)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.cross_batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   collate_fn=dataset.collate,
                                                   num_workers=4,
                                                   worker_init_fn=(lambda id:
                                                                   np.random.seed(np.random.get_state()[1][0] + id)))
    
    ### create model
    opt = init_data_opt_augment(opt)
    model_full = I_RED_full(opt).to(opt.base_device)
    model_partial = I_RED_partial(opt).to(opt.base_device)
    model_RandD = I_RED_rd_cage(opt).to(opt.base_device)
    s_epoch = 0
    t_epoch = 200
    lr_first_decay = 140
    lr_second_decay = 180
    
    ######################################################################################################################### fix the parameters of the model_full
    model_full.load_state_dict(torch.load('Table/model_199.pth'))
    model_full.eval()
    for param in model_full.parameters():
        param.requires_grad = False
    
    
    
    model_RandD.load_state_dict(torch.load('Table_FULL_rd_kp_seg/table-12kpt/model_randd_199.pth'),False)
    
    
    for param in model_RandD.full_keypoint_net.parameters():
        param.requires_grad = False
    for param in model_RandD.full_seg_net.parameters():
        param.requires_grad = False
    #########################################################################################################################
    
    ### init train
    total_iters = 0
    global_step = 0
    
    params_partial = model_partial.get_optim_params()
    params_rd = model_RandD.get_optim_params()
    params = itertools.chain(params_partial, params_rd)
    optimizer = torch.optim.Adam(params, lr=opt.cross_lr)
    lr_milestones = [lr_first_decay, lr_second_decay]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    Partial_loss_func = Partial_loss(opt)
    
    Consis_loss_func = Consis_loss(opt)
    
    for epoch in range(s_epoch, t_epoch):
        st_time = time.time()
        for i, data in enumerate(train_dataloader):
            total_iters += 1
            #################################################################
            #                      data generation                          #
            #################################################################
            pc_full = data['target_shape']
            source_full = data['source_shape']
            input_list_full = model_full.online_data_augment(pc_full)
            pc = data['target_partial_shape']
            pc_dual = data['target_partial_shape_dual']
            pc_f_p_list = data['target_partial_mask']
            input_list_partial = model_partial.online_data_augment(pc, pc_dual,full_input_list=input_list_full)
            
            #################################################################
            #                      full model infer                         #
            #################################################################
            with torch.no_grad():
                pred_full_recon, _, _, _, _, _, _, pred_full_RandD = model_full(input_list_full, mode='cross')
            
            #################################################################
            #                      partial model infer                      #
            #################################################################
            input_list_partial['full_pc_full'] = input_list_full['pc_full']
            
            pred_p_recon, pred_p_add_noise, pred_p_fps, pred_p_remove_knn, pred_p_resample,\
            pred_p_randRT, pred_p_can_rot, pred_p_dual, pred_p_RandD = model_partial(input_list_partial, mode='cross')
            ####################
            loss_list_partial, loss_sum_partial = Partial_loss_func.get_all_loss_terms(pred_p_recon, pred_p_add_noise,
                                                                                       pred_p_fps, pred_p_remove_knn,
                                                                                       pred_p_resample, pred_p_randRT,
                                                                                       pred_p_can_rot, pred_p_dual)
            
            #################################################################
            #                          R&D model infer                      #
            #################################################################
            pred_rd_full, pred_rd_partial= model_RandD(pred_full_RandD, pred_p_RandD)
            loss_list_rd, loss_sum_rd = Consis_loss_func.get_all_loss_terms(pred_rd_full, pred_rd_partial, pc_f_p_list)
            
            
            loss_sum = loss_sum_rd + loss_sum_partial
            
            loss_sum.backward()
            
            torch.nn.utils.clip_grad_norm_(model_partial.parameters(), 5.)
            torch.nn.utils.clip_grad_norm_(model_RandD.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()

            if total_iters % opt.full_log_intervals == 0:
                writer.add_scalar(tag='sum_loss_partial', scalar_value=loss_sum_partial, global_step=global_step)
                writer.add_scalar(tag='sum_loss_rd', scalar_value=loss_sum_rd, global_step=global_step)
                global_step += 1
                print('recon_full: {0:04f}, keypoint: {1:04f}, feature_x: {2:04f}, kp_recon: {3:04f}'.format(loss_list_rd['recon_full'],loss_list_rd['kp'],loss_list_rd['feat_x'],loss_list_rd['kp_recon']))
                

        end_time = time.time()
        # update according to the epoch
        scheduler.step()
        if (epoch + 1) % opt.cross_save_intervals == 0 or (epoch + 1) == opt.cross_total_epochs:
            torch.save(model_partial.state_dict(), '{0}/model_partial_{1:02d}.pth'.format(log_dir, epoch))
            torch.save(model_RandD.state_dict(), '{0}/model_randd_{1:02d}.pth'.format(log_dir, epoch))
        print('Training Epoch: {0} Finished, using {1:04f}.'.format(epoch, end_time - st_time))


class Consis_loss_(nn.Module):
    def __init__(self, opt):
        super(Consis_loss_, self).__init__()
        self.opt = opt
        self.criterionPCRecon = loss.chamfer_distance
        self.criterionMSE = nn.MSELoss()
        self.criterionCosine = F.cosine_similarity
        self.criterionROT = RotationLoss(device=opt.base_device, which_metric=opt.partial_which_rot_metric)
        self.criterionOrtho = OrthogonalLoss(device=opt.base_device, which_metric=opt.partial_which_ortho_metric)

    #  the point segmentation consistency loss term for the full and partial branches
    def cal_seg_loss(self, pred_rd_full, pred_rd_partial, fp_list = None):
        pc_seg_full = pred_rd_full['pc_seg']
        pc_seg_partial = pred_rd_partial['pc_seg']
        # pc_seg_full    bs x nofP_full x nofkp
        # fp_list: the corresponding list of each partial point to its full point
        # fp_list: bsx nof_partial
        seg_loss = []
        bs, nofp_f, nofKP = pc_seg_full.shape
        for i in range(bs):
            pc_seg_full_now = pc_seg_full[i]  # nofP_full x nofKP
            
            #####DEBUG fp_list out idx 
            if fp_list != None:
                pc_seg_full_to_partial = pc_seg_full_now[fp_list[i], :]
            else:
                pc_seg_full_to_partial = pc_seg_full_now
            
            pc_seg_partial_now = pc_seg_partial[i]
            seg_loss_now = 1. - self.criterionCosine(pc_seg_partial_now, pc_seg_full_to_partial)
            seg_loss_now = seg_loss_now.mean()
            seg_loss.append(seg_loss_now)
        seg_loss = torch.stack(seg_loss)
        return torch.sum(seg_loss) / (bs + 1) * self.opt.cross_weight_seg

    #  we also supervise the extracted pointwise features. Corresponding point features in the full and partial
    #  branch should be the same.
    def cal_feature_loss(self, pred_rd_full, pred_rd_partial, fp_list):
        pc_f_full = pred_rd_full['inv_f_p']
        pc_f_partial = pred_rd_partial['inv_f_p']
        # pc_f_full : bs x nofP_full x c_dim
        # pc_f_partial: bs x nofP_full x c_dim
        bs, nofP_full, _ = pc_f_full.shape
        feat_loss = []
        for i in range(bs):
            f_full_now = pc_f_full[i]  # nofP_full x c_dim
            f_full_to_partial = f_full_now[fp_list[i], :]

            f_partial_now = pc_f_partial[i]
            feat_loss_now = self.criterionMSE(f_full_to_partial, f_partial_now)
            feat_loss.append(feat_loss_now)
        feat_loss = torch.stack(feat_loss)
        feat_loss = torch.sum(feat_loss) / (bs + 1)
        
        pc_ff_full = pred_rd_full['inv_f']
        pc_ff_partial = pred_rd_partial['inv_f']
        feat_f_loss = self.criterionMSE(pc_ff_full,pc_ff_partial)
        
        
        return feat_loss * self.opt.cross_weight_feature

    #  supervise the reconstructed point cloud in the canonical space. The partial reconstruction should be a part of
    #  the full reconstruction
    def cal_recon_loss(self, pred_rd_full, pred_rd_partial, fp_list = None):
        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['recon_pc']
        if fp_list == None:
            loss_recon, _, _ = u_chamfer_distance(pc_partial, pc_full)
        else:
            bs, nofP_full, _ = pc_full.shape
            recon_loss = []
            for i in range(bs):
                pc_full_now = pc_full[i]
                pc_full_to_partial = pc_full_now[fp_list[i],:].unsqueeze(0)
                pc_partial_now = pc_partial[i].unsqueeze(0)
                recon_loss_now = self.criterionPCRecon(pc_partial_now,pc_full_to_partial)[0]
                recon_loss.append(recon_loss_now)
            recon_loss = torch.stack(recon_loss)
            loss_recon = torch.sum(recon_loss) / (bs + 1)
        
        return loss_recon * self.opt.cross_weight_recon
    
    def cal_recon_full_loss(self,pred_rd_full, pred_rd_partial):
        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['full_pc_at_inv']
        
        recon_full_loss = self.criterionPCRecon(pc_partial,pc_full)[0]
        
        return recon_full_loss * self.opt.cross_weight_recon
        
    #  supervise the direction vectors between keypoints. A consistency term between the full and partial branches
    #
    def cal_part_direction_loss(self, pred_rd_full, pred_rd_partial):
        kp_full = self.get_partial_real_kp(pred_rd_full,pred_rd_partial)
        #kp_full = pred_rd_full['keypoints']
        kp_partial = pred_rd_partial['keypoints']
        kp_dir_loss = []
        bs, nofKP, _ = kp_full.shape
        for i in range(nofKP):
            for j in range(i + 1, nofKP):
                kp_full_dir = kp_full[:, i, :] - kp_full[:, j, :]
                kp_partial_dir = kp_partial[:, i, :] - kp_partial[:, j, :]
                kp_dir_error = torch.mean((kp_full_dir - kp_partial_dir) * (kp_full_dir - kp_partial_dir))
                kp_dir_loss.append(kp_dir_error)
        nofnum = len(kp_dir_loss) + 1
        kp_dir_loss = torch.stack(kp_dir_loss)
        kp_dir_loss = torch.sum(kp_dir_loss) / nofnum * self.opt.cross_weight_pd

        return kp_dir_loss

    # extracted keypoints should be consistent between the two branches
    #>>>> keypoints
    def cal_keypoints_loss(self, pred_rd_full, pred_rd_partial):
        #kp_full = pred_rd_full['keypoints']
        kp_full = self.get_partial_real_kp(pred_rd_full,pred_rd_partial)
        kp_partial = pred_rd_partial['keypoints']
        
        #   kp_full: bs x nofKP x 3
        #   kp_partial: bs x nofKP x 3
        bs, nofKP, _ = kp_full.shape
        kp_loss_p = F.mse_loss(kp_full, kp_partial)
        return kp_loss_p * self.opt.cross_weight_kp
    
    def cal_kp_recon_loss(self, pred_rd_partial):
        kp_full = pred_rd_partial['keypoints']
        recon_pc_full = pred_rd_partial['full_pc_at_inv']
        loss_kp_recon_full, _ = self.criterionPCRecon(kp_full, recon_pc_full)
        return loss_kp_recon_full
    
    # keypoints extracted should be better in the center of its support region
    def cal_kp_seg_loss(self, pred_rd_full, pred_rd_partial, min_support_pixels=20):
        kp_full = pred_rd_full['keypoints']
        pc_seg_full = pred_rd_full['pc_seg']  # bs x nofP x nofkp
        recon_pc_full = pred_rd_full['recon_pc']  # bs x nofP x 3
        bs, nofP, nofKP = pc_seg_full.shape
        kp_support_pixel_num_full = torch.sum(pc_seg_full.detach(), dim=1)  # soft arrangement, bs x nofkp
        # bs x nfkp x 3
        kp_seg_full_loss = []
        kp_seg_partial_loss = []
        pc_seg_center_full = (pc_seg_full.permute(0, 2, 1) @ recon_pc_full)  # bs  nofKP x 3

        kp_partial = pred_rd_partial['keypoints']
        pc_seg_partial = pred_rd_partial['pc_seg']
        recon_pc_partial = pred_rd_partial['recon_pc']
        kp_support_pixel_num_partial = torch.sum(pc_seg_partial.detach(), dim=1)
        pc_seg_center_partial = (pc_seg_partial.permute(0, 2, 1) @ recon_pc_partial)

        for i in range(bs):
            for j in range(nofKP):
                pc_seg_center_now = pc_seg_center_full[i, j, :]
                pc_seg_center_support = kp_support_pixel_num_full[i, j]
                if pc_seg_center_support < min_support_pixels:
                    continue
                pc_seg_center_now = pc_seg_center_now / pc_seg_center_support
                kp_full_now = kp_full[i, j, :]
                kp_seg_full_loss_now = torch.sum((kp_full_now - pc_seg_center_now) * (kp_full_now - pc_seg_center_now))
                kp_seg_full_loss.append(kp_seg_full_loss_now)

                pc_seg_center_now = pc_seg_center_partial[i, j, :]
                pc_seg_center_support = kp_support_pixel_num_partial[i, j]
                if pc_seg_center_support < min_support_pixels:
                    continue
                pc_seg_center_now = pc_seg_center_now / pc_seg_center_support
                kp_partial_now = kp_partial[i, j, :]
                kp_seg_partial_loss_now = torch.sum((kp_partial_now - pc_seg_center_now) *
                                                    (kp_partial_now - pc_seg_center_now))
                kp_seg_partial_loss.append(kp_seg_partial_loss_now)

        nofL_partial = len(kp_seg_partial_loss)
        nofL_full = len(kp_seg_full_loss)
        if nofL_partial == 0 or nofL_full == 0:
            return 0.0
        kp_seg_full_loss = torch.stack(kp_seg_full_loss)
        kp_seg_partial_loss = torch.stack(kp_seg_partial_loss)

        return (torch.sum(kp_seg_partial_loss) +
                torch.sum(kp_seg_full_loss)) / (nofL_partial + nofL_full) * self.opt.cross_weight_kp_seg
    
    def cal_mat_loss(self,pred_rd_full,pred_rd_partial):
        full_rot = pred_rd_full['rot']
        full_t = pred_rd_full['t']
        partial_rot = pred_rd_partial['rot']
        partial_t = pred_rd_full['t']
        
        rot_mat = self.criterionROT(full_rot,partial_rot)
        t_mat = self.criterionMSE(full_t,partial_t)
        
        return rot_mat+t_mat
    
    def get_partial_real_kp(self,pred_rd_full, pred_rd_partial):
        
        full_kp = pred_rd_full['keypoints']
        full_rot = pred_rd_full['rot']
        full_t = pred_rd_full['t']
        partial_rot = pred_rd_partial['rot']
        partial_t = pred_rd_partial['t']
        ori_kp = torch.matmul(full_kp,full_rot) + full_t
        partial_real_kp = torch.matmul(ori_kp - partial_t,partial_rot.transpose(1,2))
        
        return partial_real_kp.detach()
    
    def DEBUG_vis(self,pred_rd_full, pred_rd_partial):
        full_kp = pred_rd_full['keypoints']
        full_pc = pred_rd_full['recon_pc']
        full_seg = pred_rd_full['pc_seg']
        
        partial_kp = pred_rd_partial['keypoints']
        partial_pc = pred_rd_partial['full_pc_at_inv']
        partial_gt_kp = self.get_partial_real_kp(pred_rd_full, pred_rd_partial)
        
        #show_pc_kp(partial_pc[3].detach().cpu().numpy(),partial_gt_kp[3].detach().cpu().numpy())
        #show_pc_kp(partial_pc[4].detach().cpu().numpy(),partial_gt_kp[4].detach().cpu().numpy())
        #show_pc_kp(partial_pc[5].detach().cpu().numpy(),partial_gt_kp[5].detach().cpu().numpy())
        
        
        #seg_idx = torch.topk(full_seg,k=1,dim=-1)
        
        #show_pc_seg(full_pc[3].detach().cpu().numpy(),full_seg[3].detach().cpu())
        
        #show_pc_kp(full_pc[3].detach().cpu().numpy(),full_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),full_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[5].detach().cpu().numpy(),full_kp[5].detach().cpu().numpy())
    
    
    def get_all_loss_terms(self, pred_rd_full=None, pred_rd_partial=None, fp_list=None):
        loss_name_list = {}
        loss_sum = 0.0
        
        loss_seg = self.cal_seg_loss(pred_rd_full, pred_rd_partial, fp_list) * 0.5
        loss_name_list['seg'] = loss_seg
        loss_sum += loss_seg
        
        loss_feat = self.cal_feature_loss(pred_rd_full, pred_rd_partial, fp_list) * 0
        loss_name_list['feat'] = loss_feat
        loss_sum += loss_feat
        
        loss_recon = self.cal_recon_loss(pred_rd_full, pred_rd_partial, fp_list) * 2
        loss_name_list['recon'] = loss_recon
        loss_sum += loss_recon
        
        loss_recon_full = self.cal_recon_full_loss(pred_rd_full,pred_rd_partial) * 3
        loss_name_list['recon_full'] = loss_recon_full
        loss_sum += loss_recon_full
        
        loss_pd = self.cal_part_direction_loss(pred_rd_full, pred_rd_partial) *2
        loss_name_list['pd'] = loss_pd
        loss_sum += loss_pd
        
        loss_kp = self.cal_keypoints_loss(pred_rd_full, pred_rd_partial) * 5
        loss_name_list['kp'] = loss_kp
        loss_sum += loss_kp
        
        loss_kp_seg = self.cal_kp_seg_loss(pred_rd_full, pred_rd_partial) * 0.25
        loss_name_list['kp_seg'] = loss_kp_seg
        loss_sum += loss_kp_seg
        
        loss_kp_recon = self.cal_kp_recon_loss(pred_rd_partial) * 5
        loss_name_list['kp_recon'] = loss_kp_recon
        loss_sum += loss_kp_recon
        
        loss_mat = self.cal_mat_loss(pred_rd_full, pred_rd_partial) * 1
        loss_name_list['mat'] = loss_mat
        loss_sum += loss_mat
        
        self.DEBUG_vis(pred_rd_full, pred_rd_partial)
        
        return loss_name_list, loss_sum
    


"test!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
class Consis_loss(nn.Module):
    def __init__(self, opt):
        super(Consis_loss, self).__init__()
        self.opt = opt
        self.criterionPCRecon = loss.chamfer_distance
        self.criterionMSE = nn.MSELoss()
        self.criterionCosine = F.cosine_similarity
        self.criterionROT = RotationLoss(device=opt.base_device, which_metric=opt.partial_which_rot_metric)
        self.criterionOrtho = OrthogonalLoss(device=opt.base_device, which_metric=opt.partial_which_ortho_metric)

    #  the point segmentation consistency loss term for the full and partial branches
    def cal_seg_loss(self, pred_rd_full, pred_rd_partial, fp_list = None):
        pc_seg_full = pred_rd_full['pc_seg']
        pc_seg_partial = pred_rd_partial['pc_seg']
        # pc_seg_full    bs x nofP_full x nofkp
        # fp_list: the corresponding list of each partial point to its full point
        # fp_list: bsx nof_partial
        seg_loss = []
        bs, nofp_f, nofKP = pc_seg_full.shape
        for i in range(bs):
            pc_seg_full_now = pc_seg_full[i]  # nofP_full x nofKP
            
            #####DEBUG fp_list out idx 
            if fp_list != None:
                pc_seg_full_to_partial = pc_seg_full_now[fp_list[i], :]
            else:
                pc_seg_full_to_partial = pc_seg_full_now
            
            pc_seg_partial_now = pc_seg_partial[i]
            seg_loss_now = 1. - self.criterionCosine(pc_seg_partial_now, pc_seg_full_to_partial)
            seg_loss_now = seg_loss_now.mean()
            seg_loss.append(seg_loss_now)
        seg_loss = torch.stack(seg_loss)
        return torch.sum(seg_loss) / (bs + 1) * self.opt.cross_weight_seg

    #  we also supervise the extracted pointwise features. Corresponding point features in the full and partial
    #  branch should be the same.
    def cal_feature_loss(self, pred_rd_full, pred_rd_partial, fp_list):
        pc_f_full = pred_rd_full['inv_f_p']
        pc_f_partial = pred_rd_partial['inv_f_p']
        # pc_f_full : bs x nofP_full x c_dim
        # pc_f_partial: bs x nofP_full x c_dim
        bs, nofP_full, _ = pc_f_full.shape
        feat_loss = []
        for i in range(bs):
            f_full_now = pc_f_full[i]  # nofP_full x c_dim
            f_full_to_partial = f_full_now[fp_list[i], :]

            f_partial_now = pc_f_partial[i]
            feat_loss_now = self.criterionMSE(f_full_to_partial, f_partial_now)
            feat_loss.append(feat_loss_now)
        feat_loss = torch.stack(feat_loss)
        feat_loss = torch.sum(feat_loss) / (bs + 1)
        
        return feat_loss * self.opt.cross_weight_feature
    
    
    def cal_feature_max_loss(self, pred_rd_full, pred_rd_partial, fp_list):
        pc_f_full = pred_rd_full['inv_f']
        pc_f_partial = pred_rd_partial['inv_f']
        feat_loss = self.criterionMSE(pc_f_full,pc_f_partial)
        
        return feat_loss * self.opt.cross_weight_feature

    #  supervise the reconstructed point cloud in the canonical space. The partial reconstruction should be a part of
    #  the full reconstruction
    def cal_recon_loss(self, pred_rd_full, pred_rd_partial, fp_list = None):
        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['recon_pc']
        if fp_list == None:
            loss_recon, _, _ = u_chamfer_distance(pc_partial, pc_full)
        else:
            bs, nofP_full, _ = pc_full.shape
            recon_loss = []
            for i in range(bs):
                pc_full_now = pc_full[i]
                pc_full_to_partial = pc_full_now[fp_list[i],:].unsqueeze(0)
                pc_partial_now = pc_partial[i].unsqueeze(0)
                recon_loss_now = self.criterionPCRecon(pc_partial_now,pc_full_to_partial)[0]
                recon_loss.append(recon_loss_now)
            recon_loss = torch.stack(recon_loss)
            loss_recon = torch.sum(recon_loss) / (bs + 1)
        
        return loss_recon * self.opt.cross_weight_recon
    
    def cal_recon_full_loss(self,pred_rd_full, pred_rd_partial):
        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['full_pc_at_inv']
        
        recon_full_loss = self.criterionPCRecon(pc_partial,pc_full)[0]
        
        return recon_full_loss * self.opt.cross_weight_recon
        
    #  supervise the direction vectors between keypoints. A consistency term between the full and partial branches
    #
    def cal_part_direction_loss(self, pred_rd_full, pred_rd_partial):
        #kp_full = self.get_partial_real_kp(pred_rd_full,pred_rd_partial)
        kp_full = pred_rd_full['keypoints']
        kp_partial = pred_rd_partial['keypoints']
        kp_dir_loss = []
        bs, nofKP, _ = kp_full.shape
        for i in range(nofKP):
            for j in range(i + 1, nofKP):
                kp_full_dir = kp_full[:, i, :] - kp_full[:, j, :]
                kp_partial_dir = kp_partial[:, i, :] - kp_partial[:, j, :]
                kp_dir_error = torch.mean((kp_full_dir - kp_partial_dir) * (kp_full_dir - kp_partial_dir))
                kp_dir_loss.append(kp_dir_error)
        nofnum = len(kp_dir_loss) + 1
        kp_dir_loss = torch.stack(kp_dir_loss)
        kp_dir_loss = torch.sum(kp_dir_loss) / nofnum * self.opt.cross_weight_pd

        return kp_dir_loss

    # extracted keypoints should be consistent between the two branches
    #>>>> keypoints
    def cal_keypoints_loss(self, pred_rd_full, pred_rd_partial):
        kp_full = pred_rd_full['keypoints']
        #kp_full = self.get_partial_real_kp(pred_rd_full,pred_rd_partial)
        kp_partial = pred_rd_partial['keypoints']
        
        #   kp_full: bs x nofKP x 3
        #   kp_partial: bs x nofKP x 3
        bs, nofKP, _ = kp_full.shape
        kp_loss_p = F.mse_loss(kp_full, kp_partial)
        return kp_loss_p * self.opt.cross_weight_kp
    
    def cal_kp_recon_loss(self, pred_rd_full, pred_rd_partial):
        kp_full = pred_rd_partial['keypoints']
        #recon_pc_full = pred_rd_partial['full_pc_at_inv']
        recon_pc_full = pred_rd_full['recon_pc']
        loss_kp_recon_full, _ = self.criterionPCRecon(kp_full, recon_pc_full)
        return loss_kp_recon_full
    
    # keypoints extracted should be better in the center of its support region
    def cal_kp_seg_loss(self, pred_rd_full, pred_rd_partial, min_support_pixels=20):
        kp_full = pred_rd_full['keypoints']
        pc_seg_full = pred_rd_full['pc_seg']  # bs x nofP x nofkp
        recon_pc_full = pred_rd_full['recon_pc']  # bs x nofP x 3
        bs, nofP, nofKP = pc_seg_full.shape
        kp_support_pixel_num_full = torch.sum(pc_seg_full.detach(), dim=1)  # soft arrangement, bs x nofkp
        # bs x nfkp x 3
        kp_seg_full_loss = []
        kp_seg_partial_loss = []
        pc_seg_center_full = (pc_seg_full.permute(0, 2, 1) @ recon_pc_full)  # bs  nofKP x 3

        kp_partial = pred_rd_partial['keypoints']
        pc_seg_partial = pred_rd_partial['pc_seg']
        recon_pc_partial = pred_rd_partial['recon_pc']
        kp_support_pixel_num_partial = torch.sum(pc_seg_partial.detach(), dim=1)
        pc_seg_center_partial = (pc_seg_partial.permute(0, 2, 1) @ recon_pc_partial)

        for i in range(bs):
            for j in range(nofKP):
                pc_seg_center_now = pc_seg_center_full[i, j, :]
                pc_seg_center_support = kp_support_pixel_num_full[i, j]
                if pc_seg_center_support < min_support_pixels:
                    continue
                pc_seg_center_now = pc_seg_center_now / pc_seg_center_support
                kp_full_now = kp_full[i, j, :]
                kp_seg_full_loss_now = torch.sum((kp_full_now - pc_seg_center_now) * (kp_full_now - pc_seg_center_now))
                kp_seg_full_loss.append(kp_seg_full_loss_now)

                pc_seg_center_now = pc_seg_center_partial[i, j, :]
                pc_seg_center_support = kp_support_pixel_num_partial[i, j]
                if pc_seg_center_support < min_support_pixels:
                    continue
                pc_seg_center_now = pc_seg_center_now / pc_seg_center_support
                kp_partial_now = kp_partial[i, j, :]
                kp_seg_partial_loss_now = torch.sum((kp_partial_now - pc_seg_center_now) *
                                                    (kp_partial_now - pc_seg_center_now))
                kp_seg_partial_loss.append(kp_seg_partial_loss_now)

        nofL_partial = len(kp_seg_partial_loss)
        nofL_full = len(kp_seg_full_loss)
        if nofL_partial == 0 or nofL_full == 0:
            return 0.0
        kp_seg_full_loss = torch.stack(kp_seg_full_loss)
        kp_seg_partial_loss = torch.stack(kp_seg_partial_loss)

        return (torch.sum(kp_seg_partial_loss) +
                torch.sum(kp_seg_full_loss)) / (nofL_partial + nofL_full) * self.opt.cross_weight_kp_seg
    
    def cal_mat_loss(self,pred_rd_full,pred_rd_partial):
        full_rot = pred_rd_full['rot']
        full_t = pred_rd_full['t']
        partial_rot = pred_rd_partial['rot']
        partial_t = pred_rd_full['t']
        
        rot_mat = self.criterionROT(full_rot,partial_rot)
        t_mat = self.criterionMSE(full_t,partial_t)
        
        return rot_mat+t_mat
    
    def get_partial_real_kp(self,pred_rd_full, pred_rd_partial):
        
        full_kp = pred_rd_full['keypoints']
        full_rot = pred_rd_full['rot']
        full_t = pred_rd_full['t']
        partial_rot = pred_rd_partial['rot']
        partial_t = pred_rd_partial['t']
        ori_kp = torch.matmul(full_kp,full_rot) + full_t
        partial_real_kp = torch.matmul(ori_kp - partial_t,partial_rot.transpose(1,2))
        
        return partial_real_kp.detach()
    
    def DEBUG_vis(self,pred_rd_full, pred_rd_partial):
        full_kp = pred_rd_full['keypoints']
        full_pc = pred_rd_full['recon_pc']
        full_seg = pred_rd_full['pc_seg']
        
        partial_kp = pred_rd_partial['keypoints']
        partial_pc = pred_rd_partial['full_pc_at_inv']
        partial_gt_kp = self.get_partial_real_kp(pred_rd_full, pred_rd_partial)
        
        #show_pc_kp(partial_pc[3].detach().cpu().numpy(),partial_gt_kp[3].detach().cpu().numpy())
        #show_pc_kp(partial_pc[4].detach().cpu().numpy(),partial_gt_kp[4].detach().cpu().numpy())
        #show_pc_kp(partial_pc[5].detach().cpu().numpy(),partial_gt_kp[5].detach().cpu().numpy())
        
        #show_pc_kp(full_pc[3].detach().cpu().numpy(),partial_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),partial_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[5].detach().cpu().numpy(),partial_kp[5].detach().cpu().numpy())
        
        #seg_idx = torch.topk(full_seg,k=1,dim=-1)
        
        #show_pc_seg(full_pc[3].detach().cpu().numpy(),full_seg[3].detach().cpu())
        
        #show_pc_kp(full_pc[3].detach().cpu().numpy(),full_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[3].detach().cpu().numpy(),partial_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),full_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),partial_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[5].detach().cpu().numpy(),full_kp[5].detach().cpu().numpy())
        #show_pc_kp(full_pc[5].detach().cpu().numpy(),partial_kp[5].detach().cpu().numpy())
        #show_pc_kp(partial_pc[3].detach().cpu().numpy(),partial_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[3].detach().cpu().numpy(),partial_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),full_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),partial_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[5].detach().cpu().numpy(),full_kp[5].detach().cpu().numpy())
    
    
    def get_all_loss_terms(self, pred_rd_full=None, pred_rd_partial=None, fp_list=None):
        loss_name_list = {}
        loss_sum = 0.0
        
        loss_seg = self.cal_seg_loss(pred_rd_full, pred_rd_partial, fp_list) * 2
        loss_name_list['seg'] = loss_seg
        loss_sum += loss_seg
        
        loss_feat = self.cal_feature_loss(pred_rd_full, pred_rd_partial, fp_list) * 0 
        loss_name_list['feat'] = loss_feat
        loss_sum += loss_feat
        
        loss_feat_x = self.cal_feature_max_loss(pred_rd_full, pred_rd_partial, fp_list) * 0
        loss_name_list['feat_x'] = loss_feat_x
        loss_sum += loss_feat_x
        
        loss_recon = self.cal_recon_loss(pred_rd_full, pred_rd_partial, fp_list) * 2
        loss_name_list['recon'] = loss_recon
        loss_sum += loss_recon
        
        loss_recon_full = self.cal_recon_full_loss(pred_rd_full,pred_rd_partial) * 3
        loss_name_list['recon_full'] = loss_recon_full
        loss_sum += loss_recon_full
        
        loss_pd = self.cal_part_direction_loss(pred_rd_full, pred_rd_partial) *2
        loss_name_list['pd'] = loss_pd
        loss_sum += loss_pd
        
        loss_kp = self.cal_keypoints_loss(pred_rd_full, pred_rd_partial) * 5
        loss_name_list['kp'] = loss_kp
        loss_sum += loss_kp
        
        loss_kp_seg = self.cal_kp_seg_loss(pred_rd_full, pred_rd_partial) * 0.25
        loss_name_list['kp_seg'] = loss_kp_seg
        loss_sum += loss_kp_seg
        
        loss_kp_recon = self.cal_kp_recon_loss(pred_rd_full, pred_rd_partial) * 5
        loss_name_list['kp_recon'] = loss_kp_recon
        loss_sum += loss_kp_recon
        
        loss_mat = self.cal_mat_loss(pred_rd_full, pred_rd_partial) * 2
        loss_name_list['mat'] = loss_mat
        loss_sum += loss_mat
        
        self.DEBUG_vis(pred_rd_full, pred_rd_partial)
        
        return loss_name_list, loss_sum
    









if __name__ == '__main__':
    train()
