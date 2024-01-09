import time
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from tensorboardX import SummaryWriter
from dataset.shapenetv1 import Shapes
from models.I_RED_full import I_RED_full
from models.I_RED_partial import I_RED_partial
from models.I_RED_rd_cage import I_RED_rd_cage
import itertools

import argparse
from options.base_options import BaseOptions, combine_flags
from options.full_options import FullOptions
from options.partial_options import PartialOptions
from options.cross_options import CrossOptions
from options.dataset_options import ShapeNetV1_options
import os

import torch.nn.functional as F
from pytorch3d import loss
from losses.supp_loss_funcs import RotationLoss, OrthogonalLoss, u_chamfer_distance
from pytorch3d.ops.knn import knn_points
from visualization.vis_pc import show_pc,show_pc_kp


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
    opt.partial_add_noise = False
    opt.partial_fps = False
    opt.partial_remove_knn = False
    opt.partial_resample = False
    opt.partial_randRT = True
    opt.partial_dual = False
    opt.partial_apply_can_rot = True
    
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
    opt = combine_flags(opt, config_path=opt.config)
    log_dir = opt.log_dir+'_FULL_rd_kp_seg' + '/' + opt.name
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
    model_RandD = I_RED_rd_cage(opt,False).to(opt.base_device)
    s_epoch = 0
    t_epoch = 200
    lr_first_decay = 140
    lr_second_decay = 180
    
    ######################################################################################################################### fix the parameters of the model_full
    model_full.load_state_dict(torch.load('Table/model_199.pth'))
    model_full.eval()
    for param in model_full.parameters():
        param.requires_grad = False
    
    #model_RandD.load_state_dict(torch.load('/home/ubuntu/newdisk/wcw/code/I-RED-main/logsV14_NEW_rd_kp_seg/chair-12kpt/model_randd_199.pth'))
    #model_RandD.eval()
    #for param in model_RandD.parameters():
    #    param.requires_grad = False
    
    #########################################################################################################################
    
    ### init train
    total_iters = 0
    global_step = 0
    
    params_rd = model_RandD.get_optim_params()
    params = itertools.chain(params_rd)
    optimizer = torch.optim.Adam(params, lr=opt.cross_lr)
    lr_milestones = [lr_first_decay, lr_second_decay]#72 92
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    
    KpSeg_loss_func = KP_SEG_loss(opt)
    
    
    
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
            #################################################################
            #                      full model infer                         #
            #################################################################
            with torch.no_grad():
                _, _, _, _, _, _, _, pred_full_RandD = model_full(input_list_full, mode='cross')
            
            #################################################################
            #                          R&D model infer                      #
            #################################################################
            pred_rd_full = model_RandD(pred_full_RandD)
            loss_name_list, loss_sum = KpSeg_loss_func.get_all_loss_terms(pred_rd_full)
            
            loss_sum.backward()
            torch.nn.utils.clip_grad_norm_(model_RandD.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()
            
            if total_iters % opt.full_log_intervals == 0:
                writer.add_scalar(tag='sum_loss', scalar_value=loss_sum, global_step=global_step)
                global_step += 1
                print('kp_seg_loss: {0:04f}, kp_recon_loss: {1:04f}, seg_loss: {2:04f}'.format(loss_name_list['kp_seg'],loss_name_list['kp_recon'],loss_name_list['seg']))
        
        end_time = time.time()
        scheduler.step()
        if (epoch + 1) % opt.cross_save_intervals == 0 or (epoch + 1) == opt.cross_total_epochs:
            torch.save(model_RandD.state_dict(), '{0}/model_randd_{1:02d}.pth'.format(log_dir, epoch))
        print('Training Epoch: {0} Finished, using {1:04f}.'.format(epoch, end_time - st_time))




class KP_SEG_loss(nn.Module):
    def __init__(self, opt):
        super(KP_SEG_loss, self).__init__()
        self.opt = opt
        self.criterionPCRecon = loss.chamfer_distance
        self.criterionMSE = nn.MSELoss()
        self.criterionCosine = F.cosine_similarity
        self.criterionROT = RotationLoss(device=opt.base_device, which_metric=opt.partial_which_rot_metric)
        self.criterionOrtho = OrthogonalLoss(device=opt.base_device, which_metric=opt.partial_which_ortho_metric)

    # keypoints extracted should be better in the center of its support region
    def cal_kp_seg_loss(self, pred_rd_full, min_support_pixels=20):
        kp_full = pred_rd_full['keypoints']
        pc_seg_full = pred_rd_full['pc_seg']  # bs x nofP x nofkp
        recon_pc_full = pred_rd_full['recon_pc']  # bs x nofP x 3
        bs, nofP, nofKP = pc_seg_full.shape
        kp_support_pixel_num_full = torch.sum(pc_seg_full.detach(), dim=1)  # soft arrangement, bs x nofkp
        # bs x nfkp x 3
        kp_seg_full_loss = []
        pc_seg_center_full = (pc_seg_full.permute(0, 2, 1) @ recon_pc_full)  # bs  nofKP x 3


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


        nofL_full = len(kp_seg_full_loss)
        if nofL_full == 0:
            return 0.0
        kp_seg_full_loss = torch.stack(kp_seg_full_loss)

        return (torch.sum(kp_seg_full_loss)) / (nofL_full)

    #   keypoints should be evenly distributed in the point cloud
    def cal_kp_recon_loss(self, pred_rd_full):
        kp_full = pred_rd_full['keypoints']
        recon_pc_full = pred_rd_full['recon_pc']
        loss_kp_recon_full, _ = self.criterionPCRecon(kp_full, recon_pc_full)
        
        return loss_kp_recon_full
    
    def cal_seg_loss(self,pred_rd_full):
        kp_full = pred_rd_full['keypoints']
        pc_seg_full = pred_rd_full['pc_seg']  # bs x nofP x nofkp
        recon_pc_full = pred_rd_full['recon_pc']  # bs x nofP x 3
        
        distance = torch.cdist(recon_pc_full,kp_full)
        dis,dis_idx = torch.topk(distance,k=1,dim = -1,largest= False)
        gt_pc_seg = F.one_hot(dis_idx.squeeze(-1)).detach()
        
        loss_seg = torch.mean((pc_seg_full - gt_pc_seg)*(pc_seg_full - gt_pc_seg))
        
        return loss_seg
        
    
    def get_all_loss_terms(self, pred_rd_full=None):
        loss_name_list = {}
        loss_sum = 0.0

        loss_kp_seg = self.cal_kp_seg_loss(pred_rd_full)
        loss_name_list['kp_seg'] = loss_kp_seg
        loss_sum += loss_kp_seg
            
        loss_kp_recon = self.cal_kp_recon_loss(pred_rd_full) * 4
        loss_name_list['kp_recon'] = loss_kp_recon
        loss_sum += loss_kp_recon
        
        
        loss_seg = self.cal_seg_loss(pred_rd_full)
        #loss_seg = 0
        loss_name_list['seg'] = loss_seg
        loss_sum += loss_seg
        
        return loss_name_list, loss_sum


if __name__ == '__main__':
    train()
