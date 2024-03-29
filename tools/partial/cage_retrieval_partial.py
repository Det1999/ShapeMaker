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

import torch.nn.functional as F
from pytorch3d import loss
from losses.supp_loss_funcs import RotationLoss, OrthogonalLoss, u_chamfer_distance
from pytorch3d.ops.knn import knn_points
from visualization.vis_pc import show_pc, show_pc_kp


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
    #opt.config = 'configs/Tables.yaml'
    
    opt = combine_flags(opt, config_path=opt.config)
    log_dir = opt.log_dir+'_cage_retrieval_0.75' + '/' + opt.name
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
    
    model_partial.load_state_dict(torch.load('Table_partial_rd_0.75/cab-12kpt/model_partial_199.pth'))
    model_partial.eval()
    for param in model_partial.parameters():
        param.requires_grad = False
    
    rd_dic = torch.load('Table_cage_deform_0.75/cab-12kpt/model_randd_199.pth')
    model_RandD.load_state_dict(rd_dic,False)
    #model_RandD.load_state_dict(torch.load('/home/ubuntu/newdisk/wcw/code/I-RED-main/logsV12_NEW_deform/chair-12kpt/model_randd_199.pth'),False)
    #model_RandD.eval()
    #for param in model_RandD.parameters():
    #    param.requires_grad = False
    
    
    for param in model_RandD.full_keypoint_net.parameters():
        param.requires_grad = False
    for param in model_RandD.full_seg_net.parameters():
        param.requires_grad = False
    for param in model_RandD.cage_deform_net.parameters():
        param.requires_grad = False
        
    for param in model_RandD.partial_keypoint_net.parameters():
        param.requires_grad = False
    for param in model_RandD.partial_seg_net.parameters():
        param.requires_grad = False
    #########################################################################################################################
    
    ### init train
    total_iters = 0
    global_step = 0
    #params_full = model_full.get_optim_params()
    #params_partial = model_partial.get_optim_params()
    params_rd = model_RandD.get_optim_params()
    params = itertools.chain(params_rd)
    #params = itertools.chain(params_full, params_partial, params_rd)
    optimizer = torch.optim.Adam(params, lr=opt.cross_lr)
    lr_milestones = [lr_first_decay, lr_second_decay]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
    
    deform_loss_func = Retrieval_loss(opt)
    
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
            input_list_full_source = model_full.online_data_augment(source_full)
            pc = data['target_partial_shape']
            pc_dual = data['target_partial_shape_dual']
            pc_f_p_list = data['target_partial_mask']
            input_list_partial = model_partial.online_data_augment(pc, pc_dual,full_input_list=input_list_full)
            #input_list_partial = model_partial.online_data_augment(pc, pc_dual)
            
            rand_pose_inf={
                'src_rand_rot':input_list_full_source['rand_tort'],
                'src_rand_t':input_list_full_source['rand_t'],
                'tgt_rand_rot':input_list_partial['rand_tort'],
                'tgt_rand_t':input_list_partial['rand_t']
                
            }
            
            #################################################################
            #                      full model infer                         #
            #################################################################
            with torch.no_grad():
                #_, _, _, _, _, _, _, pred_full_RandD = model_full(input_list_full, mode='cross')
                _, _, _, _, _,_, _, pred_full_RandD_source = model_full(input_list_full_source, mode='cross')
            
            #################################################################
            #                      partial model infer                      #
            #################################################################
            input_list_partial['full_pc_full'] = input_list_full['pc_full']
            with torch.no_grad():
                _, _, _, _, _, _, _, _, pred_p_RandD = model_partial(input_list_partial, mode='cross')
            
            #################################################################
            #                          R&D model source infer               #
            #################################################################
            # source infer
            pred_rd_full_source, pred_rd_partial,cage_out = model_RandD(pred_full_RandD_source, pred_p_RandD)
            loss_list_rd_source, loss_sum_rd_source = deform_loss_func.get_all_loss_terms_source(pred_rd_full_source, pred_rd_partial,cage_out,rand_pose_inf)
            
            loss_sum_rd_source.backward()
            torch.nn.utils.clip_grad_norm_(model_RandD.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()
            
            if total_iters % opt.full_log_intervals == 0:
                writer.add_scalar(tag='sum_loss', scalar_value=loss_sum_rd_source, global_step=global_step)
                global_step += 1
                print('rd: {0:04f}'.format(loss_list_rd_source['rd']))
                

        end_time = time.time()
        # update according to the epoch
        scheduler.step()
        if (epoch + 1) % opt.cross_save_intervals == 0 or (epoch + 1) == opt.cross_total_epochs:
            torch.save(model_RandD.state_dict(), '{0}/model_randd_{1:02d}.pth'.format(log_dir, epoch))
        print('Training Epoch: {0} Finished, using {1:04f}.'.format(epoch, end_time - st_time))



class Retrieval_loss(nn.Module):
    def __init__(self, opt):
        super(Retrieval_loss, self).__init__()
        self.opt = opt
        self.criterionPCRecon = loss.chamfer_distance
        self.criterionMSE = nn.MSELoss()
        self.criterionCosine = F.cosine_similarity
        self.criterionROT = RotationLoss(device=opt.base_device, which_metric=opt.partial_which_rot_metric)
        self.criterionOrtho = OrthogonalLoss(device=opt.base_device, which_metric=opt.partial_which_ortho_metric)

    def cal_R_and_D_loss(self, pred_rd_full, pred_rd_partial,cage_out,rand_pose_inf, min_support_pixels=20, max_bear_error=20.):
        r_tokens_full = pred_rd_full['retrieval_tokens']
        r_tokens_partial = pred_rd_partial['retrieval_tokens']
        bs, nofKP, _ = r_tokens_full.shape

        pc_seg_partial = pred_rd_partial['pc_seg']
        kp_support_pixel_num_partial = torch.sum(pc_seg_partial, dim=1)
        pc_seg_full = pred_rd_full['pc_seg']
        kp_support_pixel_num_full = torch.sum(pc_seg_full, dim=1)

        #pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['recon_pc']
        pc_full_deform = cage_out['deformed'].transpose(1,2)
        
        pc_partial_at_full = self.get_cons_shape(pred_rd_full,pred_rd_partial,rand_pose_inf)
        #pc_partial2full = pred_rd_partial['full_pc_at_inv']
        #show_pc(pc_partial_at_full[3].detach().cpu().numpy())
        #show_pc(pc_full_deform[3].detach().cpu().numpy())
        #show_pc(pc_partial_at_full[4].detach().cpu().numpy())
        #show_pc(pc_full_deform[4].detach().cpu().numpy())
        #show_pc(pc_partial_at_full[5].detach().cpu().numpy())
        #show_pc(pc_full_deform[5].detach().cpu().numpy())
        
        x_nn = knn_points(pc_partial_at_full, pc_full_deform, K=1)
        cham_x = x_nn.dists[..., 0]  # (N, P1)
        cham_x = cham_x.unsqueeze(-1)  # N p1 1
        seg_2_deform_loss = (pc_seg_partial.transpose(2, 1) @ cham_x).squeeze(-1)  # bs x nofKP

        loss_R_and_D = []
        for i in range(bs):
            for j in range(nofKP):
                seg_def_loss_now = seg_2_deform_loss[i, j]
                kp_support_pixel_now = kp_support_pixel_num_partial[i, j]
                if kp_support_pixel_now < min_support_pixels:
                    continue
                kp_support_pixel_full_now = kp_support_pixel_num_full[i, j]
                if kp_support_pixel_full_now < min_support_pixels:
                    continue
                seg_def_loss_now = seg_def_loss_now / kp_support_pixel_now  # average matching error for each kp
                seg_def_loss_relative_now = seg_def_loss_now / self.opt.cross_R_and_D_average_error
                seg_def_loss_relative_now = torch.sigmoid(seg_def_loss_relative_now)

                r_tokens_full_now = r_tokens_full[i, j, :] / kp_support_pixel_full_now
                r_tokens_partial_now = r_tokens_partial[i, j, :] / kp_support_pixel_now
                r_dis = torch.sum(torch.pow((r_tokens_full_now - r_tokens_partial_now), 2.))
                loss_rd_now = torch.pow(r_dis - seg_def_loss_relative_now, 2.)
                if loss_rd_now > max_bear_error:
                    continue
                loss_R_and_D.append(loss_rd_now)

        nofL = len(loss_R_and_D)
        if nofL == 0:
            return 0.0
        loss_R_and_D = torch.stack(loss_R_and_D)
        loss_R_and_D = torch.sum(loss_R_and_D) / (nofL + 1) * self.opt.cross_weight_R_and_D
        return loss_R_and_D
    
    
    def DEBUG_vis(self,pred_rd_full, pred_rd_partial):
        full_kp = pred_rd_full['keypoints']
        full_pc = pred_rd_full['recon_pc']
        full_seg = pred_rd_full['pc_seg']
        
        partial_kp = pred_rd_partial['keypoints']
        partial_pc = pred_rd_partial['full_pc_at_inv']
    
    def get_cons_shape(self,pred_rd_full,pred_rd_partial,rand_pose_inf):
        #partial
        p_rand_rot = rand_pose_inf['tgt_rand_rot']
        p_rand_t = rand_pose_inf['tgt_rand_t']
        p_rot = pred_rd_partial['rot']
        p_t = pred_rd_partial['t']
        #full
        f_rand_rot = rand_pose_inf['src_rand_rot']
        f_rand_t = rand_pose_inf['src_rand_t']
        f_rot = pred_rd_full['rot']
        f_t = pred_rd_full['t']
        
        #partial2full
        p_pc = pred_rd_partial['recon_pc']
        f_pc = pred_rd_full['recon_pc']
        
        z_p_pc = torch.matmul(p_pc,p_rot)+p_t
        z_p_pc = torch.matmul(z_p_pc - p_rand_t,p_rand_rot.transpose(1,2))
        #show_pc(z_p_pc[3].detach().cpu().numpy())
        z_p_pc = torch.matmul(z_p_pc,f_rand_rot)+f_rand_t
        p_pc_in_f = torch.matmul(z_p_pc - f_t,f_rot.transpose(1,2))
        
        return p_pc_in_f
        
        
    
    def get_all_loss_terms_source(self, pred_rd_full=None, pred_rd_partial=None,cage_out = None,rand_pose_inf = None):
        loss_name_list = {}
        loss_sum = 0.0
        
        loss_rd = self.cal_R_and_D_loss(pred_rd_full, pred_rd_partial,cage_out,rand_pose_inf)
        loss_name_list['rd'] = loss_rd
        loss_sum += loss_rd
        
        #self.get_cons_shape(pred_rd_full,pred_rd_partial,rand_pose_inf)
        
        return loss_name_list, loss_sum



if __name__ == '__main__':
    train()
