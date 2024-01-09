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
    opt.config = 'configs/Tables.yaml'
    
    opt = combine_flags(opt, config_path=opt.config)
    log_dir = opt.log_dir+'_cage_deform_0.75' + '/' + opt.name
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
    
    model_RandD.load_state_dict(torch.load('Table_partial_rd_0.75/cab-12kpt/model_randd_199.pth'),False)
    #model_RandD.load_state_dict(torch.load('/home/ubuntu/newdisk/wcw/code/I-RED-main/logsV12_NEW_deform/chair-12kpt/model_randd_199.pth'),False)
    #model_RandD.eval()
    #for param in model_RandD.parameters():
    #    param.requires_grad = False
    
    
    for param in model_RandD.full_keypoint_net.parameters():
        param.requires_grad = False
    for param in model_RandD.full_seg_net.parameters():
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
    
    deform_loss_func = Deform_loss(opt)
    
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
            #################################################################
            #                      full model infer                         #
            #################################################################
            with torch.no_grad():
                _, _, _, _, _, _, _, pred_full_RandD = model_full(input_list_full, mode='cross')
                pred_rd_full,_ = model_RandD(pred_full_RandD)
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
            loss_list_rd_source, loss_sum_rd_source = deform_loss_func.get_all_loss_terms_source(pred_rd_full_source, pred_rd_partial, pred_rd_full,cage_out)
            
            loss_sum_rd_source.backward()
            torch.nn.utils.clip_grad_norm_(model_RandD.parameters(), 5.)
            optimizer.step()
            optimizer.zero_grad()
            
            if total_iters % opt.full_log_intervals == 0:
                writer.add_scalar(tag='sum_loss', scalar_value=loss_sum_rd_source, global_step=global_step)
                global_step += 1
                print('deform: {0:04f}, influence: {1:04f}'.format(loss_list_rd_source['deform'],loss_list_rd_source['influence']))
                

        end_time = time.time()
        # update according to the epoch
        scheduler.step()
        if (epoch + 1) % opt.cross_save_intervals == 0 or (epoch + 1) == opt.cross_total_epochs:
            torch.save(model_RandD.state_dict(), '{0}/model_randd_{1:02d}.pth'.format(log_dir, epoch))
        print('Training Epoch: {0} Finished, using {1:04f}.'.format(epoch, end_time - st_time))



class Deform_loss(nn.Module):
    def __init__(self, opt):
        super(Deform_loss, self).__init__()
        self.opt = opt
        self.criterionPCRecon = loss.chamfer_distance
        self.criterionMSE = nn.MSELoss()
        self.criterionCosine = F.cosine_similarity
        self.criterionROT = RotationLoss(device=opt.base_device, which_metric=opt.partial_which_rot_metric)
        self.criterionOrtho = OrthogonalLoss(device=opt.base_device, which_metric=opt.partial_which_ortho_metric)


    def cal_deform_loss(self, pred_rd_full, pred_rd_partial,pred_rd_full_t,cage_out):
        # deform field
        full_pc_at_inv = pred_rd_partial['full_pc_at_inv']
        full_pc = pred_rd_full_t['recon_pc']
        pc_full_deform = cage_out['deformed'].transpose(1,2)
        deform_loss = self.criterionPCRecon(full_pc, pc_full_deform)[0]
        
        full_kp = pred_rd_partial['keypoints']
        
        #show_pc(full_pc[3].detach().cpu().numpy())
        #show_pc(full_pc[4].detach().cpu().numpy())
        #show_pc(full_pc[5].detach().cpu().numpy())
        
        #show_pc_kp(full_pc[3].detach().cpu().numpy(),full_kp[3].detach().cpu().numpy())
        #show_pc_kp(full_pc[4].detach().cpu().numpy(),full_kp[4].detach().cpu().numpy())
        #show_pc_kp(full_pc[5].detach().cpu().numpy(),full_kp[5].detach().cpu().numpy())
        
        #show_pc_kp(source_pc[3].detach().cpu().numpy(),source_kp[3].detach().cpu().numpy())
        #show_pc_kp(target_pc[3].detach().cpu().numpy(),target_kp[3].detach().cpu().numpy())
        #show_pc_kp(source_pc[4].detach().cpu().numpy(),source_kp[4].detach().cpu().numpy())
        #show_pc_kp(target_pc[4].detach().cpu().numpy(),target_kp[4].detach().cpu().numpy())
        
        
        
        return deform_loss 
    
    def cal_influence_loss(self,pred_rd_full, pred_rd_partial,cage_out):
        influence = cage_out['influence']
        influence_loss = torch.mean(influence ** 2)
        return influence_loss
    
    
    def DEBUG_vis(self,pred_rd_full, pred_rd_partial):
        full_kp = pred_rd_full['keypoints']
        full_pc = pred_rd_full['recon_pc']
        full_seg = pred_rd_full['pc_seg']
        
        partial_kp = pred_rd_partial['keypoints']
        partial_pc = pred_rd_partial['full_pc_at_inv']
        
        #show_pc(full_pc[3].detach().cpu().numpy())
        #show_pc(partial_pc[3].detach().cpu().numpy())
        #show_pc(full_pc[4].detach().cpu().numpy())
        #show_pc(partial_pc[4].detach().cpu().numpy())
        #show_pc(full_pc[5].detach().cpu().numpy())
        #show_pc(partial_pc[5].detach().cpu().numpy())
    
    
    def get_all_loss_terms_source(self, pred_rd_full=None, pred_rd_partial=None,pred_rd_full_t=None,cage_out = None):
        loss_name_list = {}
        loss_sum = 0.0
        
        loss_deform = self.cal_deform_loss(pred_rd_full, pred_rd_partial,pred_rd_full_t,cage_out)
        #loss_deform = 0.0
        loss_name_list['deform'] = loss_deform
        loss_sum += loss_deform 
        
        loss_influence = self.cal_influence_loss(pred_rd_full, pred_rd_partial,cage_out)
        loss_name_list['influence'] = loss_influence
        loss_sum += loss_influence
        
        
        #self.DEBUG_vis(pred_rd_full, pred_rd_partial)
        
        return loss_name_list, loss_sum



if __name__ == '__main__':
    train()