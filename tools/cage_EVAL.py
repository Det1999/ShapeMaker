
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import json
import os
import time
from datetime import datetime
from visualization.vis_pc import show_pc

import numpy as np
import pytorch3d.io
#import torch
import torch.nn.parallel
import torch.utils.data
import sys
sys.path.append('./')
from tqdm import tqdm
import pytorch3d
from options.base_options import BaseOptions ,combine_flags
from options.full_options import FullOptions
from options.partial_options import PartialOptions
from options.cross_options import CrossOptions
from options.dataset_options import ShapeNetV1_options
from options.test_options import TestOptions
import argparse

from utils import logger

from dataset.shapenetv1 import Shapes
from models.I_RED_full import I_RED_full
from models.I_RED_partial import I_RED_partial
from models.I_RED_rd import I_RED_rd
from models.I_RED_rd_cage import I_RED_rd_cage
from models.I_RED_rd_cageV2 import I_RED_rd_cageV2

from losses.supp_loss_funcs import u_chamfer_distance
from pytorch3d import loss
from models.cage.cages import deform_with_MVC
from typing import Optional
import utils.pc_utils.pc_utils as pc_utils

import time

EPS = 1e-10

def merge_data(datas, data):
    if datas is None:
        datas = {}
        for k, v in data.items():
            if v is not None:
                datas[k] = v
    else:
        for k, v in data.items():
            if isinstance(datas[k], list):
                datas[k].extend(v)
            else:
                datas[k] = torch.cat([datas[k], v], dim = 0)
    return datas

def save_tensor(output_save_dir,tensor,name):
    torch.save(tensor.cpu(),os.path.join(output_save_dir,f'{name}.pt'))

import warnings
def _save_mesh(f, verts, faces, decimal_places: Optional[int] = None) -> None:
    """
    Faster version of https://pytorch3d.readthedocs.io/en/stable/_modules/pytorch3d/io/obj_io.html

    Adding .detach().numpy() to the input tensors makes it 10x faster
    """
    assert not len(verts) or (verts.dim() == 2 and verts.size(1) == 3)
    assert not len(faces) or (faces.dim() == 2 and faces.size(1) == 3)

    if not (len(verts) or len(faces)):
        warnings.warn("Empty 'verts' and 'faces' arguments provided")
        return

    if torch.any(faces >= verts.shape[0]) or torch.any(faces < 0):
        warnings.warn("Faces have invalid indices")

    verts, faces = verts.cpu().detach().numpy(), faces.cpu().detach().numpy()

    lines = ""

    if len(verts):
        if decimal_places is None:
            float_str = "%f"
        else:
            float_str = "%" + ".%df" % decimal_places

        V, D = verts.shape
        for i in range(V):
            vert = [float_str % verts[i, j] for j in range(D)]
            lines += "v %s\n" % " ".join(vert)

    if len(faces):
        F, P = faces.shape
        for i in range(F):
            face = ["%d" % (faces[i, j] + 1) for j in range(P)]
            if i + 1 < F:
                lines += "f %s\n" % " ".join(face)
            elif i + 1 == F:
                # No newline at the end of the file.
                lines += "f %s" % " ".join(face)

    f.write(lines)
    
    
def io_save_mesh(f,verts,faces,decimal_places = None):
    with open(f,'w') as f:
        _save_mesh(f,verts,faces,decimal_places)

def save_mesh(test_output_dir,target_name,source_face,source_mesh,source_mesh_deformed):
    save_dir = os.path.join(test_output_dir,'mesh')
    os.makedirs(save_dir,exist_ok=True)
    
    txtfile = open(os.path.join(save_dir,'A_name.txt'),'w')
    for name in target_name:
        txtfile.write(name[0]+'\n')
    txtfile.close()
    
    for i in range(len(source_face)):
        io_save_mesh(os.path.join(save_dir,'mesh_'+str(i)+'.obj'),source_mesh[i],source_face[i])
        io_save_mesh(os.path.join(save_dir,'mesh_deformed_'+str(i)+'.obj'),source_mesh_deformed[i],source_face[i])
    

def get_data(dataset, data, use_partial=False):
    data = dataset.uncollate(data)
    if "source_shape" in data and "target_shape" in data:
        print("SourceShape AND TargetShape not be realse")
        
    elif "source_shape" in data:
        source_shape = data["source_shape"]
        source_shape_t = source_shape.transpose(1,2)
        target_shape = None
        target_shape_t = None
            
    elif "target_shape" in data:
        target_shape = data["target_shape"]
        target_shape_t = target_shape.transpose(1,2)
        source_shape = None
        source_shape_t = None
            
    else:
        raise Exception("NO SHAPE IN DATA!")
    
    if use_partial:
        if "target_partial_shape" in data:
            target_parital_shape = data["target_partial_shape"]
            full_target_shape_t = target_shape_t.clone()
            target_shape_t = target_parital_shape.transpose(1,2)
        else:
            full_target_shape_t = None
        
        return source_shape_t,target_shape_t, full_target_shape_t
    else:
        full_target_shape_t = None
        return source_shape_t,target_shape_t,full_target_shape_t



def get_retrieval_full(model_pre,model_rd,shape_data):
    """
    model_pre : model_full or model_partial
    model_rd : mdoel_rd
    shape_data : pc data
    """
    EPS = 1e-10
    inv_z, inv_z_p, _, _ = model_pre.netEncoder(shape_data, mode = 'cross')
    inv_z = inv_z/(torch.norm(inv_z, dim = -1, keepdim = True) + EPS)
    full_pc_seg,_ = model_rd.full_seg_net(inv_z,inv_z_p)
    full_retrieval_tokens,f_feature = model_rd.full_retrieval_net(inv_z_p,full_pc_seg)
    
    
    return full_retrieval_tokens,full_pc_seg,f_feature.mean(1)

def get_retrieval_partial(model_pre,model_rd,shape_data):
    """
    model_pre : model_full or model_partial
    model_rd : mdoel_rd
    shape_data : pc data
    """
    EPS = 1e-10
    inv_z, inv_z_p, _, _ = model_pre.netEncoder(shape_data, mode = 'cross')
    inv_z = inv_z/(torch.norm(inv_z, dim = -1, keepdim = True) + EPS)
    partial_pc_seg,_ = model_rd.partial_seg_net(inv_z,inv_z_p)
    partial_retrieval_tokens,p_feature = model_rd.partial_retrieval_net(inv_z_p,partial_pc_seg)
    
    return partial_retrieval_tokens,partial_pc_seg,p_feature.mean(1)

def get_retrival_dist(full_tokens,full_seg,partial_tokens,partial_seg, min_support_pixel = 20):
    """_summary_

    Args:
        full_tokens (_type_): shape[492,12,1020]
        full_seg (_type_): shape[492,2500,12]
        partial_tokens (_type_): shape[12,1020]
        partial_seg (_type_): shape[2500,12]
        min_support_pixel (int, optional): _description_. Defaults to 20.

    Returns:
        dist: shape[492]
    """
    bs,nofKP,_ = full_tokens.shape
    
    kp_support_pixel_num_full = torch.sum(full_seg, dim = 1)
    kp_support_pixel_num_partial = torch.sum(partial_seg, dim = 0)
    
    
    batch_all_dis = []
    for i in range(bs):
        r_all_dis = []
        for j in range(nofKP):
            kp_support_pixel_now = kp_support_pixel_num_partial[j]
            if kp_support_pixel_now < min_support_pixel:
                continue
             
            kp_support_pixel_full_now = kp_support_pixel_num_full[i,j]
            if kp_support_pixel_full_now < min_support_pixel:
                continue
            
            r_tokens_full_now = full_tokens[i,j,:]/kp_support_pixel_full_now
            r_tokens_partial_now = partial_tokens[j,:]/kp_support_pixel_now
            
            r_dis = torch.sum(torch.pow((r_tokens_full_now - r_tokens_partial_now), 2.))
            r_all_dis.append(r_dis)
        r_all_dis = torch.stack(r_all_dis)
        batch_all_dis.append(r_all_dis.sum())
    batch_all_dis = torch.stack(batch_all_dis)
    
    return batch_all_dis


def get_deform_and_loss(opt_test,pred_rd_full_source,pred_rd_partial, cage_out):
    pc_partial = pred_rd_partial['recon_pc']
    pc_full = pred_rd_full_source['recon_pc']
    pc_full_deform = cage_out['deformed'].transpose(1,2)
    kp_partial = pred_rd_partial['keypoints']
    kp_full = pred_rd_full_source['keypoints']
    
    # if opt_test.use_partial:
    #     deform_loss,DEBUG1,DEBUG2 = u_chamfer_distance(pc_partial,pc_full_deform,batch_reduction=None)
    #     only_r_loss,_,_ = u_chamfer_distance(pc_partial,pc_full,batch_reduction=None)
    #     deform_loss_d = loss.chamfer_distance(pc_partial,pc_full_deform,batch_reduction=None)
    # else:
    #     deform_loss = loss.chamfer_distance(pc_partial,pc_full_deform,batch_reduction=None)
    #     deform_loss = deform_loss[0]
        
    #     only_r_loss = loss.chamfer_distance(pc_partial,pc_full,batch_reduction=None)
    #     only_r_loss = only_r_loss[0]
    deform_loss = loss.chamfer_distance(pc_partial,pc_full_deform,batch_reduction=None)
    deform_loss = deform_loss[0]
      
    only_r_loss = loss.chamfer_distance(pc_partial,pc_full,batch_reduction=None)
    only_r_loss = only_r_loss[0]
        
    return pc_full_deform,kp_partial,kp_full,deform_loss,only_r_loss


def occu_select(opt,rate=0.90):
    N_points = 2500
    Now_points = int(N_points * rate)
    opt.base_decoder_npoint = Now_points
    opt.base_resample_npoints = Now_points
    opt.partial_resample_npoints = Now_points
    
    return opt


def test(opt, opt_dataset, opt_test):
    log_dir = os.path.join(opt.log_dir,opt.name)
    
    # load source_datasets
    logger.info('Loading source datasets......')
    #opt.config = 'configs/Cab.yaml'
    opt_dataset.config = opt.config
    opt_dataset.category = '02871439'
    
    opt_test.config = opt.config
    RATE = 1
    occu_select(opt,RATE)
    
    opt_dataset.load_mesh = True
    
    dataset = Shapes(opt_dataset)
    dataset.load_source = True
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt_test.test_batch,
                                                   shuffle=False,
                                                   drop_last=False,
                                                   collate_fn=dataset.collate,
                                                   num_workers=4,
                                                   worker_init_fn=(lambda id:
                                                                   np.random.seed(np.random.get_state()[1][0] + id)))
    logger.info('Loaded source size : {}'.format(len(dataset)))
    
    # load model (I_RED_full)
    model_full = I_RED_full(opt).to(opt.base_device)
    if opt_test.full_check_name:
        full_ckp_path = os.path.join(opt_test.load_dir,opt_test.full_check_name)
        #model_full.load_state_dict(torch.load(full_ckp_path))
        model_full.load_state_dict(torch.load('/home/ubuntu/newdisk/wcw/code/I-RED-main/chair_log_full/TEXT_chair_full/model_199.pth'))
    logger.info('full model\'s checkpoint is being loaded')
    model_full.eval()
    
    # load model (I_RED_partial)
    model_partial = I_RED_partial(opt).to(opt.base_device)
    if opt_test.partial_check_name:
        partial_ckp_path = os.path.join(opt_test.load_dir,opt_test.partial_check_name)
        #model_partial.load_state_dict(torch.load(partial_ckp_path))
        #model_partial.load_state_dict(torch.load('/home/ubuntu/newdisk/wcw/code/I-RED-main/table_log_0.5/Table_NEW_partial_rd_table_0.5/table-12kpt/model_partial_199.pth'))
    logger.info('partial model\'s checkpoint is being loaded')
    model_partial.eval()
    
    
    # load model (I_RED_rd)
    if opt_test.use_partial == False:
        model_rd = I_RED_rd_cage(opt,False).to(opt.base_device)
    else:
        model_rd = I_RED_rd_cage(opt).to(opt.base_device)
        
    if opt_test.rd_check_name:
        rd_ckp_path = os.path.join(opt_test.load_dir,opt_test.rd_check_name)
        #model_rd.load_state_dict(torch.load(rd_ckp_path))
        model_rd.load_state_dict(torch.load('/home/ubuntu/newdisk/wcw/code/I-RED-main/chair_log_full/TEXT_NEWFULL_cage_retrieval/chair-12kpt/model_randd_199.pth'),False)
    logger.info('rd model\'s checkpoint is being loaded')
    model_rd.eval()
    
    
    # create test dir
    test_output_dir = os.path.join(opt_test.load_dir,opt_test.save_dir)
    os.makedirs(test_output_dir,exist_ok=True)
    
    with torch.no_grad():
        source_tokens_all = torch.zeros([0,opt_dataset.n_keypoints,1020]).cuda()
        source_pc_seg_all = torch.zeros([0,opt_dataset.num_point,opt_dataset.n_keypoints]).cuda()
        source_shapes_all = torch.zeros([0,3,opt_dataset.num_point]).cuda()
        source_feature_all = torch.zeros([0,1020]).cuda()
        source_face_all = []
        source_mesh_all = []
        #source_name_all = []
        source_datas = None
        
        logger.info("generating latent codes for source shapes")
        
        for data in tqdm(dataloader):
        #DEBUG
        #for data in dataloader:
            
            # get source data
            data = dataset.uncollate(data)
            source_face_all.append(data['source_face'][0])
            source_mesh_all.append(data['source_mesh'][0])
            #source_name_all.append(data['source_name'])
            source_shape_t, _, _ = get_data(dataset,data)
            
            #DEBUG
            source_retrieval_tokens,full_pc_seg,f_feature = get_retrieval_full(model_full,model_rd,source_shape_t)
            
            
            # update tokens_all, shapes_all, and source_datas
            source_tokens_all = torch.cat([source_tokens_all,source_retrieval_tokens], dim = 0)
            source_pc_seg_all = torch.cat([source_pc_seg_all,full_pc_seg], dim = 0)
            source_shapes_all = torch.cat([source_shapes_all,source_shape_t] , dim = 0)
            source_feature_all = torch.cat([source_feature_all,f_feature],dim = 0)
            source_datas = merge_data(source_datas, data)
            
            #DEBUG
            #print(full_retrieval_tokens.shape)
            
        save_tensor(test_output_dir, source_shapes_all, 'source_shapes')
        save_tensor(test_output_dir, source_tokens_all, 'source_tokens')
        save_tensor(test_output_dir, source_pc_seg_all, 'source_pc_seg')
        torch.save(source_datas['source_file'], os.path.join(test_output_dir, 'source_files.pkl'))
        
        
        
        dataset.load_source = False
        
        #dataset.opt.load_mesh = True
    
        
        cd_loss_all = torch.zeros([0]).cuda()
        only_r_loss_all = torch.zeros([0]).cuda()
        
        source_after_deforms = torch.zeros([0,3,opt_dataset.num_point]).cuda()
        source_selected_top1 = torch.zeros([0,3,opt_dataset.num_point]).cuda()
        full_target_shape_loaded = torch.zeros([0,3,opt_dataset.num_point]).cuda()
        full_target_seg = torch.zeros([0,opt_dataset.num_point,12]).cuda()
        target_name_all = []
        source_mesh_deforms = []
        source_selected_mesh = []
        source_mesh_face = []
        
        
        if opt_test.use_partial == False:
            target_shape_loaded = torch.zeros([0,3,opt_dataset.num_point]).cuda()
        else:
            target_shape_loaded = torch.zeros([0,3,int(opt_dataset.num_point*RATE)]).cuda()
        
        logger.info('retrieval and deformation')
        
        start_time = time.time()
        
        for data in tqdm(dataloader):
            data = dataset.uncollate(data)
            target_name_all.append(data['target_name'])
            _, target_shape_t, full_target_shape_t  = get_data(dataset,data,opt_test.use_partial)
            
            if opt_dataset.partial_pc and opt_dataset.retrieval_full_shape:
                partial_target_shape_t = target_shape_t
                target_shape_t = full_target_shape_t
        
            target_shape_ori = target_shape_t
            full_target_shape_ori = full_target_shape_t
            
            rot_info = {}
            
            if not opt.partial_randRT:
                target_shape_t,trot,t = pc_utils.rotate(target_shape_t, 'se3', opt.base_device,t=opt.base_se3_T, return_trot=True)
                rot_base = trot.detach()
                t_base = t.detach()
                rot_info['rot_base'] = rot_base
                rot_info['t_base'] = t_base
            
            #DEBUG
            if opt_test.use_partial == False:
                partial_retrieval_tokens,partial_pc_seg,t_feature = get_retrieval_full(model_full,model_rd,target_shape_t)
            else:
                partial_retrieval_tokens,partial_pc_seg,t_feature = get_retrieval_partial(model_partial,model_rd,target_shape_t)
            
            full_target_seg = torch.cat([full_target_seg, partial_pc_seg], dim = 0)
            
            # select top k source data
            sel_indices = torch.zeros([0,opt_test.top_k], dtype=torch.long).cuda()
            i = -1
            for target_tokens,target_pc_seg,token_feature in zip(partial_retrieval_tokens,partial_pc_seg,t_feature):
                i += 1
                
                dist = get_retrival_dist(source_tokens_all,source_pc_seg_all,target_tokens,target_pc_seg)
                
                #######################################################################################################
                #dist = ((token_feature - source_feature_all)*(token_feature - source_feature_all)).sum(1)
                #######################################################################################################
                
                knn = dist.topk(opt_test.top_k, largest=False)
                index = knn.indices
                sel_indices = torch.cat([sel_indices,index.unsqueeze(0)], dim = 0)
            
            if opt_dataset.partial_pc and opt_dataset.retrieval_full_shape:
                target_shape_t = partial_target_shape_t
            
            min_error = None
            only_r_error = None
            final_indices = None
            best_outputs = None
            best_mesh_deformed = None
            best_mesh_face = None
            best_mesh = None
            
            # calculate error
            for k in range(opt_test.top_k):
                source_indices = sel_indices[:,k]
                source_shape_t = source_shapes_all[source_indices]
                source_mesh_t = source_mesh_all[source_indices]
                source_face_t = source_face_all[source_indices]
                
                #DEBUG
                # try to get the source_def shape
                # having source_shape_t and target_source_t in this time.
                source_input = {}
                target_input = {}
                source_input['pc_full'] = source_shape_t
                source_input['rand_t'] = 0
                target_input['pc_full'] = target_shape_t
                target_input['rand_t'] = 0
                
                
                #target_shape_t is being rotated now.
                #source_shape_t is not be rotated.
                
                pred_recon, _, _, _, _,_, _, pred_full_RandD_source = model_full(source_input, mode='cross')
                
                rot_pre = pred_recon['rot']
                t_pre = pred_recon['t_vec']
                
                mesh_at_inv = torch.matmul(source_mesh_t - t_pre, rot_pre.transpose(1, 2))
                
                
                
                #DEBUG
                if opt_test.use_partial == True:
                    pred_recon, _, _, _, _,_, _, _, pred_partial_RandD_target = model_partial(target_input, mode='cross')
                else:
                    pred_recon, _, _, _, _,_, _, pred_partial_RandD_target = model_full(target_input, mode='cross')
                
                rot_info['rot_pre'] = pred_recon['rot']
                rot_info['t_pre'] = pred_recon['t_vec']
                
                
                pred_rd_full_source, pred_rd_partial, cage_out = model_rd(pred_full_RandD_source, pred_partial_RandD_target)
                
                deform_mesh, weights, _ = deform_with_MVC(
                       cage_out['cage'],cage_out['new_cage'], cage_out['cage_face'],mesh_at_inv, verbose=True
                )
                
                source_after_deform, kp_partial, kp_full, deform_loss, only_r_loss = get_deform_and_loss(opt_test,pred_rd_full_source,pred_rd_partial, cage_out)
                source_after_deform_t = source_after_deform.transpose(1,2)
                source_after_deform_ori = (torch.matmul(source_after_deform,rot_pre)+t_pre).permute(0, 2, 1)
                source_after_deform_mesh_ori = (torch.matmul(deform_mesh,rot_pre)+t_pre)
                
                #DEBUG
                scale = data['target_scale'].float().squeeze(1).squeeze(1)
                cur_error = deform_loss*scale*scale
                cur_only_r_error = only_r_loss*scale*scale
                #cur_error = deform_loss
                
                if min_error is None:
                    min_error = cur_error
                    only_r_error = cur_only_r_error
                    final_indices = source_indices
                    best_outputs = source_after_deform_ori
                    best_mesh_deformed = source_after_deform_mesh_ori
                    best_mesh_face = source_face_t
                    best_mesh = source_mesh_t
                else:
                    mask = cur_error < min_error
                    final_indices[mask] = source_indices[mask]
                    min_error[mask] = cur_error[mask]
                    only_r_error[mask] = cur_only_r_error[mask]
                    best_outputs[mask] = source_after_deform_ori[mask]
                    if mask:
                        best_mesh_deformed = source_after_deform_mesh_ori
                        best_mesh_face = source_face_t
                        best_mesh = source_mesh_t
                    
            
            cd_loss_all = torch.cat([cd_loss_all,min_error])
            only_r_loss_all = torch.cat([only_r_loss_all,only_r_error])
            
            for key,value in source_datas.items():
                if isinstance(value, list):
                    data[key] = [value[i] for i in final_indices.cpu().tolist()]
                else:
                    data[key] = value[final_indices]
            
            
            source_mesh_deforms.append(best_mesh_deformed.squeeze(0))
            source_selected_mesh.append(best_mesh)
            source_mesh_face.append(best_mesh_face)
            
            source_after_deforms = torch.cat([source_after_deforms,best_outputs], dim = 0)
            source_selected_top1 = torch.cat([source_selected_top1,data['source_shape'].transpose(1,2)], dim = 0)
            target_shape_loaded = torch.cat([target_shape_loaded, target_shape_ori], dim = 0)
            if full_target_shape_ori != None:
                full_target_shape_loaded = torch.cat([full_target_shape_loaded, full_target_shape_ori], dim = 0)
            #DEBUG
            #save_mesh(test_output_dir,target_name_all,source_mesh_face,source_selected_mesh,source_mesh_deforms)
        
        
        end_time = time.time()
        print("Time:",end_time - start_time)
        save_mesh(test_output_dir,target_name_all,source_mesh_face,source_selected_mesh,source_mesh_deforms)
        save_tensor(test_output_dir, full_target_seg, 'full_target_seg')
        save_tensor(test_output_dir, source_after_deforms, 'source_after_deforms')
        save_tensor(test_output_dir, source_selected_top1, 'source_selected_top1')
        save_tensor(test_output_dir, target_shape_loaded, 'target_shape_loaded')
        save_tensor(test_output_dir, full_target_shape_loaded, 'full_target_shape_loaded')
        save_tensor(test_output_dir, cd_loss_all, 'cd_loss_all')
        save_tensor(test_output_dir, only_r_loss_all, 'only_r_loss_all')
        
        
        logger.info(f'chamfer distance loss without deform mean={only_r_loss_all.mean()}, std={only_r_loss_all.std()}')
        logger.info(f'chamfer distance loss mean={cd_loss_all.mean()}, std={cd_loss_all.std()}')
        
        return source_after_deforms,source_selected_top1,target_shape_loaded
                    


def init_test_config(opt):
    #init dataset config
    opt.split = 'test'
    
    # init full_model config
    opt.full_add_noise = False
    opt.full_fps = False
    opt.full_remove_knn = 0
    opt.full_resample = False
    opt.full_randRT = False
    opt.full_apply_can_rot = False
    
    #init partial_model config
    opt.partial_add_noise = False
    opt.partial_fps = False
    opt.partial_remove_knn = 0
    opt.partial_resample = False
    opt.partial_randRT = False
    opt.partial_dual = False
    opt.partial_apply_can_rot = False

    return opt


if __name__=="__main__":
    arg_parser = argparse.ArgumentParser()
    base_options = BaseOptions()
    full_options = FullOptions()
    partial_options = PartialOptions()
    cross_options = CrossOptions()
    dataset_options = ShapeNetV1_options()
    test_options = TestOptions()
    arg_parser = base_options.initialize(arg_parser)
    arg_parser = full_options.initialize(arg_parser)
    arg_parser = partial_options.initialize(arg_parser)
    arg_parser = cross_options.initialize(arg_parser)
    
    
    opt = arg_parser.parse_args()
    opt = combine_flags(opt, config_path=opt.config)
    
    opt = init_test_config(opt)
    
    opt_dataset = dataset_options.combine_configs(file_path=opt.config)
    opt_test = test_options.combine_configs(file_path=opt.test_config)
    
    opt_dataset.split = 'test'
    seed = opt_dataset.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    source_after_deforms,source_selected_top1,target_shape_loaded = test(opt,opt_dataset,opt_test)
    
