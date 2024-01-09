import torch
import torch.nn as nn
import itertools
import utils.pc_utils.pc_utils as pc_utils
import models.atlas.utils as atlas_utils
from models.encoder_decoder.networks import SimpleRot, ComplexRot
from models.encoder_decoder.networks import VNTSimpleEncoder
from models.encoder_decoder.networks import PointTransMLPAdjDecoder, PatchDeformMLPAdjDecoder

EPS = 1e-10


class I_RED_full(nn.Module):
    def __init__(self, opt):
        super(I_RED_full, self).__init__()
        self.opt = opt

        self.netEncoder = VNTSimpleEncoder(opt, global_feat=True, feature_transform=True)
        if self.opt.base_decoder_name == 'point':
            self.netDecoder = PointTransMLPAdjDecoder(opt, mode='full')
        else:
            self.netDecoder = PatchDeformMLPAdjDecoder(opt)
        self.netEncoder = self.netEncoder.to(opt.base_device)
        self.netDecoder.apply(atlas_utils.weights_init)
        self.netDecoder = self.netDecoder.to(opt.base_device)
        if self.opt.base_rot_net_name == 'simple':
            self.netRotation = SimpleRot(opt.base_rot_nlatent // 2 // 3, opt.base_rot_which_strict_rot)
        else:
            self.netRotation = ComplexRot((opt.base_nlatent // 3), opt.base_rot_which_strict_rot)

    def get_optim_params(self):
        params = itertools.chain(self.netEncoder.parameters(), self.netDecoder.parameters(),
                                 self.netRotation.parameters())

        return params

    def forward(self, input_list, mode='normal'):
        pc_full = input_list['pc_full']
        
        inv_z, inv_z_p, eq_z, t_vec = self.netEncoder(pc_full, mode=mode)
        
        decoded_pc_inv, patches = self.netDecoder(inv_z)
        
        rot_mat = self.netRotation(eq_z)
        
        recon_pc = (torch.matmul(decoded_pc_inv, rot_mat) + t_vec).permute(0, 2, 1)
        pc_at_inv = torch.matmul(pc_full.permute(0, 2, 1) - t_vec, rot_mat.transpose(1, 2)).permute(0, 2, 1)
        recon_pc_inv = decoded_pc_inv.permute(0, 2, 1)

        pred_recon = {
            'pc': pc_full,
            'recon_pc': recon_pc,
            'pc_at_inv': pc_at_inv,
            'recon_pc_inv': recon_pc_inv,
            'rot': rot_mat,
            'inv_z': inv_z,
            'inv_z_p': inv_z_p,
            'eq_z': eq_z,
            't_vec': t_vec,
            't_rand': input_list['rand_t']
        }

        pred_add_noise = {}
        pred_fps = {}
        pred_remove_knn = {}
        pred_resample = {}
        pred_randRT = {}
        pred_can_rot = {}

        if self.opt.full_add_noise:
            noised_pc = input_list['pc_noised']
            noise_inv_z, _, noise_eq_z, t_noised_vec = self.netEncoder(noised_pc)
            noised_rot_mat = self.netRotation(noise_eq_z)
            pred_add_noise['rot_aug'] = noised_rot_mat
            pred_add_noise['t_aug'] = t_noised_vec
            pred_add_noise['rot'] = rot_mat
            pred_add_noise['t'] = t_vec

        if self.opt.full_fps:
            fps_pc = input_list['pc_fps']
            fps_inv_z, _, fps_eq_z, t_fps_vec = self.netEncoder(fps_pc)
            fps_rot_mat = self.netRotation(fps_eq_z)
            pred_fps['rot_aug'] = fps_rot_mat
            pred_fps['t_aug'] = t_fps_vec
            pred_fps['rot'] = rot_mat
            pred_fps['t'] = t_vec

        if self.opt.full_remove_knn > 0:
            part_pc = input_list['pc_part']
            part_inv_z, _, part_eq_z, t_part_vec = self.netEncoder(part_pc)
            part_rot_mat = self.netRotation(part_eq_z)
            pred_remove_knn['rot_aug'] = part_rot_mat
            pred_remove_knn['t_aug'] = t_part_vec
            pred_remove_knn['rot'] = rot_mat
            pred_remove_knn['t'] = t_vec

        if self.opt.full_resample:
            sample_pc = input_list['pc_sample']
            sample_inv_z, _, sample_eq_z, t_sample_vec = self.netEncoder(sample_pc)
            sample_rot_mat = self.netRotation(sample_eq_z)
            pred_resample['rot_aug'] = sample_rot_mat
            pred_resample['t_aug'] = t_sample_vec
            pred_resample['rot'] = rot_mat
            pred_resample['t'] = t_vec

        if self.opt.full_randRT:
            randRT_pc = input_list['pc_randRT']
            randRT_inv_z, _, randRT_eq_z, t_randRT_vec = self.netEncoder(randRT_pc)
            randRT_decoded_pc_inv, _ = self.netDecoder(randRT_inv_z)
            pred_randRT['pc'] = pc_at_inv
            pred_randRT['pc_randRT'] = randRT_decoded_pc_inv

        if self.opt.full_apply_can_rot:
            rotated_recon, trot, rand_t = pc_utils.rotate(recon_pc_inv, "so3", self.opt.base_device, return_trot=True)
            #expected_rot = trot.get_matrix()[:, :3, :3].detach()
            expected_rot = trot
            _, _, rotated_recon_eq_z, t_rot_can_vec = self.netEncoder(rotated_recon.detach())
            rotated_recon_rot_mat = self.netRotation(rotated_recon_eq_z).squeeze(-1)
            pred_can_rot['rot_aug'] = rotated_recon_rot_mat
            pred_can_rot['t_aug'] = t_rot_can_vec
            pred_can_rot['rot'] = expected_rot
            pred_can_rot['t'] = torch.zeros_like(t_rot_can_vec, device=self.opt.base_device)
            #pred_can_rot['t'] = rand_t
            
        if mode == 'normal':
            return pred_recon, pred_add_noise, pred_fps, pred_remove_knn, pred_resample, pred_randRT, pred_can_rot
        elif mode == 'cross':
            pred_RandD = {
                'inv_f': inv_z,
                'inv_f_p': inv_z_p,
                'recon_pc': pc_at_inv,
                'rot':rot_mat,
                't':t_vec
            }
            return pred_recon, pred_add_noise, pred_fps, pred_remove_knn, \
                   pred_resample, pred_randRT, pred_can_rot, pred_RandD
        else:
            raise NotImplementedError

    def online_data_augment(self, pc, use_rand_trans=True):
        pc = pc.permute(0, 2, 1).to(self.opt.base_device)
        input_list = {}
        # False
        if self.opt.full_downsample:
            pc = pc_utils.sample(pc.clone().detach(), self.opt.full_resample_npoints, self.opt.base_device)
        # True
        if use_rand_trans:
            pc, trot, rand_t = pc_utils.rotate(pc, self.opt.base_rot, self.opt.base_device, t=self.opt.base_se3_T,
                                       return_trot=True)
            input_list['pc_full'] = pc
            input_list['rand_tort'] = trot
            input_list['rand_t'] = rand_t
        # True
        # add randn bias 
        if self.opt.full_add_noise:
            noised_pc = pc.clone() + self.opt.full_noise_amp * torch.randn(pc.size(), device=self.opt.base_device)
            input_list['pc_noised'] = noised_pc
        # True
        # what
        if self.opt.full_remove_knn:
            id_to_remove = torch.randint(0, pc.size(2), (pc.size(0), 1), device=self.opt.base_device).contiguous()
            pc_part = pc_utils.remove_knn(pc.clone().detach().contiguous(), id_to_remove,
                                          k=self.opt.full_remove_knn_points, device=self.opt.base_device)
            input_list['pc_part'] = pc_part
        # True
        # FPS
        if self.opt.full_fps:
            num_fps_points = torch.randint(self.opt.full_min_num_fps_points,
                                           self.opt.full_max_num_fps_points, (1,)).item()
            pc = pc.clone().detach()
            _, new_pc = pc_utils.farthest_point_sample_xyz(pc.transpose(1, 2), num_fps_points)
            pc_fps = new_pc.transpose(1, 2).detach()
            input_list['pc_fps'] = pc_fps
        # True
        # 2500 points
        if self.opt.full_resample:
            pc_sample = pc_utils.sample(pc.clone().detach(), self.opt.full_resample_npoints, self.opt.base_device)
            input_list['pc_sample'] = pc_sample
        # True
        if self.opt.full_randRT:
            pc_randRT = pc_utils.rotate(pc, self.opt.base_rot, self.opt.base_device,
                                        t=self.opt.base_se3_T, return_trot=False)
            input_list['pc_randRT'] = pc_randRT

        return input_list
