import torch
import torch.nn as nn
from pytorch3d import loss
from losses.supp_loss_funcs import RotationLoss, OrthogonalLoss


class Full_loss(nn.Module):
    def __init__(self, opt):
        super(Full_loss, self).__init__()
        self.opt = opt
        self.criterionPCRecon = loss.chamfer_distance
        self.criterionMSE = nn.MSELoss()
        self.criterionROT = RotationLoss(device=opt.base_device, which_metric=opt.full_which_rot_metric)
        self.criterionOrtho = OrthogonalLoss(device=opt.base_device, which_metric=opt.full_which_ortho_metric)

    def cal_rot_dist(self, R1, R2, detach_R2=False):
        if detach_R2:
            dist = self.criterionROT(R1, R2.detach())
        else:
            dist = self.criterionROT(R1, R2)
        return dist

    def cal_trans_dist(self, T1, T2, detach_T2=False):
        if detach_T2:
            dist = self.criterionMSE(T1, T2.detach())
        else:
            dist = self.criterionMSE(T1, T2)
        return dist

    def cal_recon_loss(self, pred_recon):
        # pred_recon is a list
        recon_pc = pred_recon['recon_pc']
        pc = pred_recon['pc']
        loss_recon_1 = self.criterionPCRecon(recon_pc.permute(0, 2, 1), pc.permute(0, 2, 1))[0] * \
                       self.opt.full_weight_recon1
        return loss_recon_1

    def cal_recon_inv_loss(self, pred_recon):
        recon_pc_inv = pred_recon['recon_pc_inv']
        pc_at_inv = pred_recon['pc_at_inv']
        if self.opt.full_detached_recon_2:
            loss_recon_2 = self.criterionPCRecon(pc_at_inv.permute(0, 2, 1).detach(),
                                                 recon_pc_inv.permute(0, 2, 1))[0] * self.opt.full_weight_recon2
        else:
            loss_recon_2 = self.criterionPCRecon(pc_at_inv.permute(0, 2, 1),
                                                 recon_pc_inv.permute(0, 2, 1))[0] * self.opt.full_weight_recon2
        return loss_recon_2

    def cal_ortho_loss(self, pred_recon):
        loss_ortho = self.criterionOrtho(pred_recon['rot']) * self.opt.full_weight_ortho
        t_vec = pred_recon['t_vec']
        t_rand = pred_recon['t_rand']
        loss_t = self.cal_trans_dist(t_vec,t_rand,True)
        return loss_ortho + loss_t
        #return loss_ortho

    def cal_add_noise_loss(self, pred_add_noise=None):
        if pred_add_noise is None:
            return 0.0, 0.0, 0.0
        if self.opt.full_add_noise:
            noised_rot_mat = pred_add_noise['rot_aug']
            t_noised_vec = pred_add_noise['t_aug']
            rot_mat = pred_add_noise['rot']
            t_vec = pred_add_noise['t']
            loss_noised_rot = self.cal_rot_dist(noised_rot_mat, rot_mat,
                                                self.opt.full_detach_aug_loss) * self.opt.full_weight_noise_R
            loss_noised_T = self.cal_trans_dist(t_noised_vec, t_vec,
                                                self.opt.full_detach_aug_loss) * self.opt.full_weight_noise_T
            if self.opt.full_add_ortho_aug:
                loss_noised_ortho = self.criterionOrtho(noised_rot_mat) * self.opt.full_weight_ortho
                return loss_noised_rot, loss_noised_T, loss_noised_ortho
            else:
                return loss_noised_rot, loss_noised_T, 0.0
        else:
            return 0.0, 0.0, 0.0

    def cal_fps_loss(self, pred_fps=None):
        if pred_fps is None:
            return 0.0, 0.0, 0.0
        if self.opt.full_fps:
            fps_rot_mat = pred_fps['rot_aug']
            t_fps_vec = pred_fps['t_aug']
            rot_mat = pred_fps['rot']
            t_vec = pred_fps['t']
            loss_fps_rot = self.cal_rot_dist(fps_rot_mat, rot_mat,
                                             self.opt.full_detach_aug_loss) * self.opt.full_weight_fps_R
            loss_fps_T = self.cal_trans_dist(t_fps_vec, t_vec,
                                             self.opt.full_detach_aug_loss) * self.opt.full_weight_fps_T
            if self.opt.full_add_ortho_aug:
                loss_fps_ortho = self.criterionOrtho(fps_rot_mat) * self.opt.full_weight_ortho
                return loss_fps_rot, loss_fps_T, loss_fps_ortho
            else:
                return loss_fps_rot, loss_fps_T, 0.0
        else:
            return 0.0, 0.0, 0.0

    def cal_knn_loss(self, pred_remove_knn=None):
        if pred_remove_knn is None:
            return 0.0, 0.0, 0.0
        if self.opt.full_remove_knn:
            remove_knn_rot_mat = pred_remove_knn['rot_aug']
            t_remove_knn_vec = pred_remove_knn['t_aug']
            rot_mat = pred_remove_knn['rot']
            t_vec = pred_remove_knn['t']
            loss_remove_knn_rot = self.cal_rot_dist(remove_knn_rot_mat, rot_mat,
                                                    self.opt.full_detach_aug_loss) * self.opt.full_weight_part_R
            loss_remove_knn_T = self.cal_trans_dist(t_remove_knn_vec, t_vec,
                                                    self.opt.full_detach_aug_loss) * self.opt.full_weight_part_T
            if self.opt.full_add_ortho_aug:
                loss_remove_knn_ortho = self.criterionOrtho(remove_knn_rot_mat) * self.opt.full_weight_ortho
                return loss_remove_knn_rot, loss_remove_knn_T, loss_remove_knn_ortho
            else:
                return loss_remove_knn_rot, loss_remove_knn_T, 0.0
        else:
            return 0.0, 0.0, 0.0

    def cal_resample_loss(self, pred_resample=None):
        if pred_resample is None:
            return 0.0, 0.0, 0.0
        if self.opt.full_fps:
            resample_rot_mat = pred_resample['rot_aug']
            t_resample_vec = pred_resample['t_aug']
            rot_mat = pred_resample['rot']
            t_vec = pred_resample['t']
            loss_resample_rot = self.cal_rot_dist(resample_rot_mat, rot_mat,
                                                  self.opt.full_detach_aug_loss) * self.opt.full_weight_sample_R
            loss_resample_T = self.cal_trans_dist(t_resample_vec, t_vec,
                                                  self.opt.full_detach_aug_loss) * self.opt.full_weight_sample_T
            if self.opt.full_add_ortho_aug:
                loss_resample_ortho = self.criterionOrtho(resample_rot_mat) * self.opt.full_weight_ortho
                return loss_resample_rot, loss_resample_T, loss_resample_ortho
            else:
                return loss_resample_rot, loss_resample_T, 0.0
        else:
            return 0.0, 0.0, 0.0

    def cal_can_rot_loss(self, pred_can_rot=None):
        if pred_can_rot is None:
            return 0.0, 0.0, 0.0
        if self.opt.full_apply_can_rot:
            can_rot_rot_mat = pred_can_rot['rot_aug']
            t_can_rot_vec = pred_can_rot['t_aug']
            rot_mat = pred_can_rot['rot']
            t_vec = pred_can_rot['t']
            loss_can_rot_rot = self.cal_rot_dist(can_rot_rot_mat, rot_mat,
                                                 False) * self.opt.full_weight_can_R
            loss_can_rot_T = self.cal_trans_dist(t_can_rot_vec, t_vec,
                                                 False) * self.opt.full_weight_can_T
            if self.opt.full_add_ortho_aug:
                loss_can_rot_ortho = self.criterionOrtho(can_rot_rot_mat) * self.opt.full_weight_ortho
                return loss_can_rot_rot, loss_can_rot_T, loss_can_rot_ortho
            else:
                return loss_can_rot_rot, loss_can_rot_T, 0.0
        else:
            return 0.0, 0.0, 0.0

    def cal_randrt_loss(self, pred_randRT):
        if pred_randRT is None:
            return 0.0
        if self.opt.full_randRT:
            pc = pred_randRT['pc']
            recon_pc_randRT = pred_randRT['pc_randRT']
            loss_randrt = self.criterionPCRecon(recon_pc_randRT, pc.permute(0, 2, 1))[0] * \
                          self.opt.full_weight_rand_RT
            return loss_randrt
        else:
            return 0.0

    def get_all_loss_terms(self, pred_recon=None, pred_add_noise=None, pred_fps=None, pred_remove_knn=None,
                           pred_resample=None, pred_randRT=None, pred_can_rot=None):
        loss_name_list = {}
        loss_sum = 0.0
        loss_recon_1 = self.cal_recon_loss(pred_recon)
        loss_name_list['recon_1'] = loss_recon_1
        loss_sum += loss_recon_1

        loss_recon_2 = self.cal_recon_inv_loss(pred_recon)
        loss_name_list['recon_2'] = loss_recon_2
        loss_sum += loss_recon_2

        loss_ortho = self.cal_ortho_loss(pred_recon)
        loss_name_list['ortho'] = loss_ortho
        loss_sum += loss_ortho

        if self.opt.full_add_noise:
            loss_noised_rot, loss_noised_T, loss_noised_ortho = self.cal_add_noise_loss(pred_add_noise)
            loss_name_list['noised_rot'] = loss_noised_rot
            loss_name_list['noised_T'] = loss_noised_T
            loss_sum += (loss_noised_ortho + loss_noised_rot + loss_noised_T)

        if self.opt.full_fps:
            loss_fps_rot, loss_fps_T, loss_fps_ortho = self.cal_fps_loss(pred_fps)
            loss_name_list['fps_rot'] = loss_fps_rot
            loss_name_list['fps_T'] = loss_fps_T
            loss_sum += (loss_fps_ortho + loss_fps_rot + loss_fps_T)

        if self.opt.full_remove_knn:
            loss_knn_rot, loss_knn_T, loss_knn_ortho = self.cal_knn_loss(pred_remove_knn)
            loss_name_list['knn_rot'] = loss_knn_rot
            loss_name_list['knn_T'] = loss_knn_T
            loss_sum += (loss_knn_ortho + loss_knn_rot + loss_knn_T)

        if self.opt.full_resample:
            loss_resample_rot, loss_resample_T, loss_resample_ortho = self.cal_resample_loss(pred_resample)
            loss_name_list['resample_rot'] = loss_resample_rot
            loss_name_list['resample_T'] = loss_resample_T
            loss_sum += (loss_resample_ortho + loss_resample_rot + loss_resample_T)

        if self.opt.full_apply_can_rot:
            loss_can_rot, loss_can_T, loss_can_ortho = self.cal_can_rot_loss(pred_can_rot)
            loss_name_list['can_rot'] = loss_can_rot
            loss_name_list['can_T'] = loss_can_T
            loss_sum += (loss_can_ortho + loss_can_rot + loss_can_T)

        if self.opt.full_randRT:
            loss_randrt = self.cal_randrt_loss(pred_randRT)
            loss_name_list['randrt'] = loss_randrt
            loss_sum += loss_randrt

        return loss_name_list, loss_sum
