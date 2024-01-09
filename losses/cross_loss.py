import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d import loss
from losses.supp_loss_funcs import RotationLoss, OrthogonalLoss, u_chamfer_distance
from pytorch3d.ops.knn import knn_points
from visualization.vis_pc import show_pc
# developed by yan
# All these losses are used to supervise the consistency of the full and partial branches
# as well as the retrieval and deformation process
# for each loss, a detailed annotation is provided above


class Cross_loss(nn.Module):
    def __init__(self, opt):
        super(Cross_loss, self).__init__()
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
        return torch.sum(feat_loss) / (bs + 1) * self.opt.cross_weight_feature

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

    #  supervise the direction vectors between keypoints. A consistency term between the full and partial branches
    def cal_part_direction_loss(self, pred_rd_full, pred_rd_partial):
        #
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
    def cal_keypoints_loss(self, pred_rd_full, pred_rd_partial):
        kp_full = pred_rd_full['keypoints']
        kp_partial = pred_rd_partial['keypoints']
        #   kp_full: bs x nofKP x 3
        #   kp_partial: bs x nofKP x 3
        bs, nofKP, _ = kp_full.shape
        kp_loss_p = F.mse_loss(kp_full, kp_partial)
        return kp_loss_p * self.opt.cross_weight_kp

    # most important loss function, retrieval tokens represent the deformation error
    def cal_R_and_D_loss(self, pred_rd_full, pred_rd_partial, min_support_pixels=20, max_bear_error=20.):
        r_tokens_full = pred_rd_full['retrieval_tokens']
        r_tokens_partial = pred_rd_partial['retrieval_tokens']
        bs, nofKP, _ = r_tokens_full.shape

        pc_seg_partial = pred_rd_partial['pc_seg']
        kp_support_pixel_num_partial = torch.sum(pc_seg_partial, dim=1)
        pc_seg_full = pred_rd_full['pc_seg']
        kp_support_pixel_num_full = torch.sum(pc_seg_full, dim=1)

        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['recon_pc']
        deform_field = pred_rd_full['deform_field']   # bs x nofP x nofKP
        kp_full = pred_rd_full['keypoints']
        kp_partial = pred_rd_partial['keypoints']   # bs x nofKP x 3

        pc_full_deform = pc_full + (deform_field @ (kp_partial - kp_full))
        x_nn = knn_points(pc_partial, pc_full_deform, K=1)
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
    
    def cal_R_and_D_lossV2(self, pred_rd_full, pred_rd_partial, fp_list, min_support_pixels=20, max_bear_error=20.):
        r_tokens_full = pred_rd_full['retrieval_tokens']
        r_tokens_partial = pred_rd_partial['retrieval_tokens']
        bs, nofKP, _ = r_tokens_full.shape

        pc_seg_partial = pred_rd_partial['pc_seg']
        kp_support_pixel_num_partial = torch.sum(pc_seg_partial, dim=1)
        pc_seg_full = pred_rd_full['pc_seg']
        kp_support_pixel_num_full = torch.sum(pc_seg_full, dim=1)

        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['recon_pc']
        deform_field = pred_rd_full['deform_field']   # bs x nofP x nofKP
        kp_full = pred_rd_full['keypoints']
        kp_partial = pred_rd_partial['keypoints']   # bs x nofKP x 3

        pc_full_deform = pc_full + (deform_field @ (kp_partial - kp_full))
        pc_full_deform_to_partial = []
        for i in range(bs):
            pc_full_deform_now = pc_full_deform[i]
            pc_full_deform_to_partial_now = pc_full_deform_now[fp_list[i],:].unsqueeze(0)
            pc_full_deform_to_partial.append(pc_full_deform_to_partial_now)
        pc_full_deform_to_partial = torch.cat(pc_full_deform_to_partial,dim = 0)
        x_nn = knn_points(pc_partial, pc_full_deform_to_partial, K=1)
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
    
    # deformed full shape should match the partial shape
    def cal_deform_flow_loss(self, pred_rd_full, pred_rd_partial):
        # deform field
        deform_field = pred_rd_full['deform_field']
        pc_full = pred_rd_full['recon_pc']
        pc_partial = pred_rd_partial['recon_pc']
        kp_full = pred_rd_full['keypoints']
        kp_partial = pred_rd_partial['keypoints']  # bs x nofKP x 3

        pc_full_deform = pc_full + (deform_field @ (kp_partial - kp_full))
        deform_loss, _, _ = u_chamfer_distance(pc_partial, pc_full_deform)
        return deform_loss * self.opt.cross_weight_deform

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

    #   keypoints should be evenly distributed in the point cloud
    def cal_kp_recon_loss(self, pred_rd_full):
        kp_full = pred_rd_full['keypoints']
        recon_pc_full = pred_rd_full['recon_pc']
        loss_kp_recon_full, _ = self.criterionPCRecon(kp_full, recon_pc_full)
        return loss_kp_recon_full * self.opt.cross_weight_kp_recon

    def get_all_loss_terms(self, pred_rd_full=None, pred_rd_partial=None, fp_list=None, cham = False):
        loss_name_list = {}
        loss_sum = 0.0
        #loss_rd = self.cal_R_and_D_loss(pred_rd_full, pred_rd_partial)
        loss_rd = 0
        loss_name_list['rd'] = loss_rd
        loss_sum += loss_rd
        if self.opt.cross_seg_flag:
            loss_seg = self.cal_seg_loss(pred_rd_full, pred_rd_partial, fp_list)
            loss_name_list['seg'] = loss_seg
            loss_sum += loss_seg
        if self.opt.cross_feat_flag:
            loss_feat = self.cal_feature_loss(pred_rd_full, pred_rd_partial, fp_list)
            loss_name_list['feat'] = loss_feat
            loss_sum += loss_feat
        if self.opt.cross_recon_flag:
            if cham == True:
                loss_recon = self.cal_recon_loss(pred_rd_full, pred_rd_partial, fp_list)
            else:
                loss_recon = self.cal_recon_loss(pred_rd_full, pred_rd_partial)
            loss_name_list['recon'] = loss_recon
            loss_sum += loss_recon
        if self.opt.cross_pd_flag:
            loss_pd = self.cal_part_direction_loss(pred_rd_full, pred_rd_partial)
            loss_name_list['pd'] = loss_pd
            loss_sum += loss_pd
        if self.opt.cross_kp_flag:
            loss_kp = self.cal_keypoints_loss(pred_rd_full, pred_rd_partial)
            loss_name_list['kp'] = loss_kp
            loss_sum += loss_kp
        ####
        #if self.opt.cross_deform_flag:
        #    loss_deform = self.cal_deform_flow_loss(pred_rd_full, pred_rd_partial)
        #    loss_name_list['deform'] = loss_deform
        #    loss_sum += loss_deform
        if self.opt.cross_kp_seg_flag:
            loss_kp_seg = self.cal_kp_seg_loss(pred_rd_full, pred_rd_partial)
            loss_name_list['kp_seg'] = loss_kp_seg
            loss_sum += loss_kp_seg
        if self.opt.cross_kp_recon_flag:
            loss_kp_recon = self.cal_kp_recon_loss(pred_rd_full)
            loss_name_list['kp_recon'] = loss_kp_recon
            loss_sum += loss_kp_recon
        return loss_name_list, loss_sum
    

    def get_all_loss_terms_source(self, pred_rd_full=None, pred_rd_partial=None, fp_list=None):
        loss_name_list = {}
        loss_sum = 0.0
        loss_rd = self.cal_R_and_D_loss(pred_rd_full, pred_rd_partial)
        loss_name_list['rd'] = loss_rd
        loss_sum += loss_rd * self.opt.cross_source_weight_rd

        if self.opt.cross_deform_flag:
            loss_deform = self.cal_deform_flow_loss(pred_rd_full, pred_rd_partial)
            loss_name_list['deform'] = loss_deform
            loss_sum += loss_deform * self.opt.cross_source_weight_rd
       
        return loss_name_list, loss_sum
    
    def get_all_loss_terms_fullcross(self, pred_rd_full1=None, pred_rd_full2=None):
        loss_name_list = {}
        loss_sum = 0.0
        loss_name_list = {}
        loss_sum = 0.0
        #loss_rd = self.cal_R_and_D_loss(pred_rd_full, pred_rd_partial)
        loss_rd = 0
        loss_name_list['rd'] = loss_rd
        loss_sum += loss_rd
        if self.opt.cross_seg_flag:
            loss_seg = self.cal_seg_loss(pred_rd_full1, pred_rd_full2)
            loss_name_list['seg'] = loss_seg
            loss_sum += loss_seg
        if self.opt.cross_pd_flag:
            loss_pd = self.cal_part_direction_loss(pred_rd_full1, pred_rd_full2)
            loss_name_list['pd'] = loss_pd
            loss_sum += loss_pd
        if self.opt.cross_kp_flag:
            loss_kp = self.cal_keypoints_loss(pred_rd_full1, pred_rd_full2)
            loss_name_list['kp'] = loss_kp
            loss_sum += loss_kp
        if self.opt.cross_kp_seg_flag:
            loss_kp_seg = self.cal_kp_seg_loss(pred_rd_full1, pred_rd_full2)
            loss_name_list['kp_seg'] = loss_kp_seg
            loss_sum += loss_kp_seg
        if self.opt.cross_kp_recon_flag:
            loss_kp_recon = self.cal_kp_recon_loss(pred_rd_full1)
            loss_name_list['kp_recon'] = loss_kp_recon
            loss_sum += loss_kp_recon
            
            loss_kp_recon2 = self.cal_kp_recon_loss(pred_rd_full2)
            loss_name_list['kp_recon2'] = loss_kp_recon2
            loss_sum += loss_kp_recon2
            
        return loss_name_list, loss_sum
