import torch
import torch.nn as nn


# this file contains two classes
# The goal is to get the keypoint features
# and decide how to deform the source shape according to keypoint displacement

# given the point-wise feature and segmentation, get the
# feature of the corresponding keypoint

class RetrievalNet(nn.Module):
    def __init__(self, opt):
        super(RetrievalNet, self).__init__()
        self.opt = opt

    def forward(self, p_f, pc_seg):
        # p_f: bs x nof_p x c_dim
        # pc_seg: bs x nofP x 12
        # return retrieval tokens: bs x 12 x (c_dim)
        # here pc_seg is also differentiable
        pc_seg = pc_seg.permute(0, 2, 1)
        key_point_f = (pc_seg @ p_f)  # bs x 12 x (c_dim)
        return key_point_f


class RetrievalNetV2(nn.Module):
    def __init__(self, opt):
        super(RetrievalNetV2, self).__init__()
        self.opt = opt
        self.conv1d_block = nn.Sequential(
            nn.Linear(1020, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1020),
        )
    
    def forward(self,p_f, pc_seg):
        p_f = self.conv1d_block(p_f)
        pc_seg = pc_seg.permute(0, 2, 1)
        key_point_f = (pc_seg @ p_f)  # bs x 12 x (c_dim)
        return key_point_f,p_f

    


class DeformKeypoint(nn.Module):
    def __init__(self, opt):
        super(DeformKeypoint, self).__init__()
        self.opt = opt

    def forward(self, deform_w, keypoint_source, keypoint_target):
        # deform_w    bs x nofP x 12
        # keypoint_source: bs x 12 x 3
        # keypoint_target: bs x 12 x 3
        # return bs x nofP x 3    displacement of every point
        # source_point + displacement = target_point
        displacement = keypoint_target - keypoint_source  # displacement  bs x nofP x 3
        dis_per_point = (deform_w @ displacement)  # bs x nofP x 3
        return dis_per_point


class KeypointExaction(nn.Module):
    def __init__(self, opt):
        super(KeypointExaction, self).__init__()
        self.opt = opt
        self.keypoint_extraction_block = nn.Sequential(
            nn.Linear(opt.cross_keypoint_pred_dim_input, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3 * opt.cross_seg_class),
        )
        

    def forward(self, invariant_embedding):
        #  bs x c_dim  ===>  bs x 3 x keypoints
        bs = invariant_embedding.shape[0]
        keypoints = self.keypoint_extraction_block(invariant_embedding)
        keypoints = keypoints.contiguous().view(bs, 3, -1).permute(0, 2, 1)
        return keypoints

class KeypointExactionV2(nn.Module):
    def __init__(self, opt):
        super(KeypointExactionV2, self).__init__()
        self.opt = opt
        self.keypoint_extraction_block = nn.Sequential(
            nn.Linear(2040, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3 * opt.cross_seg_class),
        )
        

    def forward(self, inv_feature, pc_f_p):
        #  bs x c_dim  ===>  bs x 3 x keypoints
        bs, nofp = pc_f_p.shape[0], pc_f_p.shape[1]
        inv_feature_expand = inv_feature.unsqueeze(1).repeat(1, nofp, 1)
        combined_f = torch.cat([inv_feature_expand, pc_f_p], dim=-1)
        keypoints = self.keypoint_extraction_block(combined_f)
        keypoints = keypoints.contiguous().view(bs,nofp,3,-1).permute(0,1,3,2)
        keypoints = torch.mean(keypoints,dim = 1)
        #keypoints = keypoints.contiguous().view(bs, 3, -1).permute(0, 2, 1)
        return keypoints    


