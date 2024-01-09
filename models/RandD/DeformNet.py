import torch
import torch.nn as nn
import torch.nn.functional as F


# This network is used to predict the
# displacement of each point according to each keypoint
# only one difference. The loss
# for segmentation, the point cloud is restricted to local
# but for this one, one point can be determined be any keypoints
class Deform_net(nn.Module):
    def __init__(self, opt):
        super(Deform_net, self).__init__()
        self.opt = opt
        self.conv1d_block = nn.Sequential(
            nn.Linear(opt.cross_deform_dim_input, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, opt.cross_seg_class),
        )

    def forward(self, inv_feature, f_p):
        # inv_feature: bs x feature_dim
        # f_f:  bs x nofp x feature_dim_2
        bs, nofp = f_p.shape[0], f_p.shape[1]
        inv_feature_expand = inv_feature.unsqueeze(1).repeat(1, nofp, 1)
        combined_f = torch.cat([inv_feature_expand, f_p], dim=-1)
        out_f = self.conv1d_block(combined_f)
        ##############################################################################  softmax's dim has error
        ##############################################################################  i think dim should be 2
        ##############################################################################  when update its, train full
        #out_seg_class_DEBUG = F.softmax(out_f, dim=1)
        out_seg_class = F.softmax(out_f, dim=2)
        return out_seg_class


if __name__ == '__main__':
    device = torch.device("cuda")
    B, N = 16, 512
    inv = torch.rand(B, N).to(device)
    hand_craft_p_f = torch.rand(B, 1024, 128).to(device)
    import argparse

    opt = argparse.ArgumentParser()
    #  self.npoint // self.npatch  must be dividable
    opt.add_argument('--deform_dim_input', type=int, default=512 + 128)
    opt.add_argument('--seg_class', type=int, default=12)
    opt = opt.parse_args()

    network = Deform_net(opt).to(device)
    out_seg = network(inv, hand_craft_p_f)
    sdf = 1
