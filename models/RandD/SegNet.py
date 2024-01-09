import torch
import torch.nn as nn
import torch.nn.functional as F


class Seg_net(nn.Module):
    def __init__(self, opt):
        super(Seg_net, self).__init__()
        self.opt = opt
        self.pc_seg_block = nn.Sequential(
            nn.Linear(opt.cross_seg_dim_input, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, opt.cross_seg_class),
        )
        self.keypoint_vote_block = nn.Sequential(
            nn.Linear(opt.cross_keypoint_vote_dim_input, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3),
        )

    def forward(self, inv_feature, pc_f_p):
        # inv_feature: bs x feature_dim
        # hand_crafted_p_f:  bs x nofp x feature_dim_2
        # pc_f_p : bs x c_dimx nofp
        bs, nofp = pc_f_p.shape[0], pc_f_p.shape[1]
        inv_feature_expand = inv_feature.unsqueeze(1).repeat(1, nofp, 1)
        combined_f = torch.cat([inv_feature_expand, pc_f_p], dim=-1)
        if self.opt.cross_return_seg_class:
            out_f = self.pc_seg_block(combined_f)
            out_seg_class = F.softmax(out_f, dim=-1)
            #DEBUG = out_seg_class[2]
            #DEBUG_idx = torch.topk(DEBUG,k=1,dim=-1)
            
            
        else:
            out_seg_class = None
        if self.opt.cross_return_keypoint_vote:
            out_keypoint_vote = self.keypoint_vote_block(combined_f)
            out_keypoint_vote = out_keypoint_vote / (torch.norm(out_keypoint_vote, dim=-1, keepdim=True) + 1e-5)
        else:
            out_keypoint_vote = None
        return out_seg_class, out_keypoint_vote


if __name__ == '__main__':
    device = torch.device("cuda")
    B, N = 16, 512
    inv = torch.rand(B, N).to(device)
    hand_craft_p_f = torch.rand(B, 1024, 128).to(device)
    import argparse

    opt = argparse.ArgumentParser()
    #  self.npoint // self.npatch  must be dividable
    opt.add_argument('--seg_dim_input', type=int, default=512 + 128)
    opt.add_argument('--seg_class', type=int, default=12)
    opt.add_argument('--keypoint_vote_dim_input', type=int, default=512 + 128)
    opt.add_argument('--return_keypoint_vote', type=bool, default=True)
    opt.add_argument('--return_seg_class', type=bool, default=False)
    opt = opt.parse_args()

    network = Seg_net(opt).to(device)
    out_seg = network(inv, hand_craft_p_f)
    sdf = 1
