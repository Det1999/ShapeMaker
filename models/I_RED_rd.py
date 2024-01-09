import torch
import itertools
import torch.nn as nn
from models.RandD.RandD import DeformKeypoint, RetrievalNet, KeypointExaction, RetrievalNetV2,KeypointExactionV2
from models.RandD.SegNet import Seg_net
from models.RandD.DeformNet import Deform_net

EPS = 1e-10


class I_RED_rd(nn.Module):
    def __init__(self, opt, partial = True):
        super(I_RED_rd, self).__init__()
        self.opt = opt
        self.full_deform_net = Deform_net(opt)
        self.full_seg_net = Seg_net(opt)
        self.full_retrieval_net = RetrievalNetV2(opt)
        self.full_keypoint_net = KeypointExactionV2(opt)
        
        if partial == True:
            self.partial_seg_net = Seg_net(opt)
            self.partial_retrieval_net = RetrievalNetV2(opt)
            self.partial_keypoint_net = KeypointExactionV2(opt)
        
        self.use_partial = partial


    def forward(self, pred_f_RandD, pred_p_RandD = None):
        # V9
        pred_f_RandD['inv_f'] = pred_f_RandD['inv_f'] / (torch.norm(pred_f_RandD['inv_f'], dim=-1, keepdim=True) + EPS)
        # V9
        if pred_p_RandD != None:
            pred_p_RandD['inv_f'] = pred_p_RandD['inv_f'] / (torch.norm(pred_p_RandD['inv_f'], dim=-1, keepdim=True) + EPS)
        
        
        f_pc_seg, _ = self.full_seg_net(pred_f_RandD['inv_f'], pred_f_RandD['inv_f_p'])
        f_deform_field = self.full_deform_net(pred_f_RandD['inv_f'], pred_f_RandD['inv_f_p'])
        f_retrieval_tokens = self.full_retrieval_net(pred_f_RandD['inv_f_p'], f_pc_seg)
        f_keypoints = self.full_keypoint_net(pred_f_RandD['inv_f'], pred_f_RandD['inv_f_p']) #2.0
        #f_keypoints = self.full_keypoint_net(pred_f_RandD['inv_f'])  #1.0
        
        if pred_p_RandD !=None:
            if self.use_partial:
                p_pc_seg, _ = self.partial_seg_net(pred_p_RandD['inv_f'], pred_p_RandD['inv_f_p'])
                p_retrieval_tokens = self.partial_retrieval_net(pred_p_RandD['inv_f_p'], p_pc_seg)
                p_keypoints = self.partial_keypoint_net(pred_p_RandD['inv_f'], pred_p_RandD['inv_f_p']) #2.0
                #p_keypoints = self.partial_keypoint_net(pred_p_RandD['inv_f']) #1.0
            else:
                p_pc_seg, _ = self.full_seg_net(pred_p_RandD['inv_f'], pred_p_RandD['inv_f_p'])
                p_retrieval_tokens = self.full_retrieval_net(pred_p_RandD['inv_f_p'], p_pc_seg)
                p_keypoints = self.full_keypoint_net(pred_p_RandD['inv_f'], pred_p_RandD['inv_f_p']) #2.0
                #p_keypoints = self.full_keypoint_net(pred_p_RandD['inv_f']) #1.0
        
        #DEBUG = torch.norm(pred_f_RandD['inv_f'], dim=-1, keepdim=True).unsqueeze(-1)
        
        pred_rd_full = {
            'inv_f': pred_f_RandD['inv_f'],
            'inv_f_p': pred_f_RandD['inv_f_p'],
            'recon_pc': pred_f_RandD['recon_pc'].permute(0, 2, 1),
            'pc_seg': f_pc_seg,
            'deform_field': f_deform_field,
            'retrieval_tokens': f_retrieval_tokens,
            # *
            'keypoints': f_keypoints*torch.norm(pred_f_RandD['inv_f'], dim=-1, keepdim=True).unsqueeze(-1),
            'rot':pred_f_RandD['rot'],
            't':pred_f_RandD['t']
        }
        
        if pred_p_RandD !=None:
            pred_rd_partial = {
                'inv_f': pred_p_RandD['inv_f'],
                'inv_f_p': pred_p_RandD['inv_f_p'],
                'recon_pc': pred_p_RandD['recon_pc'].permute(0, 2, 1),
                'pc_seg': p_pc_seg,
                'retrieval_tokens': p_retrieval_tokens,
                # *
                'keypoints': p_keypoints*torch.norm(pred_p_RandD['inv_f'], dim=-1, keepdim=True).unsqueeze(-1),
                'rot':pred_p_RandD['rot'],
                't':pred_p_RandD['t']
            }
        
            if 'full_pc_at_inv' in pred_p_RandD:
                    pred_rd_partial['full_pc_at_inv'] = pred_p_RandD['full_pc_at_inv'].permute(0,2,1)
        
        if pred_p_RandD !=None:
            return pred_rd_full, pred_rd_partial
        else:
            return pred_rd_full


    def get_optim_params(self):
        if self.use_partial:
            params = itertools.chain(self.full_seg_net.parameters(), self.full_deform_net.parameters(),
                                     self.full_keypoint_net.parameters(), self.full_retrieval_net.parameters(),
                                     self.partial_seg_net.parameters(), self.partial_retrieval_net.parameters(),
                                     self.partial_keypoint_net.parameters())
        else:
            params = itertools.chain(self.full_seg_net.parameters(), self.full_deform_net.parameters(),
                                     self.full_keypoint_net.parameters(), self.full_retrieval_net.parameters())

        return params
