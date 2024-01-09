import torch
from vis_pc import show_pc,show_pc_seg
import os



def DEBUG_i(index):
    show_pc(target_shape_loaded[index].cpu().numpy())
    show_pc(source_selected_top1[index].cpu().numpy())
    show_pc(source_after_deforms[index].cpu().numpy())
    print("befor deform loss:",only_r_loss_all[index].cpu().numpy())
    print("after deform loss:",cd_loss_all[index].cpu().numpy())

def vis_seg(index):
    show_pc_seg(target_shape_loaded[index].cpu().numpy(),full_target_seg[index].cpu())


if __name__=="__main__":
    load_root = "chair_log_full/TEXT_NEWFULL_cage_retrieval/chair-12kpt/test_partial"
    source_after_deforms = torch.load(os.path.join(load_root,"source_after_deforms.pt"))
    source_selected_top1 = torch.load(os.path.join(load_root,"source_selected_top1.pt"))
    source_shapes = torch.load(os.path.join(load_root,"source_shapes.pt"))
    source_tokens = torch.load(os.path.join(load_root,"source_tokens.pt"))
    target_shape_loaded = torch.load(os.path.join(load_root,"target_shape_loaded.pt"))
    full_target_seg = torch.load(os.path.join(load_root,"full_target_seg.pt"))
    cd_loss_all = torch.load(os.path.join(load_root,"cd_loss_all.pt"))
    only_r_loss_all = torch.load(os.path.join(load_root,"only_r_loss_all.pt"))
    
    #DEBUG_i(4)
    vis_seg(5)
    
    #5,6,10,13,18,20,21,25