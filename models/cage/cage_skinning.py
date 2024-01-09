import pytorch3d.loss
import pytorch3d.utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from einops import rearrange

from models.cage.utils import normalize_to_box
from models.cage.cages import deform_with_MVC

class CageSkinning(nn.Module):
    
    def __init__(self, opt):
        super(CageSkinning, self).__init__()
        self.opt = opt
        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices,template_faces)
        
        self.decoder_influence_offset = nn.Sequential(
            nn.Linear(opt.cross_seg_dim_input, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 42)
        )
        
    
    def create_cage(self):
        # cage (1, N, 3)
        ico_sphere_div = 1
        cage_size = 1.4
        
        # ico_shape_div = 1
        mesh = pytorch3d.utils.ico_sphere(ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        # cage_size = 1.4
        init_cage_V = cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    
    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage
    
    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.opt.n_keypoints, self.template_vertices.shape[2]), requires_grad=True)
    
    def get_features(self,source_f,target_f):
        combined_f = torch.cat([source_f, target_f], dim=-1)
        combined_f = combined_f.unsqueeze(1)
        return combined_f
    
    def forward(self,pred_f_RandD,pred_p_RandD):
        """
        source_shape (B,3,N)
        target_shape (B,3,N)
        """
        source_shape = pred_f_RandD['recon_pc']
        target_shape = pred_p_RandD['recon_pc']
        source_f = pred_f_RandD['inv_f']
        target_f = pred_p_RandD['inv_f']
        source_keypoints = pred_f_RandD['keypoints'].transpose(1,2)
        target_keypoints = pred_p_RandD['keypoints'].transpose(1,2)
        cage = self.template_vertices
        self.influence = self.influence_param[None]
        
        B, _, _ = source_shape.size()
        cage = self.optimize_cage(cage, source_shape)
        
        
        ### need to cal the influce_offset
        input_feature = self.get_features(source_f,target_f)
        influce_offset = self.decoder_influence_offset(input_feature)
        self.influence = self.influence + influce_offset
        
        ###DEBUG
        distance = torch.sum((source_keypoints[..., None] - cage[:,:, None]) ** 2, dim=1)
        n_influence = int(distance.shape[2] / distance.shape[1])
        n_influence = max(5, n_influence)
        threshold = torch.topk(distance,n_influence,largest=False)[0][:, :, -1]
        threshold = threshold[..., None]
        keep = distance <= threshold
        influence = self.influence *keep
        
        base_cage = cage
        keypoints_offset = target_keypoints - source_keypoints
        cage_offset = torch.sum(keypoints_offset[..., None] * influence[:,None], dim = 2)
        #cageoffset
        new_cage = base_cage + cage_offset
        
        cage = cage.transpose(1,2)
        new_cage = new_cage.transpose(1,2)
        deform_shapes, weights, _ = deform_with_MVC(
            cage,new_cage, self.template_faces.expand(B,-1,-1),source_shape.transpose(1,2), verbose=True
        )
        
        self.deformed_shapes = deform_shapes
        
        outputs = {
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": self.deformed_shapes.transpose(1,2),
            "weight": weights,
            "influence": influce_offset
        }
        
        return outputs
        

class CageSkinningV2(nn.Module):
    
    def __init__(self, opt):
        super(CageSkinningV2, self).__init__()
        self.opt = opt
        template_vertices, template_faces = self.create_cage()
        self.init_template(template_vertices,template_faces)
        
        self.decoder_influence_offset = nn.Sequential(
            nn.Linear(opt.cross_seg_dim_input, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 42*3)
        )
        
    
    def create_cage(self):
        # cage (1, N, 3)
        ico_sphere_div = 1
        cage_size = 1.4
        
        # ico_shape_div = 1
        mesh = pytorch3d.utils.ico_sphere(ico_sphere_div, device='cuda:0')
        init_cage_V = mesh.verts_padded()
        init_cage_F = mesh.faces_padded()
        # cage_size = 1.4
        init_cage_V = cage_size * normalize_to_box(init_cage_V)[0]
        init_cage_V = init_cage_V.transpose(1, 2)
        return init_cage_V, init_cage_F
    
    def optimize_cage(self, cage, shape, distance=0.4, iters=100, step=0.01):
        """
        pull cage vertices as close to the origin, stop when distance to the shape is bellow the threshold
        """
        for _ in range(iters):
            vector = -cage
            current_distance = torch.sum((cage[..., None] - shape[:, :, None]) ** 2, dim=1) ** 0.5
            min_distance, _ = torch.min(current_distance, dim=2)
            do_update = min_distance > distance
            cage = cage + step * vector * do_update[:, None]
        return cage
    
    def init_template(self, template_vertices, template_faces):
        # save template as buffer
        self.register_buffer("template_faces", template_faces)
        self.register_buffer("template_vertices", template_vertices)
        
        # n_keypoints x number of vertices
        self.influence_param = nn.Parameter(torch.zeros(self.opt.n_keypoints, self.template_vertices.shape[2]), requires_grad=True)
    
    def get_features(self,source_f,target_f):
        combined_f = torch.cat([source_f, target_f], dim=-1)
        #combined_f = combined_f.unsqueeze(1)
        return combined_f
    
    def forward(self,pred_f_RandD,pred_p_RandD):
        """
        source_shape (B,3,N)
        target_shape (B,3,N)
        """
        source_shape = pred_f_RandD['recon_pc']
        target_shape = pred_p_RandD['recon_pc']
        source_f = pred_f_RandD['inv_f']
        target_f = pred_p_RandD['inv_f']
        source_keypoints = pred_f_RandD['keypoints'].transpose(1,2)
        target_keypoints = pred_p_RandD['keypoints'].transpose(1,2)
        cage = self.template_vertices
        self.influence = self.influence_param[None]
        
        B, _, _ = source_shape.size()
        cage = self.optimize_cage(cage, source_shape)
        
        
        ### need to cal the influce_offset
        input_feature = self.get_features(source_f,target_f)
        influce_offset = self.decoder_influence_offset(input_feature)
        cage_offset = influce_offset.contiguous().view(B,3,-1)
    
        base_cage = cage
        new_cage = base_cage + cage_offset
        
        cage = cage.transpose(1,2)
        new_cage = new_cage.transpose(1,2)
        deform_shapes, weights, _ = deform_with_MVC(
            cage,new_cage, self.template_faces.expand(B,-1,-1),source_shape.transpose(1,2), verbose=True
        )
        
        self.deformed_shapes = deform_shapes
        
        outputs = {
            "cage": cage,
            "cage_face": self.template_faces,
            "new_cage": new_cage,
            "deformed": self.deformed_shapes.transpose(1,2),
            "weight": weights,
            "influence": influce_offset
        }
        
        return outputs
             
