import itertools
import json
import os
from collections import Counter
import traceback
import warnings
import torch.utils.data as data
from scipy.spatial.transform import Rotation
import ipdb
import numpy as np
import pandas
import torch
# from torch._six import container_abcs
import collections.abc
from utils.utils import resample_mesh, normalize_to_box
from utils.io import read_keypoints, read_pcd, read_mesh, find_files
from utils.data_utils import generate_partial_point_slice, sample_points, quaternion_rotation_matrix
import cv2

MESH_FILE_EXT = 'obj'
POINT_CLOUD_FILE_EXT = 'pts'
DO_NOT_BATCH = [
    'source_face', 'source_mesh', 'target_face', 'target_mesh', 'source_seg_points',
    'target_seg_points', 'source_seg_labels', 'target_seg_labels', 'source_mesh_obj',
    'target_mesh_obj']
CATEGORY2SYNSETOFFSET = {'airplane': '02691156', 'bag': '02773838', 'cap': '02954340', 'car': '02958343',
                         'chair': '03001627', 'earphone': '03261776', 'guitar': '03467517', 'knife': '03624134',
                         'lamp': '03636649', 'laptop': '03642806', 'motorbike': '03790512', 'mug': '03797390',
                         'pistol': '03948459', 'rocket': '04099429', 'skateboard': '04225987', 'table': '04379243'}

SYNSETOFFSET2CATEGORY = {v: k for k, v in CATEGORY2SYNSETOFFSET.items()}


class Shapes(data.Dataset):
    def __init__(self, opt):
        super(Shapes, self).__init__()
        self.opt = opt
        self.mesh_dir = opt.mesh_dir
        self.dataset = self.load_dataset()
        self.load_source = False
        self.partial_pc = opt.partial_pc
        self.partial_pc_dual = opt.partial_pc_dual
        print("dataset size %d" % len(self))

    def normalize(self, x):
        if self.opt.normalize == 'unit_box':
            pc, center, scale = normalize_to_box(x)
        else:
            raise ValueError()
        return pc, center, scale

    def _load_from_split_file(self, ):
        df = pandas.read_csv(self.opt.split_file)
        # find names from the category and split
        df_target = df.loc[(df.synsetId == int(self.opt.category)) & (df.split == self.opt.split)]
        # if (df.split == 'source').any():
        #     if int(self.opt.category) == 2871439:
        #         df_source = df.loc[((df.synsetId == int(self.opt.category)) | (df.synsetId == 2933112) | (df.synsetId == 3337140)) & (df.split == 'source')]
        #     else:
        
        ## add
        if (df.split == 'source').any():
            if int(self.opt.category) == 2871439:
                df_source = df.loc[((df.synsetId == int(self.opt.category)) | (df.synsetId == 2933112) | (df.synsetId == 3337140)) & (df.split == 'source')]
            else:
                df_source = df.loc[(df.synsetId == int(self.opt.category)) & (df.split == 'source')]
        else:
            if int(self.opt.category) == 2871439:
                df_source = df.loc[((df.synsetId == int(self.opt.category)) | (df.synsetId == 2933112) | (df.synsetId == 3337140)) & (df.split == 'val')]
            else:
                df_source = df.loc[(df.synsetId == int(self.opt.category)) & (df.split == 'val')]
        names = df_target.modelId.values
        self.source_names = df_source.modelId.values
        if self.opt.split == 'train':
            names = np.concatenate([names, self.source_names])
        return names

    def _load_from_files(self):
        files = find_files(os.path.join(self.opt.points_dir, self.opt.category), self.POINT_CLOUD_FILE_EXT)
        # extract name from files
        names = [x.split(os.path.sep)[-2] for x in files]
        names = sorted(names)
        return names

    def _load_seg_split_file(self, seg_split_file):
        with open(seg_split_file) as f:
            files = json.load(f)
            # ['04379243', '9db8f8c94dbbfe751d742b64ea8bc701'], ['02691156', '329a018e131ece70f23c3116d040903f'], ...
            names = [x.split(os.path.sep)[-2:] for x in files]
            # filter out other categories
            names = [x[1] for x in names if x[0] == self.opt.category]
        return names

    def _load_seg_split(self):
        seg_split_file = os.path.join(self.opt.seg_split_dir, 'shuffled_%s_file_list.json' % self.opt.split)
        return self._load_seg_split_file(seg_split_file)

    def _load_keypointnet_split(self, split_name):
        # load split
        with open(os.path.join(self.opt.keypointnet_dir, 'splits', split_name + '.txt')) as f:
            lines = f.read().splitlines()
        # line looks like this: 02691156-ecbb6df185a7b260760d31bf9510e4b7
        split = set([x[len(self.opt.category) + 1:] for x in lines if x.startswith(self.opt.category)])
        return split

    def _load_keypointnet(self):
        # load keypoints
        file_path = os.path.join(self.opt.keypointnet_dir, 'annotations',
                                 self.SYNSETOFFSET2CATEGORY[self.opt.category] + '.json')
        with open(file_path) as f:
            data = json.load(f)
        keypoints = {}
        for item in data:
            name = item['model_id']
            keypoints_sample = [x['xyz'] for x in item['keypoints']]
            keypoint_ids_sample = [x['semantic_id'] for x in item['keypoints']]
            keypoints_sample = np.array(keypoints_sample, dtype=np.float32)
            keypoints[name] = (keypoints_sample, keypoint_ids_sample)

        if self.opt.keypointnet_common_keypoints:
            # get most common keypoint ids
            ids = [list(id) for _, id in keypoints.values()]
            max_keypoints = len(set(itertools.chain(*ids)))
            # start with the highest number of keypoints
            success = False
            for n_common_keypoints in range(max_keypoints, self.opt.keypointnet_min_n_common_keypoints, -1):
                most_common_ids = sorted([x[0] for x in Counter(itertools.chain(*ids)).most_common(n_common_keypoints)])
                # prune keypoints
                pruned_keypoints = {}
                for name, (sample_keypoints, id) in keypoints.items():
                    if set(most_common_ids).issubset(id):
                        indices = [id.index(x) for x in most_common_ids]
                        new_keypoints = sample_keypoints[indices]
                        pruned_keypoints[name] = new_keypoints
                if len(pruned_keypoints) / len(keypoints) > self.opt.keypointnet_min_samples:
                    success = True
                    break
            if not success:
                raise ValueError()
            keypoints = pruned_keypoints
        else:
            keypoints = {k: v[0] for k, v in keypoints.items()}

        return keypoints

    def _get_shapenet_id_to_model_id(self):
        df = pandas.read_csv(self.opt.split_file)
        return {k: v for k, v in zip(df.id, df.modelId)}

    def _load_test_pairs(self):
        with open(self.opt.test_pairs_file, 'r') as f:
            lines = f.read().splitlines()
        names = []
        partners = []
        for line in lines:
            name = line.split(' ')[0]
            partner = line.split(' ')[1]
            names += [name]
            partners += [partner]
        return names, partners

    def _load_ROCA_annos(self):
        self.width = 480
        self.height = 360
        instances_file = open(os.path.join(self.opt.roca_dir, 'Dataset', 'scan2cad_instances_val.json'))
        inst_json = json.load(instances_file)
        instances_file.close()
        print(self.opt.roca_category_id)
        self.annos = [inst_json['annotations'][i] for i in range(len(inst_json['annotations'])) if
                      inst_json['annotations'][i]['category_id'] == int(self.opt.roca_category_id)]
        self.images = inst_json['images']
        self.target_points, self.ids, self.scales = self.get_target_points()
        self.n_samples = self.target_points.shape[0]
        print("Number of targets: " + str(self.n_samples))

    def get_target_points(self):
        x, y = np.meshgrid(list(range(self.width)), list(range(self.height)))
        target_points = []
        ids = []
        scales = []
        for anno in self.annos:
            intrinsics = anno['intrinsics']
            q = anno['q']
            s = anno['s']
            t = anno['t']
            #R_mat = self.quaternion_rotation_matrix(q)
            image_list = [self.images[i] for i in range(len(self.images)) if self.images[i]['id'] == anno['image_id']]
            assert len(image_list) == 1
            image = image_list[0]
            image_file = image['file_name'][25:]
            # img = cv2.imread(os.path.join(self.data_root, 'Images/tasks/scannet_frames_25k', image_file))

            depth_file = image_file.replace('color', 'depth')
            depth_file = depth_file.replace('jpg', 'png')
            depth = cv2.imread(os.path.join(self.opt.roca_dir, 'Rendering', depth_file), cv2.IMREAD_UNCHANGED)
            depth = depth / 1000.

            fx = intrinsics[0][0]
            fy = intrinsics[1][1]
            ux = intrinsics[0][2]
            uy = intrinsics[1][2]
            pc_x = depth * (x - ux) / fx
            pc_y = depth * (y - uy) / fy
            pc = np.concatenate((np.expand_dims(pc_x, 2), np.expand_dims(pc_y, 2), np.expand_dims(depth, 2)),
                                axis=2)  # pc in camera coordinate
            # mask
            mask_file = image_file.replace('color', 'instance')
            mask_file = mask_file.replace('jpg', 'png')
            mask = cv2.imread(os.path.join(self.opt.roca_dir, 'Rendering', mask_file), cv2.IMREAD_UNCHANGED)
            mask = mask == anno['alignment_id']
            pc_masked = pc[mask]
            # random down sampling to 2048
            #if pc_masked.shape[0] < 2048:
            #    continue
            #rdm_idx = np.random.choice(pc_masked.shape[0], 2048, replace=False)
            rdm_idx = np.random.choice(pc_masked.shape[0], int(2500*0.75), replace=True)
            # transform pc into object coord
            
            pc_final = pc_masked[rdm_idx]
            ##########################################################################################
            #pc_final = pc_final - t
            #pc_final = (np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]).dot(R_mat.T).dot(pc_final.T)).T
            # pc_final = ((R_mat.T).dot(pc_final.T)).T
            #pc_final = pc_final / s
            ##########################################################################################

            # model_id = anno['model']['id_cad']
            # mesh_path = self._get_mesh_path(model_id)
            # try:
            #     _, _, mesh_obj = read_mesh(mesh_path, return_mesh=True)
            #     points = resample_mesh(mesh_obj, 2048)
            #     points[:, :3], center, scale = self.normalize(points[:, :3])
            #     pc_final = (pc_final - center.numpy()) / scale
            #     # pc_final = points[:, :3].copy()
            # except:
            #     print('full mesh not found')
            #     continue

            pc_final, _, scale = self.normalize(pc_final)

            target_points.append(pc_final)
            ids.append(str(anno['image_id']) + '_' + str(anno['alignment_id']))
            scales.append(scale)
            # align_ids.append(anno['alignment_id'])
            # imgs.append(img)

        return np.stack(target_points, 0), ids, np.stack(scales, 0)

    def load_dataset(self):
        # import ipdb
        # ipdb.set_trace()
        dataset = {}
        if self.opt.data_type == 'shapenet':
            names = self._load_from_split_file()
        elif self.opt.data_type == 'keypointnet':
            keypoints = self._load_keypointnet()
            names = list(self._load_keypointnet_split(self.opt.split))
            dataset['keypoints'] = keypoints
        elif self.opt.data_type == 'shapenetseg':
            names = self._load_seg_split()
        elif self.opt.data_type == 'files':
            names = self._load_from_files()
        elif self.opt.data_type == 'ROCA':
            self._load_ROCA_annos()
            names = self._load_from_split_file()
        else:
            raise ValueError()

        if self.opt.keypoints_gt_source == 'keypointnet':
            keypoints = self._load_keypointnet()
            names = [x for x in names if x in keypoints]
            dataset['keypoints'] = keypoints

        assert len(names) > 0

        if self.opt.keypointnet_compatible:
            if self.opt.split == 'train':
                # remove keypointnet val and test from training
                val_split = self._load_keypointnet_split('val')
                test_split = self._load_keypointnet_split('test')
                names = set(names)
                names -= val_split
                names -= test_split

        # names = sorted(list(names))

        if self.opt.load_test_pairs:
            names, partners = self._load_test_pairs()
            dataset['partners'] = partners
        else:
            names = sorted(list(names))

        dataset['name'] = names

        return dataset

    def _get_pointcloud_path(self, name):
        return os.path.join(self.opt.points_dir, self.opt.category, name, "model.pts")

    def _get_partial_pointcloud_path(self, name):
        return os.path.join(self.opt.points_dir, self.opt.category, name, "partial_model.pts")

    def _get_partial_mask_path(self, name):
        return os.path.join(self.opt.points_dir, self.opt.category, name, "partial_mask.txt")

    def _get_center_path(self, name):
        return os.path.join(self.opt.points_dir, self.opt.category, name, "center.txt")

    def _get_scale_path(self, name):
        return os.path.join(self.opt.points_dir, self.opt.category, name, "scale.txt")

    def _get_mesh_path(self, name):
        if os.path.exists(os.path.join(self.opt.mesh_dir, self.opt.category, name, "model.obj")):
            return os.path.join(self.opt.mesh_dir, self.opt.category, name, "model.obj")
        elif int(self.opt.category) == 2871439:
            if os.path.exists(os.path.join(self.opt.mesh_dir, '02933112', name, "model.obj")):
                return os.path.join(self.opt.mesh_dir, '02933112', name, "model.obj")
            elif os.path.exists(os.path.join(self.opt.mesh_dir, '03337140', name, "model.obj")):
                return os.path.join(self.opt.mesh_dir, '03337140', name, "model.obj")
            elif os.path.exists(os.path.join(self.opt.mesh_dir, '02871439', name, "model.obj")):
                return os.path.join(self.opt.mesh_dir, '02871439', name, "model.obj")
            else:
                raise NameError
        else:
            raise NameError

    def _get_keypoints_path(self, name):
        return os.path.join(self.opt.keypoints_dir, self.opt.category, name, "keypoints.txt")

    def _get_seg_points_path(self, name):
        return os.path.join(self.opt.segmentations_dir, self.opt.category, 'points', name + '.pts')

    def _get_seg_labels_path(self, name):
        return os.path.join(self.opt.segmentations_dir, self.opt.category, 'points_label', name + '.seg')

    def _read_keypointnet_keypoints(self, name):
        keypoints = torch.from_numpy(self.dataset['keypoints'][name]).float()

        # fix axis
        keypoints = keypoints[:, [2, 1, 0]] * torch.FloatTensor([[-1, 1, 1]])

        # compensate for their normalization
        # load associated point cloud
        pcd_path = os.path.join(self.opt.keypointnet_dir, 'pcds', self.opt.category, name + '.pcd')
        points = read_pcd(pcd_path)
        points = torch.from_numpy(points).float()
        points = points[:, [2, 1, 0]] * torch.FloatTensor([[-1, 1, 1]])
        _, center, scale = self.normalize(points)
        keypoints = (keypoints - center) / scale

        return keypoints, center[0], scale[0]

    def _read_txt_keypoints(self, name):
        keypoints = read_keypoints(self._get_keypoints_path(name))
        keypoints = torch.from_numpy(keypoints).float()
        return keypoints

    def get_item(self, index):
        return self.get_item_by_name(self.dataset['name'][index])

    def get_item_by_name(self, name, sample_mesh=False, load_mesh=True, load_partial=False, random_rts=False):
        if load_mesh or sample_mesh:
            mesh_path = self._get_mesh_path(name)
            V_mesh, F_mesh, mesh_obj = read_mesh(mesh_path, return_mesh=True)

        if not sample_mesh:

            point_file = self._get_pointcloud_path(name)
            if load_partial:
                partial_point_file = self._get_partial_pointcloud_path(name)
                partial_mask_file = self._get_partial_mask_path(name)
                generate_point_file = not os.path.exists(point_file) or not os.path.exists(partial_point_file)
            else:
                generate_point_file = not os.path.exists(point_file)
        else:
            generate_point_file = False

        # DEBUG
        # generate_point_file = True
        
        ###   sample mesh to get pc
        if sample_mesh or generate_point_file:
            if load_partial:
                keep_ratio = self.opt.partial_ratio
                num_point = self.opt.num_point
                points = resample_mesh(mesh_obj, num_point)
                points[:, :3], center, scale = self.normalize(points[:, :3])
                
                ### get part points
                #partial_num_point = self.opt.num_point // keep_ratio
                partial_num_point = int(self.opt.num_point * 0.5)
                
                partial_shape, partial_mask = generate_partial_point_slice(points[:, :3], partial_num_point)
                partial_mask = torch.from_numpy(partial_mask)
                partial_points = points[partial_mask]
                #keep_ratio = np.random.uniform(low=self.opt.min_partial_ratio,high=self.opt.max_partial_ratio)\
                #    if self.opt.split == 'train' else self.opt.test_partial_ratio
                #num_point = self.opt.num_point / keep_ratio
                #num_point = int(num_point) + 1
                #points = resample_mesh(mesh_obj,num_point)
                #points[:,:3],center,scale = self.normalize(points[:,:3])
                
                #partial_num_point = int(keep_ratio * num_point)  if self.opt.split == 'train' else self.opt.num_point
                #partial_shape,partial_mask = generate_partial_point_slice(points[:,:3], partial_num_point)
                #partial_mask = torch.from_numpy(partial_mask)
                #partial_points = points[partial_mask]
                #partial_points, _ = sample_points(partial_points, self.opt.num_point)
                
                if self.partial_pc_dual:
                    partial_shape_dual, partial_mask_dual = generate_partial_point_slice(points[:, :3], partial_num_point)
                    partial_mask_dual = torch.from_numpy(partial_mask_dual)
                    partial_points_dual = points[partial_mask]
                    
                #points, _ = sample_points(points,self.opt.num_point)

            else:
                points = resample_mesh(mesh_obj, self.opt.num_point)
                points[:, :3], center, scale = self.normalize(points[:, :3])
        else:
            # load points sampled from a mesh
            points = np.loadtxt(self._get_pointcloud_path(name), dtype=np.float32)
            points = torch.from_numpy(points).float()
            center = np.loadtxt(self._get_center_path(name), dtype=np.float32)
            scale = np.loadtxt(self._get_scale_path(name), dtype=np.float32)

            # points[:, :3], center, scale = self.normalize(points[:, :3])

            if load_partial:
                partial_points = np.loadtxt(self._get_partial_pointcloud_path(name), dtype=np.float32)
                partial_mask = np.loadtxt(self._get_partial_mask_path(name), dtype=int)
                partial_points = torch.from_numpy(partial_points).float()
                partial_mask = torch.from_numpy(partial_mask)
                if self.partial_pc_dual:
                    partial_points_dual = np.loadtxt(self._get_partial_pointcloud_path(name), dtype=np.float32)
                    partial_mask_dual = np.loadtxt(self._get_partial_mask_path(name), dtype=int)
                    partial_points_dual = torch.from_numpy(partial_points_dual).float()
                    partial_mask_dual = torch.from_numpy(partial_mask_dual)
        
        ### save point info
        if generate_point_file:
            os.makedirs(os.path.dirname(point_file), exist_ok=True)
            center_file = self._get_center_path(name)
            np.savetxt(center_file, center)
            scale_file = self._get_scale_path(name)
            np.savetxt(scale_file, scale)
            np.savetxt(point_file, points)
            if load_partial:
                np.savetxt(partial_point_file, partial_points)
                np.savetxt(self._get_partial_mask_path(name), partial_mask)
                if self.partial_pc_dual:
                    np.savetxt(partial_point_file, partial_points_dual)
                    np.savetxt(self._get_partial_mask_path(name), partial_mask_dual)

        points = points.clone()

        normals = points[:, 3:6].clone()
        label = points[:, -1].clone()
        shape = points[:, :3].clone()
        
        ### data augment
        #  add random rts if specified
        if random_rts:
            R_full_gt = torch.from_numpy(Rotation.random().as_matrix()).type(torch.float32)
            t_full_gt = (torch.rand(3, 1) - 0.5) * 2.0 * self.opt.max_t_aug  # t is set to max 1.0
            s_full_gt = (torch.rand(1) - 1.0 / self.opt.max_s_aug) * (self.opt.max_s_aug - 1. / self.opt.max_s_aug)
            shape = s_full_gt * torch.matmul(R_full_gt, shape.transpose(1, 0)) + t_full_gt
            shape = shape.transpose(1, 0)
            ### ???
            normals = R_full_gt @ normals.transpose(1, 0)
            normals = normals.transpose(1, 0)
            
            if load_partial:
                R_partial_gt = torch.from_numpy(Rotation.random().as_matrix()).type(torch.float32)
                t_partial_gt = (torch.rand(3, 1) - 0.5) * 2.0 * self.opt.max_t_aug  # t is set to max 1.0
                s_partial_gt = (torch.rand(1) - 1.0 / self.opt.max_s_aug) * (self.opt.max_s_aug - 1. / self.opt.max_s_aug)
                partial_points[:, :3] = (s_partial_gt *
                                         torch.matmul(R_partial_gt, partial_points[:, :3].transpose(1, 0))
                                         + t_partial_gt).transpose(1, 0)
                partial_points[:, 3:6] = (R_partial_gt @ partial_points[:, 3:6].transpose(1, 0)).transpose(1, 0)

        result = {'shape': shape, 'normals': normals, 'label': label, 'cat': self.opt.category, 'file': name}
        
        ### save rts in result
        if random_rts:
            result['r_full_gt'] = R_full_gt.clone()
            result['t_full_gt'] = t_full_gt.clone()
            result['s_full_gt'] = s_full_gt.clone()
        
        ### save part info in result
        if load_partial:
            result['partial_shape'] = partial_points[:, :3].clone()
            result['partial_normal'] = partial_points[:, 3:6].clone()
            result['partial_label'] = partial_points[:, -1].clone()
            result['partial_mask'] = partial_mask.clone()
            if random_rts:
                result['r_partial_gt'] = R_partial_gt.clone()
                result['t_partial_gt'] = t_partial_gt.clone()
                result['s_partial_gt'] = s_partial_gt.clone()
            if self.partial_pc_dual:
                result['partial_shape_dual'] = partial_points_dual[:, :3].clone()
                result['partial_normal_dual'] = partial_points_dual[:, 3:6].clone()

        if load_mesh:
            V_mesh = V_mesh[:, :3]
            F_mesh = F_mesh[:, :3]
            V_mesh = (V_mesh - center) / scale
            result.update({'mesh': V_mesh, 'face': F_mesh, 'mesh_obj': mesh_obj, 'name': name})

        if not torch.is_tensor(scale):
            scale = torch.from_numpy(scale).float()
        if not torch.is_tensor(center):
            center = torch.from_numpy(center).float()
        result.update({'scale': scale.reshape([1, 1]), 'center': center.reshape([1, 3])})

        # load labels
        if self.opt.keypoints_gt_source == 'keypointnet':
            keypoints_gt, keypoints_gt_center, keypoints_gt_scale = self._read_keypointnet_keypoints(name)
            result['keypoints_gt'] = keypoints_gt
            result['keypoints_gt_center'] = keypoints_gt_center
            result['keypoints_gt_scale'] = keypoints_gt_scale

        if self.opt.data_type == 'shapenetseg':
            assert self.opt.segmentations_dir is not None
            seg_points = np.loadtxt(self._get_seg_points_path(name)).astype(np.float32)
            seg_labels = np.loadtxt(self._get_seg_labels_path(name)).astype(np.int32)
            seg_points = torch.from_numpy(seg_points)
            seg_labels = torch.from_numpy(seg_labels)
            seg_points = (seg_points - center) / scale
            result.update({'seg_labels': seg_labels, 'seg_points': seg_points})

        return result

    def get_sample(self, index):
        # this function is only used while training!!
        index_2 = np.random.randint(self.get_real_length())

        if self.opt.fixed_source_index is not None:
            index = self.opt.fixed_source_index

        if self.opt.fixed_target_index is not None:
            index_2 = self.opt.fixed_target_index

        name = self.dataset['name'][index]
        if self.opt.load_cages_test_pairs or self.opt.load_test_pairs:
            name_2 = self.dataset['partners'][index]
        else:
            name_2 = self.dataset['name'][index_2]

        sample_mesh = self.opt.sample_mesh or self.opt.points_dir is None
        source_data = self.get_item_by_name(name, load_mesh=self.opt.load_mesh, sample_mesh=sample_mesh)
        target_data = self.get_item_by_name(name_2, load_mesh=self.opt.load_mesh, sample_mesh=sample_mesh,
                                            load_partial=self.partial_pc, random_rts=self.opt.random_rts_aug)

        result = {'source_' + k: v for k, v in source_data.items()}
        result.update({'target_' + k: v for k, v in target_data.items()})
        result.update({'source_index': index})
        result.update({'target_index': index_2})

        # import ipdb
        # ipdb.set_trace()

        return result

    def __getitem__(self, index):
        if self.load_source:
            sample_mesh = self.opt.sample_mesh or self.opt.points_dir is None
            source_data = self.get_item_by_name(self.source_names[index], load_mesh=self.opt.load_mesh,
                                                sample_mesh=sample_mesh)
            result = {'source_' + k: v for k, v in source_data.items()}
            return result
        elif self.opt.split != 'train':
            if self.opt.data_type != 'ROCA':
                sample_mesh = self.opt.sample_mesh or self.opt.points_dir is None
                target_data = self.get_item_by_name(self.dataset['name'][index], load_mesh=self.opt.load_mesh,
                                                    sample_mesh=sample_mesh, load_partial=self.partial_pc)
                result = {'target_' + k: v for k, v in target_data.items()}
                return result
            else:
                points = torch.tensor(self.target_points[index], dtype=torch.float32)
                result = {'target_partial_shape': points, 'target_shape': points, 'target_cat': self.opt.category,
                          'target_file': self.ids[index], 'target_scale': self.scales[index]}
                return result
        else:
            for _ in range(10):
                index = index % self.get_real_length()
                try:
                    return self.get_sample(index)
                except Exception as e:
                    warnings.warn(f"Error loading sample {index}: " + ''.join(
                        traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)))
                    ipdb.set_trace()
                    index += 1

    @classmethod
    def collate(cls, batch):
        batched = {}
        elem = batch[0]
        for key in elem:
            if key in DO_NOT_BATCH:
                batched[key] = [e[key] for e in batch]
            else:
                try:
                    batched[key] = torch.utils.data.dataloader.default_collate([e[key] for e in batch])
                except Exception as e:
                    print(e)
                    print(key)
                    ipdb.set_trace()
                    print()

        return batched

    @staticmethod
    def uncollate(batched):
        for k, v in batched.items():
            if isinstance(v, torch.Tensor):
                batched[k] = v.cuda()
            elif isinstance(v, collections.abc.Sequence):
                if isinstance(v[0], torch.Tensor):
                    batched[k] = [e.cuda() for e in v]
        return batched

    def get_real_length(self):
        return len(self.dataset['name'])

    def __len__(self):
        if self.load_source:
            return len(self.source_names)
        if self.opt.fixed_target_index is not None:
            return 1000
        else:
            if self.opt.data_type == 'ROCA':
                return self.n_samples
            else:
                return self.get_real_length() * self.opt.multiply




