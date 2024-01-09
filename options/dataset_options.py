# developed by yan di
import argparse

import yaml


class ShapeNetV1_options:
    def __init__(self):
        self.initialized = False
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        ###############################################################################################
        #                                   common settings                                           #
        ###############################################################################################
        self.parser.add_argument('-c', '--config', required=False, default='configs/Chairs', help='config file path')
        self.parser.add_argument('-t', '--test_config', required=False, default='configs/Chairs',
                                 help='config file path')
        # basic parameters
        self.parser.add_argument("--name", required=False, type=str, default='random_test', help="experiment name")
        self.parser.add_argument("--dataset", type=str, default='shapes', help="dataset name")
        self.parser.add_argument("--num_point", type=int, help="number of input points", default=2500)
        self.parser.add_argument("--points_dir", type=str, help="points data root", default=None)

        self.parser.add_argument("--subdir", type=str, help="save to directory name", default="test")

        self.parser.add_argument("--n_keypoints", type=int, help="")
        self.parser.add_argument("--cage_size", type=float, default=1.4, help="")
        self.parser.add_argument("--print_options", action="store_true", help="")
        # training setup
        self.parser.add_argument("--phase", type=str, choices=["test", "train", "refine_deform", "train_retrieval"],
                                 default="train")
        self.parser.add_argument("--seed", type=int, default=0, help="")
        self.parser.add_argument("--normalization", type=str, choices=["batch", "instance", "none"], default="none")
        # dataset related options
        self.parser.add_argument("--mesh_dir", type=str, default='/data/wanggu/Storage/shapenet/shape_data', help="")
        self.parser.add_argument("--keypoints_dir", type=str, help="")

        self.parser.add_argument("--partial_pc", type=bool, default=True)
        self.parser.add_argument("--partial_pc_dual", type=bool, default=True)
        self.parser.add_argument("--partial_ratio", type=float, default=2)  # numpoint // partial_ratio
        self.parser.add_argument("--retrieval_full_shape", type=bool, default=False)
        self.parser.add_argument("--valid_threshold", type=int, default=32)
        self.parser.add_argument("--retrieval_startup_iter", type=int, default=1000)
        self.parser.add_argument("--retrieval_full_token", type=bool, default=False)
        self.parser.add_argument("--use_partial_retrieval", type=bool, default=False)
        self.parser.add_argument("--lambda_full_token_l1", type=float, default=1.0)

        # for ROCA
        self.parser.add_argument("--roca_dir", type=str, default='/home/ubuntu/newdisk/wcw/datasets/ROCA')

        ######################################################################################
        #                                dataset settings                                    #
        ######################################################################################
        # dataset control
        self.parser.add_argument('--category', required=False, default='03001627', type=str, help='')
        self.parser.add_argument('--segmentations_dir', type=str, default=None, help='')
        self.parser.add_argument('--seg_split_dir', type=str, default=None, help='')
        self.parser.add_argument('--keypointnet_dir', type=str, default=None, help='')
        self.parser.add_argument('--keypointnet_compatible', type=str, default=None, help='')
        self.parser.add_argument('--keypointnet_common_keypoints', action='store_true', help='')
        self.parser.add_argument("--keypointnet_min_n_common_keypoints", type=int, default=6, help="")
        self.parser.add_argument("--keypointnet_min_samples", type=float, default=0.8, help="")
        self.parser.add_argument('--keypoints_gt_source', type=str, default=None, help='')
        self.parser.add_argument('--data_type', type=str, default='shapenet', help='')
        self.parser.add_argument('--split_file', type=str, default='/home/yan/RED/I_red/data/shapenet_split/all.csv',
                                 help='')
        self.parser.add_argument('--split', type=str, default='train', help='')
        self.parser.add_argument("--fixed_source_index", type=int, default=None, help="")
        self.parser.add_argument("--fixed_target_index", type=int, default=None, help="")
        self.parser.add_argument('--normalize', type=str, default='unit_box', help='')
        self.parser.add_argument("--multiply", type=int, default=1, help="")
        self.parser.add_argument('--load_cages_test_pairs', action='store_true', help='')
        self.parser.add_argument('--load_test_pairs', action='store_true', help='')
        self.parser.add_argument('--load_mesh', action='store_true', help='')
        self.parser.add_argument('--sample_mesh', action='store_true', help='')
        self.parser.add_argument('--test_pairs_file', type=str, default=None, help='')
        self.parser.add_argument("--min_partial_ratio", type=float, default=0.25)
        self.parser.add_argument("--max_partial_ratio", type=float, default=0.9)
        self.parser.add_argument("--test_partial_ratio", type=float, default=0.5)

        self.parser.add_argument("--max_t_aug", type=float, default=0.0)
        self.parser.add_argument("--max_s_aug", type=float, default=2.0)
        self.parser.add_argument("--random_rts_aug", type=bool, default=False)
        # for ROCA
        # chair: 5; cabinet: 3; table: 8
        self.parser.add_argument('--roca_category_id', type=str, default=5)

        self.initialized = True

    def combine_configs(self, combine=True, file_path=None):
        self.initialize()
        opt = self.parser.parse_args()
        if combine is False:
            return opt
        with open(file_path, 'r') as yaml_file:
            cfg = yaml.safe_load(yaml_file)

        opt.name = cfg['name']
        opt.category = cfg['category']
        opt.split = cfg['split']
        opt.mesh_dir = cfg['mesh_dir']
        opt.split_file = cfg['split_file']
        opt.log_dir = cfg['log_dir']
        opt.n_keypoints = cfg['n_keypoints']
        return opt
