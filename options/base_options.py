import yaml


class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        #############################################################################
        #                        basic settings                                   #
        #############################################################################
        parser.add_argument('-c', '--config', default="configs/Chairs.yaml" , required=False, help='config file path')
        parser.add_argument('-t', '--test_config', default="configs/test_full.yaml", required=False, help='config file path')

        parser.add_argument('--base_model_name', type=str, default='full_shape_model')
        parser.add_argument('--base_device', type=str, default='cuda')

        parser.add_argument('--base_encoder_n_knn', default=40, type=int,
                            help='Number of nearest neighbors to use [default: 40]')
        parser.add_argument('--base_encoder_which_norm_VNT', type=str, default='norm',
                            help='Normalization method of VNTLinear layer [default: norm]')
        parser.add_argument('--base_encoder_base_ch', default=64, type=int, help='base encoder_decoder channels')
        parser.add_argument('--base_encoder_nlatent', type=int, default=1020, help='')
        parser.add_argument('--base_encoder_pooling', type=str, default='mean', help='Pooling method [default: mean]',
                            choices=['mean', 'max'])
        parser.add_argument('--base_encoder_global_bias', type=float, default=0.0, help='')
        ###########################################################################################
        parser.add_argument('--base_decoder_name', default='point', type=str,
                            help='use which decoder')
        parser.add_argument('--base_decoder_npoint', default=1250, type=int,
                            help='number of points after sampling the point cloud')
        parser.add_argument('--base_decoder_patchDim', type=int, default=2,
                            help='Dimension of patch, relevant for atlasNet')
        parser.add_argument('--base_decoder_patchDeformDim', type=int, default=3,
                            help='Output dimension of atlas net decoder, relevant for atlasNet')
        parser.add_argument('--base_decoder_npatch', type=int, default=10,
                            help='number of patches, relevant for atlasNet')
        parser.add_argument('--base_decoder_nlatent', type=int, default=1020, help='')
        #############################################################################################
        parser.add_argument('--base_rot_net_name', default='simple', type=str,
                            help='use which rot net')
        parser.add_argument('--base_rot_nlatent', type=int, default=1020, help='')
        parser.add_argument('--base_rot_which_strict_rot', type=str, default='svd',
                            choices=['svd', 'gram_schmidt', 'None'],
                            help='Define rotation tansform, [default: None]')
        ##############################################################################################
        parser.add_argument('--base_normalize', default='unit_box', type=str)
        # dataset parameters
        parser.add_argument('--base_loaded_npoints', default=2500, type=int,
                            help='number of points in the loaded point cloud')
        parser.add_argument('--base_resample_npoints', default=1250, type=int,
                            help='number of points after sampling the point cloud')

        parser.add_argument('--base_rot', type=str, default='se3', help='Apply transformation to input point cloud')
        parser.add_argument('--base_se3_T', type=float, default=0.1, help='range of translation')

        ##################################dataset control###################################
        parser.add_argument('--base_data_type', default='shapenet', type=str)
        parser.add_argument('--base_segmentations_dir', type=str, default=None, help='')
        parser.add_argument('--base_seg_split_dir', type=str, default=None, help='')
        parser.add_argument('--base_keypointnet_dir', type=str, default=None, help='')
        parser.add_argument('--base_keypointnet_compatible', type=str, default=None, help='')
        parser.add_argument('--base_keypointnet_common_keypoints', action='store_true', help='')
        parser.add_argument("--base_keypointnet_min_n_common_keypoints", type=int, default=6, help="")
        parser.add_argument("--base_keypointnet_min_samples", type=float, default=0.8, help="")
        parser.add_argument('--base_keypoints_gt_source', type=str, default=None, help='')
        parser.add_argument('--base_load_test_pairs', type=bool, default=False, help='')
        parser.add_argument("--base_fixed_source_index", type=int, default=None, help="")
        parser.add_argument("--base_fixed_target_index", type=int, default=None, help="")
        parser.add_argument("--base_multiply", type=int, default=1, help="")
        parser.add_argument('--base_load_cages_test_pairs', type=bool, default=False, help='')
        parser.add_argument('--base_load_mesh', type=bool, default=False, help='')
        parser.add_argument('--base_sample_mesh', type=bool, default=False, help='')


        self.initialized = True
        return parser


def combine_flags(opt, config_path=''):
    with open(config_path, 'r') as yaml_file:
        cfg = yaml.safe_load(yaml_file)
    opt.name = cfg['name']
    opt.category = cfg['category']
    opt.split = cfg['split']
    opt.mesh_dir = cfg['mesh_dir']
    opt.points_dir = cfg['points_dir']
    opt.split_file = cfg['split_file']
    opt.log_dir = cfg['log_dir']
    opt.n_keypoints = cfg['n_keypoints']
    opt.cross_batch_size = cfg['cross_batch_size']
    opt.cross_total_epochs = cfg['cross_total_epochs']
    opt.cross_lr = cfg['cross_lr']
    return opt

