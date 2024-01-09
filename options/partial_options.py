from options.base_options import BaseOptions


class PartialOptions(BaseOptions):
    def __init__(self):
        super(PartialOptions, self).__init__()
        self.initialized = False

    def initialize(self, parser):
        #############################################################################
        #                        network settings                                   #
        #############################################################################
        parser.add_argument('--partial_model_name', type=str, default='partial_shape_model')
        parser.add_argument('--partial_sample_hypos', type=int, default=5)

        #############################################################################
        #                        network settings                                   #
        #############################################################################
        parser.add_argument('--partial_add_noise', type=bool, default=False)
        parser.add_argument('--partial_fps', type=bool, default=False)
        parser.add_argument('--partial_remove_knn', type=bool, default=False)
        parser.add_argument('--partial_resample', type=bool, default=False)
        parser.add_argument('--partial_randRT', type=bool, default=False)
        parser.add_argument('--partial_dual', type=bool, default=False)
        parser.add_argument('--partial_downsample', type=bool, default=False)

        parser.add_argument('--partial_which_rot_metric', type=str, default='cosine',
                            help='loss selection for augmented rotation loss')
        parser.add_argument('--partial_which_ortho_metric', type=str, default='MSE',
                            help='loss selection for orthogonal loss')

        parser.add_argument('--partial_weight_ortho', type=float, default=1.0, help='weight of orthogonal loss')
        parser.add_argument('--partial_weight_recon1', type=float, default=10., help='weight of reconstruction loss')
        parser.add_argument('--partial_weight_recon2', type=float, default=1.,
                            help='weight of reconstruction loss2 (see code)')
        parser.add_argument('--partial_noise_amp', type=float, default=0.025,
                            help='noise level')
        parser.add_argument('--partial_weight_noise_T', type=float, default=1.0, help='scale noise for translation branch')
        parser.add_argument('--partial_weight_noise_R', type=float, default=1.0, help='scale noise for rotation branch')
        parser.add_argument('--partial_min_num_fps_points', type=int, default=300, help='min range for fps')
        parser.add_argument('--partial_max_num_fps_points', type=int, default=500, help='max range for fps')
        parser.add_argument('--partial_weight_fps_T', type=float, default=1.0, help='FPS weight for translation branch')
        parser.add_argument('--partial_weight_fps_R', type=float, default=1.0, help='FPS weight for rotation branch')
        parser.add_argument('--partial_remove_knn_points', type=int, default=100,
                            help='KNN removal points for translation branch')
        parser.add_argument('--partial_weight_part_T', type=float, default=1.0,
                            help='KNN removal weight for translation branch')
        parser.add_argument('--partial_weight_part_R', type=float, default=1.0,
                            help='KNN removal weight for rotation branch')

        parser.add_argument('--partial_weight_sample_T', type=float, default=1.0,
                            help='Resample weight for translation branch')
        parser.add_argument('--partial_weight_sample_R', type=float, default=1.0,
                            help='Resample weight for rotation branch')

        parser.add_argument('--partial_weight_dual_R', type=float, default=1.0,
                            help='Resample weight for translation branch')
        parser.add_argument('--partial_weight_dual_T', type=float, default=1.0,
                            help='Resample weight for rotation branch')

        # Can rot loss:
        parser.add_argument('--partial_apply_can_rot', type=int, default=0, help='Use can shape as augmentation')
        parser.add_argument('--partial_weight_can_T', type=float, default=1.0,
                            help='Canonical aug of weight for translation branch')
        parser.add_argument('--partial_weight_can_R', type=float, default=1.0,
                            help='Canonical aug weight for rotation branch')

        parser.add_argument('--partial_weight_rand_RT', type=float, default=1.0,
                            help='rand rt aug of weight for translation branch')

        parser.add_argument('--partial_detached_recon_2', default=False, type=bool,
                            help='Detach original rotated point cloud for loss')
        parser.add_argument('--partial_detach_aug_loss', default=False, type=bool,
                            help='Detach non-augmented rotation matrix for aug losses')
        parser.add_argument('--partial_add_ortho_aug', default=False, type=bool,
                            help='Apply orthogonal loss on augmented rotation')
        ###############################################################################
        parser.add_argument('--only_train_partial_model', default=True, type=bool)
        parser.add_argument('--partial_batch_size', default=16, type=int)
        parser.add_argument('--partial_lr', default=0.001, type=float)
        parser.add_argument('--partial_total_epochs', default=100, type=int)
        parser.add_argument('--partial_first_decay', type=int, default=72, help='')
        parser.add_argument('--partial_second_decay', type=int, default=92, help='')
        parser.add_argument('--partial_loaded_npoints', type=int, default=2500, help='')
        parser.add_argument('--partial_resample_npoints', type=int, default=1250, help='')

        parser.add_argument('--partial_resume', default=False, type=bool)
        parser.add_argument('--partial_resume_path', default='logs/model_19.pth', type=str)
        parser.add_argument('--partial_resume_point', type=int, default=19, help='')
        ###############################################################################
        parser.add_argument('--partial_log_intervals', type=int, default=10)
        parser.add_argument('--partial_save_intervals', type=int, default=20)

        self.initialized = True
        return parser
