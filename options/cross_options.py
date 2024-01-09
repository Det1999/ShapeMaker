from options.base_options import BaseOptions


class CrossOptions(BaseOptions):
    def __init__(self):
        super(CrossOptions, self).__init__()
        self.initialized = False

    def initialize(self, parser):
        #############################################################################
        #                        network settings                                   #
        #############################################################################
        parser.add_argument('--cross_deform_dim_input', type=int, default=1020 * 2)
        parser.add_argument('--cross_seg_class', type=int, default=12)
        parser.add_argument('--cross_seg_dim_input', type=int, default=1020 * 2)
        parser.add_argument('--cross_return_seg_class', type=bool, default=True)
        parser.add_argument('--cross_return_keypoint_vote', type=bool, default=True)
        parser.add_argument('--cross_keypoint_vote_dim_input', type=int, default=1020 * 2)
        parser.add_argument('--cross_keypoint_pred_dim_input', type=int, default=1020)

        parser.add_argument('--cross_seg_flag', type=bool, default=True)
        parser.add_argument('--cross_feat_flag', type=bool, default=True)
        parser.add_argument('--cross_recon_flag', type=bool, default=True)
        parser.add_argument('--cross_pd_flag', type=bool, default=True)
        parser.add_argument('--cross_kp_flag', type=bool, default=True)
        parser.add_argument('--cross_deform_flag', type=bool, default=True)
        parser.add_argument('--cross_kp_seg_flag', type=bool, default=True)
        parser.add_argument('--cross_kp_recon_flag', type=bool, default=True)

        parser.add_argument('--cross_overall_full', type=float, default=10.0, help='weight of full loss')
        parser.add_argument('--cross_overall_partial', type=float, default=10.0, help='weight of partial loss')
        parser.add_argument('--cross_weight_seg', type=float, default=1.0, help='weight of seg consistency loss')
        parser.add_argument('--cross_weight_feature', type=float, default=1.0, help='weight of feat consistency loss')
        parser.add_argument('--cross_weight_recon', type=float, default=1.0, help='weight of recon consistency loss')
        parser.add_argument('--cross_weight_pd', type=float, default=1.0, help='weight of recon consistency loss')
        parser.add_argument('--cross_weight_kp', type=float, default=1.0, help='weight of kp consistency loss')
        parser.add_argument('--cross_weight_deform', type=float, default=1.0, help='weight of recon consistency loss')
        parser.add_argument('--cross_weight_kp_seg', type=float, default=1.0, help='weight of recon consistency loss')
        parser.add_argument('--cross_weight_kp_recon', type=float, default=0.25, help='weight of recon consistency loss')
        parser.add_argument('--cross_weight_R_and_D', type=float, default=1.0,
                            help='weight of r and d consistency loss')
        parser.add_argument('--cross_source_weight_rd', type=float, default=1.0, help='weight of r and d source loss')
        ########################################################################################################
        parser.add_argument('--cross_R_and_D_average_error', type=float, default=0.025,
                            help='weight of recon consistency loss')
        ###############################################################################
        #                        other settings                                       #
        ###############################################################################
        parser.add_argument('--cross_batch_size', type=int, default=2)
        parser.add_argument('--cross_total_epochs', default=200, type=int)
        parser.add_argument('--cross_lr', default=0.001 / 5., type=float)
        parser.add_argument('--cross_first_decay', type=int, default=140, help='')
        parser.add_argument('--cross_second_decay', type=int, default=180, help='')
        ###############################################################################
        parser.add_argument('--cross_log_intervals', type=int, default=10)
        parser.add_argument('--cross_save_intervals', type=int, default=20)

        return parser
