# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from options.base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=1000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        # for training
        parser.add_argument('--continue_train', action='store_true',default=True, help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default=29, help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--real_reference_probability', type=float, default=0.0, help='self-supervised training probability')
        parser.add_argument('--hard_reference_probability', type=float, default=0.0, help='hard reference training probability')
        # training loss weights
        parser.add_argument('--lambda_phys', type=float, default=0.0)
        parser.add_argument('--physics_config', type=str, default='')
        parser.add_argument('--physics_feature_checkpoint', type=str, default='')
        parser.add_argument('--physics_loss_mode', choices=('features', 'pixel_l1'), default='features')
        parser.add_argument('--physics_feature_padding', choices=('reject', 'replicate'), default='reject',
                            help='Explicit feature-only right/bottom padding for detector sizes not divisible by 8')
        parser.add_argument('--physics_layers', type=int, nargs='+', default=[0, 1, 2, 3])
        parser.add_argument('--physics_layer_weights', type=float, nargs='+', default=[1., 1., 1., 1.])
        parser.add_argument('--physics_start_epoch', type=int, default=1,
                            help='Enable only once the phase prediction stage is trained')
        parser.add_argument('--physics_only', action='store_true', help='Optimize only physics; skip D updates')
        parser.add_argument('--sh_image_dir', type=str, default='images')
        parser.add_argument('--physics_amplitude_dir', type=str, default='', help='Same-stem .npy amplitude maps')
        parser.add_argument('--weight_warp_self', type=float, default=0.0, help='push warp self to ref')
        parser.add_argument('--weight_warp_cycle', type=float, default=0.0, help='push warp cycle to ref')
        parser.add_argument('--weight_novgg_featpair', type=float, default=10.0, help='in no vgg setting, use pair feat loss in domain adaptation')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--weight_gan', type=float, default=10.0, help='weight of all loss in stage1')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--weight_ganFeat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--which_perceptual', type=str, default='4_2', help='relu5_2 or relu4_2')
        parser.add_argument('--weight_perceptual', type=float, default=0.001)
        parser.add_argument('--weight_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--weight_contextual', type=float, default=1.0, help='ctx loss weight')
        parser.add_argument('--weight_fm_ratio', type=float, default=1.0, help='vgg fm loss weight comp with ctx loss')
        self.isTrain = True
        return parser
