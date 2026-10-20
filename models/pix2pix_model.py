# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
import models.networks as networks
import util.util as util
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

try:
    from torch.cuda.amp import autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class Pix2PixModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))
        self.physics_loss = None
        self.physics_active = False
        if opt.isTrain:
            import math
            if not math.isfinite(getattr(opt, 'lambda_phys', 0.0)) or getattr(opt, 'lambda_phys', 0.0) < 0:
                raise ValueError('lambda_phys must be finite and nonnegative')
            if getattr(opt, 'physics_only', False) and getattr(opt, 'lambda_phys', 0.0) <= 0:
                raise ValueError('physics_only requires lambda_phys > 0')
            if getattr(opt, 'lambda_phys', 0.0) > 0:
                from models.physics import build_physics_loss
                self.physics_loss = build_physics_loss(opt, self.net['netCorr'].adaptive_model_seg)
                self.physics_active = opt.physics_start_epoch <= 1
        # set loss functions
        if opt.isTrain and not getattr(opt, 'physics_only', False):
            # vgg network
            self.vggnet_fix = networks.architecture.VGG19_feature_color_torchversion(vgg_normal_correct=opt.vgg_normal_correct, ic=3)
            self.vggnet_fix.load_state_dict(torch.load('vgg/vgg19_conv.pth'))

            self.vggnet_fix.eval()
            for param in self.vggnet_fix.parameters():
                param.requires_grad = False
            self.vggnet_fix.to(self.opt.gpu_ids[0])
            # contextual loss
            self.contextual_forward_loss = networks.ContextualLoss_forward(opt)
            # GAN loss
            self.criterionGAN = networks.GANLoss(opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            # L1 loss
            self.criterionFeat = torch.nn.L1Loss()
            # L2 loss
            self.MSE_loss = torch.nn.MSELoss()
            # setting which layer is used in the perceptual loss
            if opt.which_perceptual == '5_2':
                self.perceptual_layer = -1
            elif opt.which_perceptual == '4_2':
                self.perceptual_layer = -2

    def forward(self, data, mode, GforD=None):
        input_label, input_semantics, real_image, self_ref, ref_image, ref_label, ref_semantics = self.preprocess_input(data, )
        generated_out = {}

        if mode == 'generator':
            if getattr(self.opt, 'physics_only', False):
                if not self.physics_active:
                    raise ValueError('physics_only requires an active physics epoch')
                with autocast(enabled=self.opt.amp):
                    corr = self.net['netCorr'](ref_image, real_image, input_semantics, ref_semantics)
                    generated_out = {**corr, 'fake_image': self.net['netG'](input_semantics, warp_out=corr['warp_out'])}
                g_loss = {}
            else:
                g_loss, generated_out = self.compute_generator_loss(input_label, \
                                    input_semantics, real_image, ref_label, \
                                    ref_semantics, ref_image, self_ref)
            out = {}
            if self.physics_loss is not None:
                existing = sum(g_loss.values()).mean() if g_loss else generated_out['fake_image'].new_zeros(())
                raw = existing.new_zeros(())
                if self.physics_active:
                    raw = self.physics_loss(generated_out['fake_image'].float(),
                                            real_image, data.get('physics_amplitude'))
                    g_loss['physics'] = self.opt.lambda_phys * raw
                total = existing + self.opt.lambda_phys * raw
                out['physics_metrics'] = {'L_phys': raw.detach(), 'G_total_existing': existing.detach(),
                                          'G_total_new': total.detach()}
            out['fake_image'] = generated_out['fake_image']
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            out['adaptive_feature_seg'] = None if 'adaptive_feature_seg' not in generated_out else generated_out['adaptive_feature_seg']
            out['adaptive_feature_img'] = None if 'adaptive_feature_img' not in generated_out else generated_out['adaptive_feature_img']
            out['warp_cycle'] = None if 'warp_cycle' not in generated_out else generated_out['warp_cycle']
            return g_loss, out

        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, \
                                        real_image, GforD, label=input_label)
            return d_loss

        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(input_semantics, ref_semantics=ref_semantics, \
                                        ref_image=ref_image, self_ref=self_ref, \
                                        real_image=None)
            out['input_semantics'] = input_semantics
            out['ref_semantics'] = ref_semantics
            return out

        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2
        optimizer_G = torch.optim.Adam(itertools.chain(self.net['netG'].parameters(), \
                    self.net['netCorr'].parameters()), lr=G_lr, betas=(beta1, beta2), eps=1e-3)
        optimizer_D = torch.optim.Adam(itertools.chain(self.net['netD'].parameters()), \
                    lr=D_lr, betas=(beta1, beta2))
        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netD'], 'D', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netD'] = networks.define_D(opt) if opt.isTrain else None
        net['netCorr'] = networks.define_Corr(opt)
        if not opt.isTrain or opt.continue_train:
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            if opt.isTrain:
                net['netD'] = util.load_network(net['netD'], 'D', opt.which_epoch, opt)
        return net

    def preprocess_input(self, data):
        if self.use_gpu():
            for k in data.keys():
                try:
                    data[k] = data[k].cuda()
                except:
                    continue
        label = data['label'][:,:3,:,:].float()
        label_ref = data['label_ref'][:,:3,:,:].float()
        input_semantics = data['label'].float()
        ref_semantics = data['label_ref'].float()
        image = data['image']
        ref = data['ref']
        self_ref = data['self_ref']
        return label, input_semantics, image, self_ref, ref, label_ref, ref_semantics

    def get_ctx_loss(self, source, target):
        contextual_style5_1 = torch.mean(self.contextual_forward_loss(source[-1], target[-1].detach())) * 8
        contextual_style4_1 = torch.mean(self.contextual_forward_loss(source[-2], target[-2].detach())) * 4
        contextual_style3_1 = torch.mean(self.contextual_forward_loss(F.avg_pool2d(source[-3], 2), F.avg_pool2d(target[-3].detach(), 2))) * 2
        return contextual_style5_1 + contextual_style4_1 + contextual_style3_1

    def compute_generator_loss(self, input_label, input_semantics, real_image, ref_label=None, ref_semantics=None, ref_image=None, self_ref=None):
        G_losses = {}
        generate_out = self.generate_fake(input_semantics, real_image, ref_semantics=ref_semantics, ref_image=ref_image, self_ref=self_ref)
        generate_out['fake_image'] = generate_out['fake_image'].float()
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        sample_weights = self_ref/(sum(self_ref)+1e-5)
        sample_weights = sample_weights.view(-1, 1, 1, 1)
        """domain align"""
        if 'loss_novgg_featpair' in generate_out and generate_out['loss_novgg_featpair'] is not None:
            G_losses['no_vgg_feat'] = generate_out['loss_novgg_featpair']
        """warping cycle"""
        if self.opt.weight_warp_cycle > 0:
            warp_cycle = generate_out['warp_cycle']
            scale_factor = ref_image.size()[-1] // warp_cycle.size()[-1]
            ref = F.avg_pool2d(ref_image, scale_factor, stride=scale_factor)
            G_losses['G_warp_cycle'] = F.l1_loss(warp_cycle, ref) * self.opt.weight_warp_cycle
        """warping loss"""
        if self.opt.weight_warp_self > 0:
            """512x512"""
            warp1, warp2, warp3, warp4 = generate_out['warp_out']
            G_losses['G_warp_self'] = \
                torch.mean(F.l1_loss(warp4, real_image, reduction='none') * sample_weights) * self.opt.weight_warp_self * 1.0 + \
                torch.mean(F.l1_loss(warp3, F.avg_pool2d(real_image, 2, stride=2), reduction='none') * sample_weights) * self.opt.weight_warp_self * 1.0 + \
                torch.mean(F.l1_loss(warp2, F.avg_pool2d(real_image, 4, stride=4), reduction='none') * sample_weights) * self.opt.weight_warp_self * 1.0 + \
                torch.mean(F.l1_loss(warp1, F.avg_pool2d(real_image, 8, stride=8), reduction='none') * sample_weights) * self.opt.weight_warp_self * 1.0
        """gan loss"""
        pred_fake, pred_real = self.discriminate(input_semantics, generate_out['fake_image'], real_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False) * self.opt.weight_gan
        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = 0.0
            for i in range(num_D):
                # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs): 
                    # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) + self.MSE_loss(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.weight_ganFeat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss
        """feature matching loss"""
        fake_features = self.vggnet_fix(generate_out['fake_image'], ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        loss = 0
        for i in range(len(generate_out['real_features'])):
            loss += weights[i] * util.weighted_l1_loss(fake_features[i], generate_out['real_features'][i].detach(), sample_weights)
        G_losses['fm'] = loss * self.opt.weight_vgg * self.opt.weight_fm_ratio
        """perceptual loss"""
        feat_loss = util.mse_loss(fake_features[self.perceptual_layer], generate_out['real_features'][self.perceptual_layer].detach())
        G_losses['perc'] = feat_loss * self.opt.weight_perceptual
        """contextual loss"""
        G_losses['contextual'] = self.get_ctx_loss(fake_features, generate_out['ref_features']) * self.opt.weight_vgg * self.opt.weight_contextual
        return G_losses, generate_out

    def compute_discriminator_loss(self, input_semantics, real_image, GforD, label=None):
        D_losses = {}
        with torch.no_grad():
            fake_image = GforD['fake_image'].detach()
            fake_image.requires_grad_()
        pred_fake, pred_real = self.discriminate(input_semantics, fake_image, real_image)
        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True) * self.opt.weight_gan
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True) * self.opt.weight_gan
        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.net['netE'](real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, ref_semantics=None, ref_image=None, self_ref=None):
        generate_out = {}
        generate_out['ref_features'] = self.vggnet_fix(ref_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        generate_out['real_features'] = self.vggnet_fix(real_image, ['r12', 'r22', 'r32', 'r42', 'r52'], preprocess=True)
        with autocast(enabled=self.opt.amp):
            corr_out = self.net['netCorr'](ref_image, real_image, input_semantics, ref_semantics)
            generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=corr_out['warp_out'])
            # generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=warp_out)
        generate_out = {**generate_out, **corr_out}
        return generate_out

    def inference(self, input_semantics, ref_semantics=None, ref_image=None, self_ref=None, real_image=None):
        generate_out = {}
        with autocast(enabled=self.opt.amp):
            corr_out = self.net['netCorr'](ref_image, None, input_semantics, ref_semantics)
            generate_out['fake_image'] = self.net['netG'](input_semantics, warp_out=corr_out['warp_out'])
        generate_out = {**generate_out, **corr_out}
        return generate_out


    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = torch.cat([input_semantics, fake_image], dim=1)
        real_concat = torch.cat([input_semantics, real_image], dim=1)
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)
        with autocast(enabled=self.opt.amp):
            discriminator_out = self.net['netD'](fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)
        return pred_fake, pred_real

    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0
