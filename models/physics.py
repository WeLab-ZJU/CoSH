"""Differentiable port of SH_resolution_角谱传输聚焦.py (active Fresnel branch).

Shared simulation parameters are explicit; no camera intensity calibration is used.
FFT shifts act on spatial dimensions only, never on batch/channel dimensions.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def load_config(path):
    if not path:
        raise ValueError('Physics requires --physics_config with dataset calibration')
    with open(path, encoding='utf-8') as stream:
        cfg = json.load(stream)
    if cfg.get('target_mode') != 'propagated_gt':
        raise ValueError('Physics target_mode must be propagated_gt; camera intensity targets are not supported')
    required = ('shape', 'wavelength_m', 'pixel_size_m', 'focal_length_m',
                'num_lenses', 'phase_to_radians', 'phase_sampling', 'amplitude',
                'pupil', 'intensity_mode',
                'simulation_gain', 'calibration_note')
    for key in required:
        if key not in cfg or cfg[key] is None:
            raise ValueError('Missing physics calibration: ' + key)
    if not cfg['calibration_note']:
        raise ValueError('Document the dataset calibration in calibration_note')
    cfg = copy.deepcopy(cfg)
    root = Path(path).resolve().parent
    for key in ('amplitude', 'pupil'):
        if cfg[key]['kind'] == 'map':
            cfg[key]['path'] = str(root / cfg[key]['path'])
    return cfg


def read_map(path):
    path = Path(path)
    if path.suffix.lower() == '.npy':
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == '.mat':
        from scipy.io import loadmat
        arrays = [v for k, v in loadmat(path).items() if not k.startswith('__')]
        if len(arrays) != 1:
            raise ValueError('Map MAT must contain exactly one array: ' + str(path))
        array = arrays[0]
    else:
        import tifffile
        array = tifffile.imread(path)
    array = np.asarray(array)
    if array.ndim != 2 or np.iscomplexobj(array) or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError('Expected finite nonnegative 2D amplitude/pupil map: ' + str(path))
    return torch.as_tensor(array.copy(), dtype=torch.float64)[None, None]


class SHForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.shape = tuple(cfg['shape'])
        if len(self.shape) != 2 or any(type(n) is not int or n < 2 for n in self.shape):
            raise ValueError('shape must be [height, width] on the optical/detector grid')
        ny, nx = self.shape
        nl = cfg['num_lenses']
        if type(nl) is not int or nl < 1 or nl > min(ny, nx):
            raise ValueError('Invalid microlens count')
        edge_policy = cfg.get('lens_edge_policy', 'reject')
        if edge_policy not in ('reject', 'source_zero_phase'):
            raise ValueError('Unknown lens_edge_policy')
        if (ny % nl or nx % nl) and edge_policy == 'reject':
            raise ValueError('Nondivisible lens grid; explicitly confirm source_zero_phase edge policy')
        for key in ('wavelength_m', 'pixel_size_m', 'focal_length_m', 'phase_to_radians',
                    'simulation_gain'):
            if not math.isfinite(cfg[key]) or cfg[key] <= 0:
                raise ValueError(key + ' must be positive and finite')
        if cfg['phase_sampling'] not in ('native', 'full_extent_bilinear'):
            raise ValueError('Unsupported phase sampling convention')
        if cfg['intensity_mode'] not in ('calibrated', 'source_peak'):
            raise ValueError('intensity_mode must be calibrated or source_peak')
        dx, wl, z = cfg['pixel_size_m'], cfg['wavelength_m'], cfg['focal_length_m']
        x = (torch.arange(nx // nl, dtype=torch.float64) - (nx // nl - 1) / 2) * dx
        y = (torch.arange(ny // nl, dtype=torch.float64) - (ny // nl - 1) / 2) * dx
        lens = math.pi / (wl * z) * (y[:, None].square() + x[None, :].square())
        lens_array = torch.zeros(ny, nx, dtype=torch.float64)
        lens_array[:(ny // nl) * nl, :(nx // nl) * nl] = lens.repeat(nl, nl)
        self.register_buffer('lens_phase', lens_array[None, None])
        # Original uses four-times zero padding and crops back to the same pixels.
        self.padding = ((3 * nx) // 2, 3 * nx - (3 * nx) // 2,
                        (3 * ny) // 2, 3 * ny - (3 * ny) // 2)
        fx = torch.fft.fftfreq(4 * nx, d=dx, dtype=torch.float64)
        fy = torch.fft.fftfreq(4 * ny, d=dx, dtype=torch.float64)
        angle = -math.pi * wl * z * (fy[:, None].square() + fx[None, :].square())
        # Keep the source global propagation phase for numerical verification.
        h = torch.exp(1j * torch.tensor(2 * math.pi * z / wl, dtype=torch.float64)) * torch.exp(1j * angle)
        self.register_buffer('transfer', h[None, None])
        for name in ('amplitude', 'pupil'):
            spec = cfg[name]
            kind = spec['kind']
            if kind == 'map':
                value = read_map(spec['path'])
            elif kind == 'per_sample' and name == 'amplitude':
                value = torch.empty(0, dtype=torch.float64)
            elif kind == 'ones':
                if not spec.get('justification'):
                    raise ValueError('Explicit justification required for uniform ' + name)
                value = torch.ones(1, 1, ny, nx, dtype=torch.float64)
            elif kind == 'source_gaussian' and name == 'amplitude':
                w0 = spec['w0_m']
                if not math.isfinite(w0) or w0 <= 0:
                    raise ValueError('w0_m must be positive')
                x = torch.arange(-nx // 2, nx // 2, dtype=torch.float64) * dx
                y = torch.arange(-ny // 2, ny // 2, dtype=torch.float64) * dx
                # Source calls this intensity, but multiplies the FIELD by it.
                value = torch.exp(-(y[:, None].square() + x[None, :].square()) / w0**2)[None, None]
            else:
                raise ValueError('Unsupported ' + name + ' kind: ' + kind)
            if value.numel() and tuple(value.shape[-2:]) != self.shape:
                raise ValueError(name + ' map must match the optical grid; no implicit resize')
            self.register_buffer(name, value)

    def forward(self, phase, amplitude=None):
        if phase.ndim != 4 or phase.shape[1] != 1 or not phase.is_floating_point():
            raise ValueError('phase must be real floating Bx1xHxW')
        # FFT and exponentials must not run in half precision under outer AMP.
        phase = phase.to(dtype=torch.float64 if phase.dtype == torch.float64 else torch.float32)
        if not torch.isfinite(phase).all():
            raise ValueError('Nonfinite predicted phase')
        if tuple(phase.shape[-2:]) != self.shape:
            if self.cfg['phase_sampling'] != 'full_extent_bilinear':
                raise ValueError('Phase grid differs; explicit physical sampling convention required')
            phase = F.interpolate(phase, size=self.shape, mode='bilinear', align_corners=False)
        amp = amplitude if amplitude is not None else self.amplitude
        if amp.numel() == 0:
            raise ValueError('Per-sample amplitude is required')
        if amp.ndim != 4 or amp.shape[1:] != (1, *self.shape) or amp.shape[0] not in (1, phase.shape[0]):
            raise ValueError('Amplitude must be 1x1xHxW or Bx1xHxW on optical grid')
        amp = amp.to(phase)
        if not torch.isfinite(amp).all() or (amp < 0).any():
            raise ValueError('Invalid amplitude')
        pupil = self.pupil.to(phase)
        if not ((amp * pupil).sum(dim=(-2, -1)) > 0).all():
            raise ValueError('Empty illuminated pupil')
        angle = phase * self.cfg['phase_to_radians'] - self.lens_phase.to(phase)
        field = amp * pupil * torch.exp(1j * angle)
        field = F.pad(field, self.padding)
        dims = (-2, -1)
        spectrum = torch.fft.fft2(torch.fft.ifftshift(field, dim=dims))
        out = torch.fft.fftshift(torch.fft.ifft2(spectrum * self.transfer.to(field)), dim=dims)
        left, _, top, _ = self.padding
        intensity = out[..., top:top+self.shape[0], left:left+self.shape[1]].abs().square()
        if self.cfg['intensity_mode'] == 'source_peak':
            # Explicit source-script convention, applied identically to both branches.
            intensity = intensity / intensity.amax(dim=dims, keepdim=True).clamp_min(torch.finfo(phase.dtype).tiny)
        intensity = intensity * self.cfg['simulation_gain']
        if not torch.isfinite(intensity).all():
            raise ValueError('Nonfinite simulated intensity; check phase/amplitude/calibration')
        return intensity


class PhysicsConsistencyLoss(nn.Module):
    """Compare SHForward(predicted phase) with SHForward(GT phase).

    Both arguments use the loader's phase/50 representation. Only the prediction
    branch is differentiable; the same optical module and amplitude serve both.
    Camera intensities are never accepted as the target.
    """
    def __init__(self, optical, extractor=None, layers=(0, 1, 2, 3), weights=(1, 1, 1, 1), mode='features',
                 feature_padding='reject'):
        super().__init__()
        if mode not in ('features', 'pixel_l1'):
            raise ValueError('Unknown physics loss mode')
        if mode == 'features' and extractor is None:
            raise ValueError('Feature physics loss needs a fixed SH feature extractor')
        if len(layers) != len(weights) or not layers or any(i not in range(4) for i in layers):
            raise ValueError('Select SH encoder layers 0..3 and matching weights')
        if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('Feature weights must be finite, nonnegative and not all zero')
        self.optical, self.extractor = optical, extractor
        self.layers, self.weights, self.mode = tuple(layers), tuple(weights), mode
        if feature_padding not in ('reject', 'replicate'):
            raise ValueError('Unknown feature padding policy')
        self.feature_pad = (0, (-optical.shape[1]) % 8, 0, (-optical.shape[0]) % 8)
        if mode == 'features' and any(self.feature_pad) and feature_padding == 'reject':
            raise ValueError('SH encoder needs multiples of 8; explicitly select physics_feature_padding=replicate')
        if extractor is not None:
            extractor.requires_grad_(False)
            extractor.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.extractor is not None:
            self.extractor.eval()
        return self

    def forward(self, phase, gt_phase, amplitude=None):
        if gt_phase.shape != phase.shape:
            raise ValueError('Predicted and GT phase must have identical Bx1xHxW grids')
        predicted = self.optical(phase, amplitude)
        with torch.no_grad():
            target = self.optical(gt_phase.detach().to(predicted), amplitude)
        if self.mode == 'pixel_l1':
            return F.l1_loss(predicted, target)
        if any(self.feature_pad):
            # Feature encoder shape adaptation only: no detector resize/shift.
            predicted = F.pad(predicted, self.feature_pad, mode='replicate')
            target = F.pad(target, self.feature_pad, mode='replicate')
        # The existing SH encoder takes (input, conditioning), both the SH image.
        source_features = self.extractor(predicted, predicted)
        with torch.no_grad():
            target_features = self.extractor(target, target)
        return sum(w * F.l1_loss(source_features[i], target_features[i])
                   for i, w in zip(self.layers, self.weights))


def build_physics_loss(opt, sh_encoder):
    cfg = load_config(opt.physics_config)
    extractor = None
    if opt.physics_loss_mode == 'features':
        if not opt.physics_feature_checkpoint:
            raise ValueError('Provide --physics_feature_checkpoint: trained Corr or SH encoder state dict')
        extractor = copy.deepcopy(sh_encoder)
        state = torch.load(opt.physics_feature_checkpoint, map_location='cpu', weights_only=True)
        prefix = 'adaptive_model_seg.'
        if any(k.startswith(prefix) for k in state):
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        extractor.load_state_dict(state, strict=True)
    return PhysicsConsistencyLoss(SHForward(cfg), extractor, opt.physics_layers,
                                  opt.physics_layer_weights, opt.physics_loss_mode,
                                  opt.physics_feature_padding)
