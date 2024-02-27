from charset_normalizer import logging
import tqdm
import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from flowDet_mmdet.utils.gmmDet_utils import summarise_performance
from flowDet_mmdet.utils.helper_utils import load_json_data, get_logger

# NFs impl
import normflows as nf
import larsflow as lf

class ActNorm_gpu(nf.flows.ActNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done > 0.0:
            assert self.s is not None and self.t is not None
            s_init = -torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = (
                -z.mean(dim=self.batch_dims, keepdim=True) * torch.exp(self.s)
            ).data
            self.data_dep_init_done = torch.tensor(1.0).cuda()
        return super().forward(z)

    def inverse(self, z):
        # first batch is used for initialization, c.f. batchnorm
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None
            s_init = torch.log(z.std(dim=self.batch_dims, keepdim=True) + 1e-6)
            self.s.data = s_init.data
            self.t.data = z.mean(dim=self.batch_dims, keepdim=True).data
            self.data_dep_init_done = torch.tensor(1.0).cuda()
        return super().inverse(z)


class ResampledGaussianClsOut(nf.distributions.BaseDistribution):
    """
    Resampled Gaussian factorized over second dimension,
    i.e. first non-batch dimension; can be class-conditional
    """
    def __init__(self, d, a, T, eps, num_classes, trainable=False, affine_shape=None):
        """
        Constructor
        :param shape: Shape of the variables (after mapped through the flows)
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param affine_shape: Shape of the affine layer serving as mean and
        standard deviation; if None, no affine transformation is applied
        :param num_classes: Number of classes in the class-conditional case;
        if None, the distribution is not class conditional
        """
        super().__init__()
        # Write parameters to object
        self.dim = d
        self.a = a
        self.T = T
        self.eps = eps
        self.n_cls = num_classes
        
        # Normalization constant
        self.register_buffer("Z", -torch.ones(self.n_cls))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.dim))
            self.log_scale = nn.Parameter(torch.zeros(1, self.dim))
        else:
            self.register_buffer("loc", torch.zeros(1, self.dim))
            self.register_buffer("log_scale", torch.zeros(1, self.dim))

        # Affine transformation
        self.affine_shape = affine_shape
        if self.affine_shape is None:
            self.affine_transform = None
        elif self.class_cond:
            self.affine_transform = nf.flows.CCAffineConst(self.affine_shape,
                                                           self.n_cls)
        else:
            self.affine_transform = nf.flows.AffineConstFlow(self.affine_shape)

    def log_prob(self, z, y=None, return_uncond_logp=False):
        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            eps = torch.randn_like(z)
            acc_ = self.a(eps)
            Z_batch = torch.mean(acc_, dim=0)
            if torch.any(self.Z < 0.):
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z

        # Get values of a
        batch_size = z.size(0)
        eps_z = (z - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = - 0.5 * self.dim * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1) \
                      - torch.sum(0.5 * torch.pow(eps_z, 2), 1)
        acc = self.a(eps_z)
         # bz * cls
        acc = acc.view(batch_size, self.n_cls)
        alpha = (1 - Z) ** (self.T - 1)
        all_base_log_pxy = torch.log((1 - alpha) * acc / Z + alpha) # bz*cls
        base_log_px = torch.logsumexp(all_base_log_pxy, dim=1) # torch.sum(torch.log) # bz*1
        log_px = log_p_gauss + base_log_px
        if y is not None:
            if return_uncond_logp:
                base_log_pxy = torch.sum(torch.log_softmax(all_base_log_pxy, 1) * y, dim=1)
                log_pxy = log_p_gauss + base_log_pxy
                return log_pxy, log_px
            else:
                # Get normalization constant
                # acc_cond = torch.sum(acc * y, dim=1)
                # Z_cond = torch.sum(y * Z, dim=1)
                # alpha_cond = (1 - Z_cond) ** (self.T - 1)
                # log_pxy = torch.log((1 - alpha_cond) * acc_cond / Z_cond + alpha_cond)
                base_log_pxy = torch.sum(all_base_log_pxy * y, dim=1)
                log_pxy = log_p_gauss + base_log_pxy
                return log_pxy

        else:
            return log_px

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.dim), dtype=dtype, device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_, dim=0)
                self.Z = self.Z + Z_batch.detach() / num_batches

class ResampledGaussianClsGMM(nf.distributions.BaseDistribution):
    """
    Resampled Gaussian factorized over second dimension,
    i.e. first non-batch dimension; can be class-conditional
    """
    def __init__(self, d, a, T, eps, num_classes, loc=None, scale=None, weights=None, trainable=False):
        """
        Constructor
        :param shape: Shape of the variables (after mapped through the flows)
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        standard deviation; if None, no affine transformation is applied
        :param num_classes: Number of classes in the class-conditional case;
        if None, the distribution is not class conditional
        """
        super().__init__()
        # Write parameters to object
        self.dim = d
        self.a = a
        self.T = T
        self.eps = eps
        self.n_cls = num_classes
        self.register_buffer("Z", -torch.ones(self.n_cls))
        self.init_proposal_func(num_classes, loc, scale, weights, trainable)
        
    def init_proposal_func(self, n_modes, loc, scale, weights, trainable):
        if loc is None:
            loc = np.random.randn(n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights))) 
        
        # Normalization constant
        # self.register_buffer("Z", -torch.ones(self.n_cls))
        # if trainable:
        #     self.loc = nn.Parameter(torch.zeros(1, self.dim))
        #     self.log_scale = nn.Parameter(torch.zeros(1, self.dim))
        # else:
        #     self.register_buffer("loc", torch.zeros(1, self.dim))
        #     self.register_buffer("log_scale", torch.zeros(1, self.dim))

    def update_Z_train(self, z):
        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            eps = torch.randn_like(z)  # bz * dim
            acc_ = self.a(eps) # bz * cls 
            Z_batch = torch.mean(acc_, dim=0) # cls
            if torch.any(self.Z < 0.):
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        return Z

    def log_prob(self, z, y=None, return_uncond_logp=False):
        # bz * dim -> bz * cls * dim
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        # bz * cls 
        log_p_gauss = -0.5 * self.dim * np.log(2 * np.pi)\
                      - 0.5 * torch.sum(torch.pow(eps, 2), 2)\
                      - torch.sum(self.log_scale, 2)

        # Update normalization constant
        Z = self.update_Z_train(z) # cls

        # Get acceptance rate
        all_acc = self.a(eps) # bz * cls * cls
        acc = torch.diagonal(all_acc, dim1=1, dim2=2)  # bz * cls 
        # acc = acc.view(batch_size, self.n_cls) # bz * cls
        alpha = (1 - Z.unsqueeze(0)) ** (self.T - 1) # 1 * cls
        all_base_log_pxy = torch.log((1 - alpha) * acc / Z.unsqueeze(0) + alpha) # bz * cls
        log_px = torch.logsumexp(all_base_log_pxy+log_p_gauss, dim=1) 
        # log_px = torch.sum(all_base_log_pxy, dim=1) # bz
        if y is not None:
            if return_uncond_logp:
                log_pxy = torch.sum(torch.log_softmax(all_base_log_pxy+log_p_gauss, 1) * y, dim=1)
                return log_pxy, log_px
            else:
                log_pxy = torch.sum((all_base_log_pxy + log_p_gauss) * y, dim=1)
                return log_pxy

        else:
            return log_px

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.dim), dtype=dtype, device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_, dim=0)
                self.Z = self.Z + Z_batch.detach() / num_batches

class ResampledGaussianClsGMM_sim(nf.distributions.BaseDistribution):
    """
    Resampled Gaussian factorized over second dimension,
    i.e. first non-batch dimension; can be class-conditional
    """
    def __init__(self, d, a, T, eps, num_classes, loc=None, scale=None, weights=None, trainable=False):
        """
        Constructor
        :param shape: Shape of the variables (after mapped through the flows)
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        standard deviation; if None, no affine transformation is applied
        :param num_classes: Number of classes in the class-conditional case;
        if None, the distribution is not class conditional
        """
        super().__init__()
        # Write parameters to object
        self.dim = d
        self.a = a
        self.T = T
        self.eps = eps
        self.n_cls = num_classes
        self.register_buffer("Z", -torch.ones(self.n_cls))
        self.init_proposal_func(num_classes, loc, scale, weights, trainable)
        
    def init_proposal_func(self, n_modes, loc, scale, weights, trainable):
        if loc is None:
            loc = np.random.randn(n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights))) 

    def update_Z_train(self, z):
        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            eps = torch.randn_like(z)  # bz * dim
            acc_ = self.a(eps) # bz * 1 
            Z_batch = torch.mean(acc_, dim=0) # 1
            if torch.any(self.Z < 0.):
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        return Z

    def log_prob(self, z, y=None, return_uncond_logp=False):
        # Get batch size (bz)
        # batch_size = z.size(0)

        # bz * dim -> bz * cls * dim
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        # bz * cls 
        log_p_gauss = -0.5 * self.dim * np.log(2 * np.pi)\
                      - 0.5 * torch.sum(torch.pow(eps, 2), 2)\
                      - torch.sum(self.log_scale, 2)

        # Update normalization constant
        Z = self.update_Z_train(z) # 1

        # Get acceptance rate
        acc = self.a(eps) #  bz * cls * dim -> bz * cls * 1
        acc = acc.squeeze(-1) # view(batch_size, self.n_cls) #  bz * cls * 1 -> bz * cls
        alpha = (1 - Z.unsqueeze(0)) ** (self.T - 1) # 1 * 1
        all_base_log_pxy = torch.log((1 - alpha) * acc / Z.unsqueeze(0) + alpha) # bz * cls
        log_px = torch.logsumexp(all_base_log_pxy+log_p_gauss, dim=1) # bz * 1
        # log_px = torch.sum(all_base_log_pxy, dim=1) # bz
        if y is not None:
            if return_uncond_logp:
                log_pxy = torch.sum(torch.log_softmax(all_base_log_pxy+log_p_gauss, 1) * y, dim=1)
                return log_pxy, log_px
            else:
                log_pxy = torch.sum((all_base_log_pxy+log_p_gauss) * y, dim=1)
                return log_pxy

        else:
            return log_px

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.dim), dtype=dtype, device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_, dim=0)
                self.Z = self.Z + Z_batch.detach() / num_batches

class ResampledGaussianClsIn(nf.distributions.BaseDistribution):
    """
    Resampled Gaussian factorized over second dimension,
    """
    def __init__(self, d, a, T, eps, num_classes, trainable=False, affine_shape=None):
        """
        Constructor
        :param shape: Shape of the variables (after mapped through the flows)
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param affine_shape: Shape of the affine layer serving as mean and
        standard deviation; if None, no affine transformation is applied
        :param num_classes: Number of classes in the class-conditional case;
        if None, the distribution is not class conditional
        """
        super().__init__()
        # Write parameters to object
        self.dim = d
        self.a = a
        self.T = T
        self.eps = eps
        self.n_cls = num_classes
        
        # Normalization constant
        self.register_buffer("Z", -torch.ones(self.n_cls))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.dim))
            self.log_scale = nn.Parameter(torch.zeros(1, self.dim))
        else:
            self.register_buffer("loc", torch.zeros(1, self.dim))
            self.register_buffer("log_scale", torch.zeros(1, self.dim))

        # Affine transformation
        self.affine_shape = affine_shape
        if self.affine_shape is None:
            self.affine_transform = None
        elif self.class_cond:
            self.affine_transform = nf.flows.CCAffineConst(self.affine_shape,
                                                           self.n_cls)
        else:
            self.affine_transform = nf.flows.AffineConstFlow(self.affine_shape)

    def create_all_cls_onehot_input(self, eps):
        # eps shape: bz*dim
        batch_size = eps.shape[0]
        y_onehot_all_cls = torch.eye(self.n_cls).unsqueeze(0).cuda()
        y_onehot_all_cls = y_onehot_all_cls.repeat_interleave(batch_size, 0) # bz * cls * (dim_logits+cls)
        eps = eps.unsqueeze(1)
        eps = eps.repeat_interleave(self.n_cls,1)
        # bz * cls * (dim+cls)
        a_input = torch.cat((eps, y_onehot_all_cls), dim=2)
        return a_input

    def log_prob(self, z, y=None, return_uncond_logp=False):
        # bz 
        eps = (z - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = - 0.5 * self.dim * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1) \
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        log_p = log_p_gauss

        # Reverse affine transform
        if self.affine_transform is not None and y is not None:
            z, log_det = self.affine_transform.inverse(z, y)
            log_p = log_p + log_det

        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            bz = eps.shape[0]
            eps_ = torch.randn_like(z)
            eps_ = self.create_all_cls_onehot_input(eps_)
            eps_ = eps_.view(bz*self.n_cls, -1)
            acc_ = self.a(eps_)
            acc_ = acc_.view(bz, self.n_cls)
            Z_batch = torch.mean(acc_, dim=0)
            if torch.any(self.Z < 0.):
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z

        # Get values of a
        # y = nn.functional.one_hot(y.to(torch.int64), self.n_cls)
        # Get batch size, dtype, and device
        batch_size = z.size(0)
        a_input = self.create_all_cls_onehot_input(eps) # bz * cls * (dim+cls)
        a_input = a_input.view(batch_size*self.n_cls, -1) # (bz*cls) * (dim+cls)
        acc = self.a(a_input) # bz * cls
        acc = acc.view(batch_size, self.n_cls)
        if y is not None:
            if return_uncond_logp:
                alpha = (1 - Z) ** (self.T - 1)
                pxy_exp = (1 - alpha) * acc / Z + alpha # bz * cls
                log_p_a = torch.sum(torch.log(pxy_exp), dim=1)
                log_px = log_p + log_p_a
                log_pxy = torch.sum(torch.log_softmax(pxy_exp, 1) * y, dim=1)
                return log_pxy, log_px
            else:
                # Get normalization constant
                acc_cond = torch.sum(acc * y, dim=1)
                Z_cond = torch.sum(y * Z, dim=1)
                alpha_cond = (1 - Z_cond) ** (self.T - 1)
                log_p_xy = torch.log((1 - alpha_cond) * acc_cond / Z_cond + alpha_cond)
                log_p += log_p_xy
                return log_p

        else:
            # bz * cls
            alpha = (1 - Z) ** (self.T - 1)
            log_p_a = torch.sum(torch.log((1 - alpha) * acc / Z + alpha), dim=1)
            log_p += log_p_a
            return log_p

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            # Get dtype and device
            dtype = self.Z.dtype
            device = self.Z.device
            for i in range(num_batches):
                eps = torch.randn((num_samples, self.dim), dtype=dtype, device=device)
                bz = eps.shape[0]
                eps = self.create_all_cls_onehot_input(eps) # bz * cls * (dim+cls)
                eps = eps.view(bz*self.n_cls, -1) # (bz*cls) * (dim+cls)
                # bz * cls
                acc_ = self.a(eps)
                acc_ = acc_.view(bz, self.n_cls)
                Z_batch = torch.mean(acc_, dim=0)
                self.Z = self.Z + Z_batch.detach() / num_batches


class GaussianMixtureCls(nf.distributions.BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix and class conditional info
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None, trainable=True
    ):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model
          dim: Number of dimensions of each Gaussian
          loc: List of mean values
          scale: List of diagonals of the covariance matrices
          weights: List of mode probabilities
          trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_cls = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

    def forward(self, y=None, num_samples=1):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        if y is not None:
            mode_1h = y
        else:
            # Sample mode indices
            mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
            mode_1h = nn.functional.one_hot(mode, self.n_cls)
            mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(
            num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device
        )
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample

        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z, y=None, return_uncond_logp=False):
        # bz * num_cls * dim
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        # bz * num_cls 
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        # bz 
        if y is not None:
            log_pxy = torch.sum(log_p * y, 1)
            if return_uncond_logp:
                log_pxy = torch.sum(torch.log_softmax(- 0.5 * torch.sum(torch.pow(eps, 2), 2), 1) * y, 1)
                log_px = torch.logsumexp(log_p, 1)
                return log_pxy, log_px
            else:
                return log_pxy
        else:
            log_p = torch.logsumexp(log_p, 1)
            return log_p
            


def get_latest_checkpoint(dir_path, key=''):
    """
    Get path to latest checkpoint in directory
    :param dir_path: Path to directory to search for checkpoints
    :param key: Key which has to be in checkpoint name
    :return: Path to latest checkpoint
    """
    if not os.path.exists(dir_path):
        return None
    checkpoints = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, f)) and key in f and ".pt" in f]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return checkpoints[-1]

def setup_dataloader(res_path, 
                    batch_size, 
                    data_type="eval", 
                    feat="logits",
                    filter_dict=None,
                    log_file=None,
                    return_fn_bboxes=False,
                    shuffle=False
                    ):
    
    logger = get_logger("flowDet_mmdet.utils.functional_utils", log_file)
    logger.info(f"data_type: {data_type}")
    if data_type == "eval":
        testTypes, testLogits, Scores, Ious, testFeats, Filenames, bboxes = load_json_data(res_path, data_type="eval")
        if feat == "logits":
            test_dataset = TensorDataset(torch.Tensor(testLogits))
        elif feat == "msfeats":
            test_dataset = TensorDataset(torch.Tensor(testFeats))
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        logger.info(f"testTypes: {testTypes.shape}")
        logger.info(f"testLogits: {testLogits.shape}")
        logger.info(f"testFeats: {testFeats.shape}")
        if return_fn_bboxes:
            return dataloader, testTypes, Filenames, bboxes, Scores
        else:
            return dataloader, testTypes
        
    elif data_type == "train":
        trainLabels, trainLogits, trnScores, trnIoUs, trainFeats = load_json_data(res_path, data_type="train")
        if filter_dict is not None:
            #mask for high iou and high conf
            mask = (trnIoUs >= filter_dict['iouThresh'])*(trnScores >= filter_dict['scoreThresh'])
            trainLogits = trainLogits[mask]
            trainLabels = trainLabels[mask]
            if feat == "msfeats":
                trainFeats = trainFeats[mask]

        if feat == "logits":
            train_dataset = TensorDataset(torch.Tensor(trainLogits), torch.Tensor(trainLabels))
        elif feat == "msfeats":
            train_dataset = TensorDataset(torch.Tensor(trainFeats), torch.Tensor(trainLabels))
        logger.info(f"trainLogits: {trainLogits.shape}")
        logger.info(f"trainLabels: {trainLabels.shape}")
        if trainFeats is not None:
            logger.info(f"trainFeats: {trainFeats.shape}")
        
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

def init_flow_module_v2(params_dict, weights_folder=None, save_model_key=''):
    """
    flow implementation from https://github.com/VincentStimper/normalizing-flows
    """
    # num_classes = params_dict['num_class']
    input_dim = params_dict['input_dim']
    hidden_units = params_dict['hidden_dim']
    hidden_layers = params_dict['num_layers_st_net']
    K = params_dict['blocks']
    flow_type = params_dict['flow_type']
    dropout = None if 'dropout' not in params_dict \
        else params_dict['dropout']

    # Get parameters specific to flow type
    if flow_type == 'rnvp':
        scale = True if params_dict['coupling'] == 'affine' else False
        scale_map = 'exp' if not 'scale_map' in params_dict \
            else params_dict['scale_map']
    elif flow_type == 'residual':
        lipschitz_const = 0.9 if not 'lipschitz_const' in params_dict \
            else params_dict['lipschitz_const']
    elif flow_type == 'nsf_ar':
        num_bins = 8 if not 'num_bins' in params_dict \
            else params_dict['num_bins']
        if dropout is None:
            dropout = 0.
    else:
        raise NotImplementedError('The flow type ' + flow_type
                                    + ' is not yet implemented.')
    init_zeros = True if not 'init_zeros' in params_dict \
        else params_dict['init_zeros']

    # Set up base distribution
    if params_dict['prior_type'] == 'resampled':
        T = params_dict['base_T']
        eps = params_dict['base_eps']
        a_hl = params_dict['base_a_hidden_layers']
        a_hu = params_dict['base_a_hidden_units']
        a_drop = params_dict['base_dropout']
        init_zeros_a = params_dict['base_init_zeros']
        a = nf.nets.MLP([input_dim] + a_hl * [a_hu] + [1], output_fn="sigmoid",
                        init_zeros=init_zeros_a, dropout=a_drop)
        q0 = lf.distributions.ResampledGaussian(input_dim, a, T, eps,
                            trainable=params_dict['base_learn_mean_var'])
    elif params_dict['prior_type'] in ['resampled_cls', 'resampled_cls_ib', 'resampled_v2_cls', 'resampled_v2_cls_ib', 'resampled_v3_cls', 'resampled_v3_cls_ib']:
        num_cls = params_dict['num_cls']
        T = params_dict['base_T']
        eps = params_dict['base_eps']
        a_hl = params_dict['base_a_hidden_layers']
        a_hu = params_dict['base_a_hidden_units']
        a_drop = params_dict['base_dropout']
        init_zeros_a = params_dict['base_init_zeros']
        if "v2" in params_dict['prior_type']:
            n_modes = params_dict['base_n_modes']
            loc_scale = params_dict['base_loc_scale']
            loc = loc_scale * np.random.rand(n_modes, input_dim)
            trainable = params_dict['base_learn_mean_var']
            a = nf.nets.MLP([input_dim] + a_hl * [a_hu] + [num_cls], output_fn="sigmoid",
                            init_zeros=init_zeros_a, dropout=a_drop)
            q0 = ResampledGaussianClsGMM(input_dim, a, T, eps, num_classes=num_cls, loc=loc, trainable=trainable)
        elif "v3" in params_dict['prior_type']:
            n_modes = params_dict['base_n_modes']
            loc_scale = params_dict['base_loc_scale']
            loc = loc_scale * np.random.rand(n_modes, input_dim)
            trainable = params_dict['base_learn_mean_var']
            a = nf.nets.MLP([input_dim] + a_hl * [a_hu] + [1], output_fn="sigmoid",
                            init_zeros=init_zeros_a, dropout=a_drop)
            q0 = ResampledGaussianClsGMM_sim(input_dim, a, T, eps, num_classes=num_cls, loc=loc, trainable=trainable)
        else:
            a = nf.nets.MLP([input_dim] + a_hl * [a_hu] + [num_cls], output_fn="sigmoid",
                            init_zeros=init_zeros_a, dropout=a_drop)
            q0 = ResampledGaussianClsOut(input_dim, a, T, eps, num_classes=num_cls, trainable=params_dict['base_learn_mean_var'])
    elif params_dict['prior_type'] == 'gauss':
        q0 = nf.distributions.DiagGaussian(input_dim,
                            trainable=params_dict['base_learn_mean_var'])
    elif params_dict['prior_type'] == 'gmm':
        n_modes = params_dict['base_n_modes']
        loc_scale = params_dict['base_loc_scale']
        loc = loc_scale * np.random.rand(n_modes, input_dim)
        trainable = params_dict['base_learn_mean_var']
        q0 = nf.distributions.GaussianMixture(n_modes, input_dim, loc=loc,
                                                trainable=trainable)
    elif params_dict['prior_type'] in ['gmm_cls', 'gmm_cls_ib']:
        num_cls = params_dict['num_cls']
        loc_scale = params_dict['base_loc_scale']
        loc = loc_scale * np.random.rand(num_cls, input_dim)
        trainable = params_dict['base_learn_mean_var']
        q0 = GaussianMixtureCls(num_cls, input_dim, loc=loc, trainable=trainable)
    else:
        raise NotImplementedError('The base distribution ' + params_dict['prior_type']
                                    + ' is not implemented.')

    flows = []
    for _ in range(K):
        # Permutation
        if 'permutation' in params_dict:
            permutation = params_dict['permutation']
            if permutation == 'affine':
                flows += [nf.flows.InvertibleAffine(input_dim)]
            elif permutation == 'permute':
                flows += [nf.flows.Permute(input_dim, mode="shuffle")] # mode="shuffle" or "swap"
            elif permutation == 'lu_permute':
                flows += [nf.flows.LULinearPermute(input_dim)]

        # Transformation layer
        if flow_type == 'rnvp':
            # Coupling layer
            param_map = nf.nets.MLP([(input_dim + 1) // 2] + hidden_layers * [hidden_units]
                                    + [(input_dim // 2) * (2 if scale else 1)],
                                    init_zeros=init_zeros, dropout=dropout)
            flows += [nf.flows.AffineCouplingBlock(param_map, scale=scale,
                                                    scale_map=scale_map)]
        elif flow_type == 'residual':
            # Residual layer
            net = nf.nets.LipschitzMLP([input_dim] + [hidden_units] * hidden_layers + [input_dim],
                                        init_zeros=init_zeros, lipschitz_const=lipschitz_const)
            flows += [nf.flows.Residual(net, reduce_memory=True)]
        elif flow_type == 'nsf_ar':
            # Autoregressive Neural Spline Flow layer
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(input_dim, hidden_layers,
                                        hidden_units, num_bins=num_bins, dropout_probability=dropout)]

        # ActNorm
        if params_dict['actnorm']:
            flows += [ActNorm_gpu(input_dim)]
        # else:
        #     flows += [nf.flows.BatchNorm()]
    
    flow_module = lf.NormalizingFlow(q0, flows)
    if weights_folder is not None:
        latest_cp = get_latest_checkpoint(weights_folder, key=save_model_key)
        if latest_cp is not None:
            flow_module.load(latest_cp)

    return flow_module

def test_flow_NLL_v2(model, 
                    test_dataloader, 
                    testType, 
                    method_name="", 
                    prior_type="",
                    test_or_eval='eval',
                    return_nlls=False, 
                    log_file=None,
                    tp_fp_idxes=[0, 2]):
    test_nlls = []

    logger = get_logger("flowDet_mmdet.utils.functional_utils", log_file)

    model.eval()
    if test_or_eval == "test" and "resampled" in prior_type:
        logger.info("Estimating Z...")
        with torch.no_grad():
            model.q0.estimate_Z(num_samples=10**4, num_batches=10**4)
    for x in tqdm.tqdm(test_dataloader, total=int(len(test_dataloader.dataset)/test_dataloader.batch_size)):	
        x = x[0].to("cuda")
        with torch.no_grad():
            log_p = model.log_prob(x)
            log_p_np = log_p.cpu().detach().numpy()
        test_nlls.append(-log_p_np)

    test_nlls = np.concatenate(test_nlls)
    tpKnown_nll = test_nlls[testType == tp_fp_idxes[0]]
    fpUnknown_nll = test_nlls[testType == tp_fp_idxes[1]]
    id_nan_flag = np.isnan(tpKnown_nll) 
    ood_nan_flag = np.isnan(fpUnknown_nll) 
    tpKnown_nll = tpKnown_nll[np.logical_not(id_nan_flag)]
    fpUnknown_nll = fpUnknown_nll[np.logical_not(ood_nan_flag)]
    logger.info(f"tpKnown_nll after filtering out {np.sum(id_nan_flag)} Nans: {np.mean(tpKnown_nll):.3f}")
    logger.info(f"fpUnknown_nll after filtering out {np.sum(ood_nan_flag)} Nans: {np.mean(fpUnknown_nll):.3f}")

    #we want results in terms of AUROC, and TPR at 5%, 10% and 20% FPR
    fprRates = [0.05, 0.1, 0.2]
    scoreResults = summarise_performance(-tpKnown_nll, -fpUnknown_nll, fprRates=fprRates, printRes=True, methodName=method_name, log_file=log_file)
    if return_nlls:
        scoreResults["inData_nll"] = tpKnown_nll
        scoreResults["outData_nll"] = fpUnknown_nll
    return scoreResults
