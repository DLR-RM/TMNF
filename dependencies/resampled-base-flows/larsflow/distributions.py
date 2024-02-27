import torch
import torch.nn as nn
import numpy as np

import normflows as nf

class ResampledGaussian(nf.distributions.BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix,
    resampled according to a acceptance probability determined by a neural network,
    see arXiv 1810.11428
    """
    def __init__(self, d, a, T, eps, trainable=True, bs_factor=1):
        """
        Constructor
        :param d: Dimension of Gaussian distribution
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param bs_factor: Factor to increase the batch size during sampling
        """
        super().__init__()
        self.dim = d
        self.a = a
        self.T = T
        self.eps = eps
        self.bs_factor = bs_factor
        self.register_buffer("Z", torch.tensor(-1.))
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, self.dim))
            self.log_scale = nn.Parameter(torch.zeros(1, self.dim))
        else:
            self.register_buffer("loc", torch.zeros(1, self.dim))
            self.register_buffer("log_scale", torch.zeros(1, self.dim))

    def forward(self, num_samples=1):
        t = 0
        eps = torch.zeros(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T // self.bs_factor + 1):
            eps_ = torch.randn((num_samples * self.bs_factor, self.dim),
                               dtype=self.loc.dtype, device=self.loc.device)
            acc = self.a(eps_)
            if self.training or self.Z < 0.:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples * self.bs_factor
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    eps[s, :] = eps_[j, :]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p_gauss = - 0.5 * self.dim * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1)\
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        acc = self.a(eps)
        if self.training or self.Z < 0.:
            eps_ = torch.randn((num_samples, self.dim), dtype=self.loc.dtype, device=self.loc.device)
            Z_batch = torch.mean(self.a(eps_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z < 0.:
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
        return z, log_p

    def log_prob(self, z):
        eps = (z - self.loc) / torch.exp(self.log_scale)
        log_p_gauss = - 0.5 * self.dim * np.log(2 * np.pi) \
                      - torch.sum(self.log_scale, 1) \
                      - torch.sum(0.5 * torch.pow(eps, 2), 1)
        acc = self.a(eps)
        if self.training or self.Z < 0.:
            eps_ = torch.randn_like(z)
            Z_batch = torch.mean(self.a(eps_))
            if self.Z < 0.:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_gauss
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
                eps = torch.randn((num_samples, self.dim), dtype=dtype,
                                  device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_)
                self.Z = self.Z + Z_batch.detach() / num_batches


class ResampledDistribution(nf.distributions.BaseDistribution):
    """
    Resampling of a general distribution
    """
    def __init__(self, dist, a, T, eps, bs_factor=1):
        """
        Constructor
        :param dist: Distribution to be resampled
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param bs_factor: Factor to increase the batch size during sampling
        """
        super().__init__()
        self.dist = dist
        self.a = a
        self.T = T
        self.eps = eps
        self.bs_factor = bs_factor
        self.register_buffer("Z", torch.tensor(-1.))

    def forward(self, num_samples=1):
        t = 0
        z = None
        log_p_dist = None
        s = 0
        n = 0
        Z_sum = 0
        for i in range(self.T // self.bs_factor + 1):
            z_, log_prob_ = self.dist(num_samples * self.bs_factor)
            if i == 0:
                z = torch.zeros_like(z_[:num_samples])
                log_p_dist = torch.zeros_like(log_prob_[:num_samples])
            acc = self.a(z_)
            if self.training or self.Z < 0.:
                Z_sum = Z_sum + torch.sum(acc).detach()
                n = n + num_samples * self.bs_factor
            dec = torch.rand_like(acc) < acc
            for j, dec_ in enumerate(dec[:, 0]):
                if dec_ or t == self.T - 1:
                    z[s, ...] = z_[j, ...]
                    log_p_dist[s] = log_prob_[j]
                    s = s + 1
                    t = 0
                else:
                    t = t + 1
                if s == num_samples:
                    break
            if s == num_samples:
                break
        acc = self.a(z)
        if self.training or self.Z < 0.:
            z_, _ = self.dist(num_samples)
            Z_batch = torch.mean(self.a(z_))
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if self.Z < 0.:
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_dist
        return z, log_p

    def log_prob(self, z):
        log_p_dist = self.dist.log_prob(z)
        acc = self.a(z)
        if self.training or self.Z < 0.:
            z_, _ = self.dist(len(z))
            Z_batch = torch.mean(self.a(z_))
            if self.Z < 0.:
                self.Z = Z_batch.detach()
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_batch.detach()
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        alpha = (1 - Z) ** (self.T - 1)
        log_p = torch.log((1 - alpha) * acc[:, 0] / Z + alpha) + log_p_dist
        return log_p

    def estimate_Z(self, num_samples, num_batches=1):
        """
        Estimate Z via Monte Carlo sampling
        :param num_samples: Number of samples to draw per batch
        :param num_batches: Number of batches to draw
        """
        with torch.no_grad():
            self.Z = self.Z * 0.
            for i in range(num_batches):
                z, _ = self.dist(num_samples)
                acc_ = self.a(z)
                Z_batch = torch.mean(acc_)
                self.Z = self.Z + Z_batch.detach() / num_batches


class FactorizedResampledGaussian(nf.distributions.BaseDistribution):
    """
    Resampled Gaussian factorized over second dimension,
    i.e. first non-batch dimension; can be class-conditional
    """
    def __init__(self, shape, a, T, eps, affine_shape=None, flows=[],
                 group_dim=0, same_dist=True, num_classes=None, Z_samples=None):
        """
        Constructor
        :param shape: Shape of the variables (after mapped through the flows)
        :param a: Function returning the acceptance probability
        :param T: Maximum number of rejections
        :param eps: Discount factor in exponential average of Z
        :param affine_shape: Shape of the affine layer serving as mean and
        standard deviation; if None, no affine transformation is applied
        :param flows: Flows to be applied after sampling from base distribution
        :param group_dim: Int or list of ints; Dimension(s) to be used for group
        formation; dimension after batch dim is 0
        :param same_dist: Flag; if true, the distribution of each of the groups
        is the same
        :param num_classes: Number of classes in the class-conditional case;
        if None, the distribution is not class conditional
        """
        super().__init__()
        # Write parameters to object
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.sum_dim = list(range(1, self.n_dim + 1))
        self.a = a
        self.T = T
        self.eps = eps
        self.Z_samples = Z_samples
        self.flows = nn.ModuleList(flows)
        self.num_classes = num_classes
        self.class_cond = num_classes is not None
        if isinstance(group_dim, int):
            group_dim = [group_dim]
        self.group_dim = group_dim
        self.group_shape = []
        self.not_group_shape = []
        for i, s in enumerate(self.shape):
            if i in self.group_dim:
                self.group_shape += [s]
            else:
                self.not_group_shape += [s]
        self.not_group_sum_dim = list(range(1, len(self.not_group_shape) + 1))
        # Get permutation indizes to form groups
        self.perm = []
        for i in range(self.n_dim):
            if i in self.group_dim:
                self.perm = self.perm + [i + 1]
            else:
                self.perm = [i + 1] + self.perm
        self.perm = [0] + self.perm
        self.perm_inv = [0] * len(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i
        self.same_dist = same_dist
        self.not_group_prod = np.prod(self.not_group_shape, dtype=np.int)
        if same_dist:
            self.num_groups = 1
        else:
            self.num_groups = self.not_group_prod
        # Normalization constant
        if self.class_cond:
            self.register_buffer("Z", -torch.ones(self.num_classes
                                                  * self.num_groups))
        else:
            self.register_buffer("Z", -torch.ones(self.num_groups))
        # Affine transformation
        self.affine_shape = affine_shape
        if self.affine_shape is None:
            self.affine_transform = None
        elif self.class_cond:
            self.affine_transform = nf.flows.CCAffineConst(self.affine_shape,
                                                           self.num_classes)
        else:
            self.affine_transform = nf.flows.AffineConstFlow(self.affine_shape)

    def forward(self, num_samples=1, y=None):
        # Get dtype and device
        dtype = self.Z.dtype
        device = self.Z.device
        # Prepare one hot encoding or sample y if needed
        if self.class_cond:
            if y is not None:
                num_samples = len(y)
            else:
                y = torch.randint(self.num_classes, (num_samples,), device=device)
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=dtype, device=device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        # Draw samples
        eps = torch.zeros(num_samples * self.not_group_prod, *self.group_shape,
                          dtype=dtype, device=device)
        sampled = torch.zeros(num_samples * self.not_group_prod, dtype=torch.bool,
                              device=device)
        n = 0
        Z_sum = 0
        for i in range(self.T):
            eps_ = torch.randn(num_samples * self.not_group_prod, *self.group_shape,
                               dtype=dtype, device=device)
            # Get a
            acc = self.a(eps_)
            # Z update
            if self.training or torch.any(self.Z < 0.):
                Z_sum = Z_sum + torch.sum(acc, dim=0).detach()
                n = n + num_samples * self.not_group_prod
            # Get relevant part of a
            if self.class_cond:
                acc = acc.view(num_samples, -1, self.num_classes, self.num_groups)
                acc = torch.sum(acc * y[:, None, :, None], dim=2)
            else:
                acc = acc.view(num_samples, -1, self.num_groups)
            if self.same_dist:
                acc = acc.view(num_samples, -1)
            else:
                acc = torch.diagonal(acc, dim1=1, dim2=2).contiguous()
            acc = acc.view(-1)
            # Make decision about acceptance
            dec = torch.rand_like(acc) < acc
            update = torch.logical_and(torch.logical_not(sampled), dec)
            sampled = torch.logical_or(sampled, dec)
            update_ = update.type(dtype).view(num_samples * self.not_group_prod,
                                              *([1] * len(self.group_dim)))
            # Update tensor with final samples
            eps += update_ * eps_
            if torch.all(sampled):
                break
        # Update all random variables which have not been sampled yet
        update = torch.logical_not(sampled)
        update_ = update.type(dtype).view(num_samples * self.not_group_prod,
                                          *([1] * len(self.group_dim)))
        eps += update_ * eps_
        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            Z_samples = num_samples if self.Z_samples is None else self.Z_samples
            eps_ = torch.randn(Z_samples, *self.group_shape, dtype=dtype,
                               device=device)
            acc_ = self.a(eps_)
            Z_batch = torch.mean(acc_, dim=0)
            Z_ = (Z_sum + Z_batch.detach() * num_samples) / (n + num_samples)
            if torch.any(self.Z < 0.):
                self.Z = Z_
            else:
                self.Z = (1 - self.eps) * self.Z + self.eps * Z_
            Z = Z_batch - Z_batch.detach() + self.Z
        else:
            Z = self.Z
        # Get log p
        # Get values of a
        acc = self.a(eps)
        if self.class_cond:
            acc = acc.view(num_samples, -1, self.num_classes, self.num_groups)
            acc = torch.sum(acc * y[:, None, :, None], dim=2)
        else:
            acc = acc.view(num_samples, -1, self.num_groups)
        if self.same_dist:
            acc = acc.view(num_samples, -1)
        else:
            acc = torch.diagonal(acc, dim1=1, dim2=2)
        acc = acc.view(num_samples, *self.not_group_shape)
        # Get normalization constant
        if self.class_cond:
            Z = y @ Z.view(self.num_classes, self.num_groups)
        if self.same_dist:
            Z = Z.view(-1, *([1] * len(self.not_group_shape)))
        else:
            Z = Z.view(-1, *self.not_group_shape)
        alpha = (1 - Z) ** (self.T - 1)
        if len(self.not_group_sum_dim) == 0:
            log_p_a = torch.log((1 - alpha) * acc / Z + alpha)
        else:
            log_p_a = torch.sum(torch.log((1 - alpha) * acc / Z + alpha),
                                dim=self.not_group_sum_dim)
        # Get z
        z = eps.view(num_samples, *self.not_group_shape, *self.group_shape)
        z = z.permute(*self.perm_inv).contiguous()
        # Get Gaussian density
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(0.5 * torch.pow(z, 2), dim=self.sum_dim)
        # Apply affine transform
        log_p_flows = 0
        if self.affine_transform is not None:
            if self.class_cond:
                z, log_det = self.affine_transform(z, y)
            else:
                z, log_det = self.affine_transform(z)
            log_p_flows = log_p_flows - log_det
        # Apply flows
        for flow in self.flows:
            z, log_det = flow(z)
            log_p_flows = log_p_flows - log_det
        # Get final density
        log_p = log_p_gauss + log_p_a + log_p_flows
        return z, log_p

    def log_prob(self, z, y=None):
        # Get batch size, dtype, and device
        batch_size = z.size(0)
        dtype = z.dtype
        device = z.device
        # Perpare onehot encoding of class if needed
        if self.class_cond:
            if y.dim() == 1:
                y_onehot = torch.zeros((len(y), self.num_classes), dtype=dtype,
                                       device=device)
                y_onehot.scatter_(1, y[:, None], 1)
                y = y_onehot
        # Reverse flows
        log_p = 0
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_p = log_p + log_det
        # Reverse affine transform
        if self.affine_transform is not None:
            if self.class_cond:
                z, log_det = self.affine_transform.inverse(z, y)
            else:
                z, log_det = self.affine_transform.inverse(z)
            log_p = log_p + log_det
        # Get Gaussian density
        log_p_gauss = - 0.5 * self.d * np.log(2 * np.pi) \
                      - torch.sum(0.5 * torch.pow(z, 2), dim=self.sum_dim)
        # Update normalization constant
        if self.training or torch.any(self.Z < 0.):
            Z_samples = batch_size if self.Z_samples is None else self.Z_samples
            eps = torch.randn(Z_samples, *self.group_shape, dtype=dtype,
                              device=device)
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
        z = z.permute(*self.perm).contiguous()
        acc = self.a(z.view(-1, *self.group_shape))
        if self.class_cond:
            acc = acc.view(batch_size, -1, self.num_classes, self.num_groups)
            acc = torch.sum(acc * y[:, None, :, None], dim=2)
        else:
            acc = acc.view(batch_size, -1, self.num_groups)
        if self.same_dist:
            acc = acc.view(batch_size, -1)
        else:
            acc = torch.diagonal(acc, dim1=1, dim2=2)
        acc = acc.view(batch_size, *self.not_group_shape)
        # Get normalization constant
        if self.class_cond:
            Z = y @ Z.view(self.num_classes, self.num_groups)
        if self.same_dist:
            Z = Z.view(-1, *([1] * len(self.not_group_shape)))
        else:
            Z = Z.view(-1, *self.not_group_shape)
        alpha = (1 - Z) ** (self.T - 1)
        if len(self.not_group_sum_dim) == 0:
            log_p_a = torch.log((1 - alpha) * acc / Z + alpha)
        else:
            log_p_a = torch.sum(torch.log((1 - alpha) * acc / Z + alpha),
                                dim=self.not_group_sum_dim)
        log_p = log_p + log_p_a + log_p_gauss
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
                eps = torch.randn(num_samples, *self.group_shape, dtype=dtype,
                                  device=device)
                acc_ = self.a(eps)
                Z_batch = torch.mean(acc_, dim=0)
                self.Z = self.Z + Z_batch.detach() / num_batches