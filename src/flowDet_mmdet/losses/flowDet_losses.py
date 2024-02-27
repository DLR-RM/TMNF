import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.distributions.dirichlet import Dirichlet

import warnings
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss


def cross_entropy_logitNorm(pred,
                  label,
                  temp=0.01,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int | None): The label index to be ignored.
            If None, it will be set to default value. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.

    Returns:
        torch.Tensor: The calculated loss
    """
    # The default value of ignore_index is the same as F.cross_entropy
    ignore_index = -100 if ignore_index is None else ignore_index
    # element-wise losses

    norms = torch.norm(pred, p=2, dim=-1, keepdim=True) + 1e-7
    logit_norm = torch.div(pred, norms) / temp
    # return F.cross_entropy(logit_norm, target)
    loss = F.cross_entropy(
        logit_norm,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and avg_non_ignore and reduction == 'mean':
        avg_factor = label.numel() - (label == ignore_index).sum().item()

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

@LOSSES.register_module()
class CrossEntropyLogitNormLoss(nn.Module):

    def __init__(self,
                temp=0.01,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 ignore_index=None,
                 loss_weight=1.0,
                 avg_non_ignore=False):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            ignore_index (int | None): The label index to be ignored.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
            avg_non_ignore (bool): The flag decides to whether the loss is
                only averaged over non-ignored targets. Default: False.
        """
        super(CrossEntropyLogitNormLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.avg_non_ignore = avg_non_ignore
        if ((ignore_index is not None) and not self.avg_non_ignore
                and self.reduction == 'mean'):
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        self.temp = temp 

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * cross_entropy_logitNorm(
            cls_score,
            label,
            temp=self.temp,
            weight=weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            ignore_index=ignore_index,
            avg_non_ignore=self.avg_non_ignore,
            **kwargs)
        return loss_cls

def expected_nll(alpha, y_one_hot):
    # with autograd.detect_anomaly():
    output_dim = alpha.shape[1]
    alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, output_dim)
    # entropy_reg = Dirichlet(alpha).entropy()
    expected_nll = y_one_hot * (torch.digamma(alpha_0) - torch.digamma(alpha))
    return expected_nll.sum(1)

def expected_nmll(alpha, y_one_hot):
    # with autograd.detect_anomaly():
    mean = alpha / alpha.sum(1)
    expected_nmll = -y_one_hot * torch.log(mean)
    return expected_nmll.sum(1)

def expected_brier_score(alpha, y_one_hot):
    # with autograd.detect_anomaly():
    dir_dist = Dirichlet(alpha)
    mean = dir_dist.mean
    var = dir_dist.variance
    error = ((mean - y_one_hot) ** 2)
    expected_brier_score = error + var

    return expected_brier_score.sum(1)

@LOSSES.register_module()
class PNCrossEntropyLoss(nn.Module):
    def __init__(self,
                num_classes,
                loss_func="nll",
                reduction='mean',
                class_weight=None,
                ignore_index=None,
                cls_num_lst=[],
                loss_weight=1.0, 
                entropy_weight = 0.1):
        super(PNCrossEntropyLoss, self).__init__()
        self.loss_func = loss_func
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.entropy_weight = entropy_weight
        self.class_weight = class_weight
        self.cls_num_lst = cls_num_lst
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        if self.loss_func == "nll":
            self.cls_criterion = expected_nll
        elif self.loss_func == "nmll":
            self.cls_criterion = expected_nmll
        elif self.loss_func == "brier_score":
            self.cls_criterion = expected_brier_score

    def _filter_preds(self, label):
        # try to remove noise of near-OOD examples
        # pos_inds = (label >= 0) & (label < self.num_classes)
        # neg_inds = (label == self.num_classes)
        # pos_cls_log_prob = cls_log_prob[pos_inds.type(torch.bool)]
        # pos_label = label[pos_inds.type(torch.bool)]
        # neg_cls_log_prob = cls_log_prob[neg_inds.type(torch.bool)][:pos_cls_log_prob.size(0)]
        # neg_label = label[neg_inds.type(torch.bool)][:pos_cls_log_prob.size(0)]
        # cls_log_prob = torch.cat((pos_cls_log_prob, neg_cls_log_prob))
        # label = torch.cat((pos_label, neg_label))
        # try to remove noise of near-OOD examples
        
        pos_inds = torch.nonzero((label >= 0) & (label < self.num_classes), as_tuple=True)[0]
        num_pos_inds = pos_inds.shape[0]
        neg_inds = torch.nonzero((label == self.num_classes), as_tuple=True)[0]
        rnd_inds = torch.randperm(neg_inds.shape[0])[:num_pos_inds]
        neg_inds = neg_inds[rnd_inds]
        return torch.cat((pos_inds, neg_inds))

    def forward(self,
                cls_score,
                cls_log_prob,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if ignore_index is None:
            ignore_index = self.ignore_index

        dim_logits = self.num_classes + 1
        # kept_indices = self._filter_preds(label)
        # cls_log_prob = cls_log_prob[kept_indices, :]
        # label = label[kept_indices]

        alpha = torch.zeros((cls_log_prob.shape[0], dim_logits)).to(cls_log_prob.device.type)
        for c in range(dim_logits):
            alpha[:, c] = 1. + (self.cls_num_lst[c] * torch.exp(cls_log_prob[:, c]))

        y_one_hot = F.one_hot(label, num_classes=dim_logits)
        loss_cls = self.cls_criterion(alpha, y_one_hot)
        if self.entropy_weight > 0.:
            entorpy_term = self.entropy_weight * Dirichlet(alpha).entropy()
            # entorpy_term = torch.unsqueeze(entorpy_term, 1)
            loss_cls += -entorpy_term

        loss_cls = weight_reduce_loss(loss_cls, reduction=reduction, avg_factor=avg_factor)
        return loss_cls

@LOSSES.register_module()
class NPNCrossEntropyLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0, anchor_weight = 0.1, num_classes = 15):
      
        super(NPNCrossEntropyLoss, self).__init__()
    
    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        pass