import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import multiclass_nms
from mmdet.models.losses import accuracy
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead 

from flowDet_mmdet.utils.functional_utils import init_flow_module_v2

logger = logging.getLogger(__name__)

__budget_functions__ = {'one': lambda N: torch.ones_like(N),
                        'log': lambda N: torch.log(N + 1.),
                        'id': lambda N: N,
                        'id_normalized': lambda N: N / N.sum(),
                        'exp': lambda N: torch.exp(N),
                        'parametrized': lambda N: torch.nn.Parameter(torch.ones_like(N).to(N.device))}

@HEADS.register_module()
class Shared2FCBBoxHeadEnergyPN(ConvFCBBoxHead):
    def __init__(self, 
                flow_hyperparams,
                cls_num_lst,  # Count of data from each class in training set. list of ints
                fc_out_channels=1024, 
                pretrined_flow_path=None,
                budget_function='id',  # Budget function name applied on class count. name
                device="cuda",
                *args, **kwargs):
        super(Shared2FCBBoxHeadEnergyPN, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
            
        """
        responsible for new hyper-params.
        """
        self.pretrined_flow_path = pretrined_flow_path
        self.flow_hyperparams = flow_hyperparams
        self.flow_module = init_flow_module_v2(flow_hyperparams, pretrined_flow_path=pretrined_flow_path, device=device)

        self.batch_norm = nn.BatchNorm1d(num_features=flow_hyperparams['input_dim'])

        if budget_function in __budget_functions__:
            self.cls_num_lst, self.budget_function = __budget_functions__[budget_function](cls_num_lst), budget_function
        else:
            raise NotImplementedError(f"budget_function({budget_function}) is not implemented!")

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        cls_score = self.fc_cls(x) if self.with_cls else None
        bn_logits = self.batch_norm(cls_score)
        if self.flow_hyperparams['prior_type'] == 'gmm':
            cls_nll, prior_ll, logdet = self.flow_module.nll(bn_logits, return_more=True) # N*(C+1)
        else:
            cls_nll = self.flow_module.nll(bn_logits)
        cls_score = torch.cat((cls_score, cls_nll), dim=1) # N * (2*(C+1)) else N*(C+2)

        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

    
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):
        """
        responsible for inference of bbox_heads in roi_heads.
        """
        # some loss (Seesaw loss..) may have custom activation
        cls_nll = cls_score[:, self.num_classes+1:]
        cls_score = cls_score[:, :self.num_classes+1]
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            # keep_ind is counted based on each detected categories without argmax
            # which means it is based on scores.reshape(-1) instead of scores (N, C)
            # where N is number of detected instance, C is number of class
            det_bboxes, det_labels, keep_ind = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=True)
                                                
            cls_nll = torch.repeat_interleave(cls_nll, repeats=self.num_classes, dim=0)
            cls_nll = cls_nll[keep_ind] # (N, C+1)
            det_bboxes = torch.cat((det_bboxes, cls_nll), dim=1) # (N, 5+C+1)

            return det_bboxes, det_labels

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        # overriding for extracting log prob from cls_score
        #TODO check the performance with mean of dirichlet distribution
        losses = dict()
        if cls_score is not None:
            cls_log_prob = -cls_score[:, self.num_classes+1:]
            cls_score = cls_score[:, :self.num_classes+1]
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    cls_log_prob,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
                    losses['nf_log_prob_acc'] = accuracy(cls_log_prob, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses

@HEADS.register_module()
class Shared2FCBBoxHeadEnergyNPN(ConvFCBBoxHead):
    def __init__(self, 
                fc_out_channels=1024, 
                pretrined_flow=None,
                *args, **kwargs):
        super(Shared2FCBBoxHeadEnergyNPN, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        # the following two can be optionally True based on the use case during testing
        self.pretrined_flow = pretrined_flow
    
    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        cls_score = self.fc_cls(x) if self.with_cls else None
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        return cls_score, bbox_pred

