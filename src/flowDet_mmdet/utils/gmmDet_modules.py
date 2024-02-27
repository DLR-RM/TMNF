# copied from https://github.com/dimitymiller/openset_detection
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmcv.cnn import ConvModule
from mmcv.ops.nms import batched_nms
from mmdet.models.builder import HEADS, LOSSES
from mmdet.models.roi_heads.bbox_heads import BBoxHead
from mmdet.models.losses.utils import weight_reduce_loss

########### customized loss function ###########
def anchor_cross_entropy(distances, 
                    logits,
                    label,
                    weight=None,
                    anchor_weight = 0.1,
                    num_classes = 15,
                    reduction='mean',
                    avg_factor=None,
                    class_weight=None):
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

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    lossCE = F.cross_entropy(logits, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    lossCE = weight_reduce_loss(
        lossCE, weight=weight, reduction=reduction, avg_factor=avg_factor)

    if len(label.size()) > 1:
        label = torch.reshape(label, (-1, num_classes))
        label = torch.argmax(label, dim = 1)
        label = label.long()

    #don't apply to background classes
    mask = label != num_classes
    if torch.sum(mask) != 0:
        label = label[mask]
        distances = distances[mask]
     
        loss_a = torch.gather(distances, 1, label.view(-1, 1)).view(-1)
        
        if weight != None:
            weight = weight.reshape(-1)[mask]
            loss_a *= weight

        if reduction == 'mean':
            avg_factor = torch.sum(mask)
            if avg_factor is not None:
                loss_a = loss_a.sum()/avg_factor
            else:
                loss_a = torch.mean(loss_a)
        else:
            loss_a = torch.sum(loss_a)
    else:
        loss_a = 0

    return lossCE + (anchor_weight*loss_a)


@LOSSES.register_module()
class AnchorwCrossEntropyLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0, anchor_weight = 0.1, num_classes = 15):
      
        super(AnchorwCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        self.num_classes = num_classes
        self.anchor_weight = anchor_weight

        #plus one to account for background class
        anch = torch.diag(torch.ones(num_classes+1)*5)
        anch = torch.where(anch != 0, anch, torch.Tensor([-5]))
        self.anchors = nn.Parameter(anch, requires_grad = False).cuda()
        
        self.cls_criterion = anchor_cross_entropy

    def euclideanDistance(self, logits):
        #plus one to account for background clss logit
        logits = logits.view(-1, self.num_classes+1)
        n = logits.size(0)
        m = self.anchors.size(0)
        d = logits.size(1)

        x = logits.unsqueeze(1).expand(n, m, d)
        anchors = self.anchors.unsqueeze(0).expand(n, m, d)

        dists = torch.norm(x-anchors, 2, 2)

        return dists

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
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None

       
        distances = self.euclideanDistance(cls_score)

        # distances = self.euclideanDistance(cls_score[:, :-1])

        loss_cls = self.loss_weight * self.cls_criterion(
            distances,
            cls_score,
            label,
            weight = weight,
            num_classes = self.num_classes,
            anchor_weight = self.anchor_weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls


########### customized bbox_head in roi_heads ###########
# adapted from mmdetection/mmdet/core/post_processing/bbox_nms.py
def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   return_inds=False, return_logits = False, multi_logits = None):
    
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)


    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    
    
    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        # remove low scoring boxes
        valid_mask = scores > score_thr
       
    # multiply score_factor after threshold to preserve more bboxes, improve
    # mAP by 1% for YOLOv3
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors
        

    if not torch.onnx.is_in_onnx_export():
        # NonZero not supported  in TensorRT
        inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
        
        bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    else:
        # TensorRT NMS plugin has invalid output filled with -1
        # add dummy data to make detection output correct.
        bboxes = torch.cat([bboxes, bboxes.new_zeros(1, 4)], dim=0)
        scores = torch.cat([scores, scores.new_zeros(1)], dim=0)
        labels = torch.cat([labels, labels.new_zeros(1)], dim=0)
        

    if return_logits:
        multi_logits = multi_logits.squeeze(0)
        new_multiLogits = torch.repeat_interleave(multi_logits, repeats = multi_logits.size(1)-1, dim = 0)
        
        logits = new_multiLogits[inds]
       

    if bboxes.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_inds:
            return bboxes, labels, inds
        else:
            return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_logits:
        logits = logits[keep]
        logits = logits[:max_num]
        dets = torch.cat((dets, logits), 1)
        

    if return_inds:
        return dets, labels[keep], keep
    else:
        return dets, labels[keep]


@HEADS.register_module()
class BBoxHeadLogits(BBoxHead):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(
                     type='DeltaXYWHBBoxCoder',
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 reg_predictor_cfg=dict(type='Linear'),
                 cls_predictor_cfg=dict(type='Linear'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 init_cfg=None):
        super(BBoxHeadLogits, self).__init__(with_avg_pool=with_avg_pool,
                                            with_cls=with_cls,
                                            with_reg=with_reg,
                                            roi_feat_size=roi_feat_size,
                                            in_channels=in_channels,
                                            num_classes=num_classes,
                                            bbox_coder=bbox_coder,
                                            reg_class_agnostic=reg_class_agnostic,
                                            reg_decoded_bbox=reg_decoded_bbox,
                                            reg_predictor_cfg=reg_predictor_cfg,
                                            cls_predictor_cfg=cls_predictor_cfg,
                                            loss_cls=loss_cls,
                                            loss_bbox=loss_bbox,
                                            init_cfg=init_cfg)

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        # TODO: revert to single image inference

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if rois.ndim == 2:
            # e.g. AugTest, Cascade R-CNN, HTC, SCNet...
            batch_mode = False
            # add batch dimension
            if scores is not None:
                scores = scores.unsqueeze(0)
            if bbox_pred is not None:
                bbox_pred = bbox_pred.unsqueeze(0)
            rois = rois.unsqueeze(0)

            assert isinstance(scale_factor, np.ndarray)
            scale_factor = (scale_factor, )

        elif rois.ndim == 3:
            # all input tensor have batch dimension
            batch_mode = True
            assert isinstance(scale_factor, tuple)
        else:
            raise NotImplementedError(f'Unexpect shape of roi {rois.shape}')

        # bbox_pred would be None in some detector when with_reg is False,
        # e.g. Grid R-CNN.
        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        num_bboxes = bboxes.size(-2)
        if rescale and num_bboxes > 0:
            # B, 1, bboxes.size(-1)
            scale_factor = bboxes.new_tensor(scale_factor).unsqueeze(1).repeat(
                1, 1,
                bboxes.size(-1) // 4)
            bboxes /= scale_factor

        det_bboxes = []
        det_labels = []
        for (bbox, score) in zip(bboxes, scores):
            if cfg is not None:
                det_bbox, det_label = multiclass_nms(bbox, score,
                                                     cfg.score_thr, cfg.nms,
                                                     cfg.max_per_img, return_logits = cfg.return_logits, multi_logits = cls_score)
            else:
                det_bbox, det_label = bbox, score
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)

        if not batch_mode:
            single_det_bboxes = det_bboxes[0]
            single_det_labels = det_labels[0]
            # tuple[Tensor, Tensor]
            return single_det_bboxes, single_det_labels
        else:
            # tuple[list[Tensor], list[Tensor]]
            return det_bboxes, det_labels


@HEADS.register_module()
class ConvFCBBoxHeadLogits(BBoxHeadLogits):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.
    .. code-block:: none
                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):

        super(ConvFCBBoxHeadLogits, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            self.fc_cls = nn.Linear(self.cls_last_dim, self.num_classes + 1)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.
        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCBBoxHeadLogits, self).init_weights()
        # conv layers are already initialized by ConvModule
        for module_list in [self.shared_fcs, self.cls_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

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
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHeadLogits(ConvFCBBoxHeadLogits):
    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHeadLogits, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
