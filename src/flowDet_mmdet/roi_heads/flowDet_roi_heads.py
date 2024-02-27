import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead 

logger = logging.getLogger(__name__)

@HEADS.register_module()
class FlowDetPNRoIHead(StandardRoIHead):
    def __init__(self, **kwargs):
        super(FlowDetPNRoIHead, self).__init__(**kwargs)
    # overriding functions: _filter_preds(), _bbox_forward_train();

    def _filter_preds(self, label):
        # try to remove noise of near-OOD examples
        pos_inds = torch.nonzero((label >= 0) & (label < self.bbox_head.num_classes), as_tuple=True)[0]
        num_pos_inds = pos_inds.shape[0]
        neg_inds = torch.nonzero((label == self.bbox_head.num_classes), as_tuple=True)[0]
        rnd_inds = torch.randperm(neg_inds.shape[0])[:num_pos_inds]
        neg_inds = neg_inds[rnd_inds]
        return torch.cat((pos_inds, neg_inds))

    def _filter_roi_targets(self, rois, bbox_targets, labels):
        kept_indices = self._filter_preds(labels)
        rois = rois[kept_indices, :]
        filtered_bbox_targets = []
        for target in bbox_targets:
            filtered_bbox_targets.append(target[kept_indices])
        del bbox_targets

        return rois, filtered_bbox_targets

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        labels, _, _, _ = bbox_targets

        # filter out most of bg and keep all gt and the same amount of gt rois
        rois, bbox_targets = self._filter_roi_targets(rois, bbox_targets, labels)

        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results


@HEADS.register_module()
class FlowDetNPNRoIHead(StandardRoIHead):
    def __init__(self, **kwargs):
        super(FlowDetNPNRoIHead, self).__init__(**kwargs)

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)

        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        labels, _, _, _ = bbox_targets
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results