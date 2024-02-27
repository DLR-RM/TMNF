import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet.core import bbox2roi # , bbox2result
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead 

logger = logging.getLogger(__name__)

def bbox2result(bboxes, labels, det_feats, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        det_feats (torch.Tensor | np.ndarray): shape (n, feat_dim)
        num_classes (int): class number

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            det_feats = det_feats.detach().cpu().numpy()
        results = []
        
        for i in range(num_classes): # no bg class
            cat_res = np.concatenate((bboxes[labels == i, :], det_feats[labels == i, :]), -1)
            results.append(cat_res)

        return results # [bboxes[labels == i, :] for i in range(num_classes)]
    
@HEADS.register_module()
class MSFeatExtractRoIHead(StandardRoIHead):
    def __init__(self, **kwargs):
        super(MSFeatExtractRoIHead, self).__init__(**kwargs)
    # overriding functions: 
    # simple_test(), 
    # simple_test_bboxes(): to enable bbox_feats return;
    
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        cls_score, bbox_pred, bbox_feats = self.bbox_head(bbox_feats)
        # print(f"bbox_feats: {bbox_feats.shape}")
        # print(f"cls_score: {cls_score.shape}")
        # print(f"bbox_pred: {bbox_pred.shape}")

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        bbox_feats = bbox_results['bbox_feats'] # .view(bbox_results['bbox_feats'].shape[0], -1)
        num_det, feat_dim = bbox_feats.shape
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        # print(f"num_proposals_per_img: {num_proposals_per_img}")
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_feats = bbox_feats.split(num_proposals_per_img, 0)
        
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_feats = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_bbox_feats = rois[i].new_zeros(0, feat_dim)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_bbox_feats = rois[i].new_zeros(0, feat_dim)
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label, keep_ind = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
                
                det_bbox_feats = torch.repeat_interleave(bbox_feats[i], repeats=self.bbox_head.num_classes, dim=0)
                det_bbox_feats = det_bbox_feats[keep_ind]
                # print(f"det_bbox_feats.shape: {det_bbox_feats.shape}")
                # print(f"det_bbox.shape: {det_bbox.shape}")
                # print(f"det_label.shape: {det_label.shape}")
                # print(f"keep_ind: {keep_ind}")
                # det_bbox_feats = bbox_feats[i][keep_ind[0]].unsqueeze(0)
                # # print(f"det_bbox_feats: {det_bbox_feats.shape}")
                # for ind in keep_ind[1:]:
                #     det_bbox_feats = torch.cat((det_bbox_feats, bbox_feats[i][ind].unsqueeze(0)), dim=0)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_feats.append(det_bbox_feats)
        del bbox_feats
        return det_bboxes, det_labels, det_feats

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_feats = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i], det_feats[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

