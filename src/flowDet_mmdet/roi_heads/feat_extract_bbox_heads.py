import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
# from mmcv.ops.nms import batched_nms

from mmdet.core import multiclass_nms
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead 

@HEADS.register_module()
class Shared2FCBBoxHeadEnergy(ConvFCBBoxHead):
    def __init__(self, 
                fc_out_channels=1024, 
                return_cls_feat=None,
                return_reg_feat=None,
                *args, **kwargs):
        super(Shared2FCBBoxHeadEnergy, self).__init__(
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
        self.return_cls_feat = return_cls_feat
        self.return_reg_feat = return_reg_feat
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        # print(f"x: {x.shape}")
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            # print(f"x: {x.shape}")
            x = x.flatten(1)
            # print(f"x: {x.shape}")

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
            
            # print(f"x: {x.shape}")
        # separate branches
        ms_feat = x
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
        if self.return_cls_feat == "logits":
            return cls_score, bbox_pred
        else:
            return cls_score, bbox_pred, ms_feat
    
    # from mmdetection/mmdet/models/roi_heads/bbox_heads/bbox_head.py 
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
                
        bboxes = self.bbox_coder.decode(
            rois[..., 1:], bbox_pred, max_shape=img_shape)

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
                                                
            # extract remaining logits and append them to det_bboxes 
            if self.return_cls_feat == "logits":
                cls_score = torch.repeat_interleave(cls_score, repeats=self.num_classes, dim=0)
                cls_score = cls_score[keep_ind] # (N, C+1)
                det_bboxes = torch.cat((det_bboxes, cls_score), dim=1)
            
            if self.return_reg_feat == "bboxes":
                bbox_pred = torch.repeat_interleave(bbox_pred, repeats=self.num_classes, dim=0)
                bbox_pred = bbox_pred[keep_ind] # (N, C*4)
                det_bboxes = torch.cat((det_bboxes, bbox_pred), dim=1)

            return det_bboxes, det_labels # , keep_ind

@HEADS.register_module()
class Shared2FC1FCBBoxHead(ConvFCBBoxHead):
    def __init__(self, 
                fc_out_channels=1024,
                *args, **kwargs):
        super(Shared2FC1FCBBoxHead, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=1,
            num_reg_convs=0,
            num_reg_fcs=1,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)

@HEADS.register_module()
class Shared2FC1FCBBoxHeadFeat(ConvFCBBoxHead):
    def __init__(self, 
                fc_out_channels=1024,
                return_cls_feat=None,
                return_reg_feat=None,
                *args, **kwargs):
        super(Shared2FC1FCBBoxHeadFeat, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=1,
            num_reg_convs=0,
            num_reg_fcs=1,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
        # the following two MUST be False during training
        # and optionally True based on the use case during testing
        self.return_cls_feat = return_cls_feat
        self.return_reg_feat = return_reg_feat

    # from mmdetection/mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
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

        # concatenate penultimate feature to cls_score and bbox_pred
        if self.return_cls_feat == "penultimate":
            cls_score = torch.cat((cls_score, x_cls), dim=1)
        if self.return_reg_feat == "penultimate":
            bbox_pred = torch.cat((bbox_pred, x_reg), dim=1)
        return cls_score, bbox_pred

   # from mmdetection/mmdet/models/roi_heads/bbox_heads/bbox_head.py 
    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                   rois,
                   cls_score,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        # recover features
        if self.return_cls_feat == "penultimate":
            feat_cls = cls_score[:, self.num_classes+1:]       
            cls_score = cls_score[:, :self.num_classes+1] 

        reg_dim = 4 if self.reg_class_agnostic else 4 * self.num_classes
        # reg_feat_flag = bbox_pred.size(1) > reg_dim
        if self.return_reg_feat == "penultimate":
            feat_reg = bbox_pred[:, reg_dim:]       
            bbox_pred = bbox_pred[:, :reg_dim] 

        # some loss (Seesaw loss..) may have custom activation
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None
                
        bboxes = self.bbox_coder.decode(
            rois[..., 1:], bbox_pred, max_shape=img_shape)

        if rescale and bboxes.size(0) > 0:
            scale_factor = bboxes.new_tensor(scale_factor)
            bboxes = (bboxes.view(bboxes.size(0), -1, 4) / scale_factor).view(
                bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels, keep_ind = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=True)
                                                
            # extract remaining logits and append them to det_bboxes 

            if self.return_cls_feat == "penultimate":
                cls_score = torch.repeat_interleave(cls_score, repeats=self.num_classes, dim=0)
                cls_score = cls_score[keep_ind] # (N, C+1)
                det_bboxes = torch.cat((det_bboxes, cls_score), dim=1)
                feat_cls = torch.repeat_interleave(feat_cls, repeats=self.num_classes, dim=0)
                feat_cls = feat_cls[keep_ind] # (N, 1024)
                det_bboxes = torch.cat((det_bboxes, feat_cls), dim=1)
            
            if self.return_reg_feat == "penultimate":
                feat_reg = torch.repeat_interleave(feat_reg, repeats=self.num_classes, dim=0)
                feat_reg = feat_reg[keep_ind] # (N, 1024)
                det_bboxes = torch.cat((det_bboxes, feat_reg), dim=1)

            return det_bboxes, det_labels
