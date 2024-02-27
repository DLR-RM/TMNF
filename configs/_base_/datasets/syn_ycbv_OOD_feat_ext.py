import os

from torch import classes
from flowDet_mmdet.flowDet_env import FLOWDET_DATASETS_ROOT
dataset_type = 'YcbvDataset'
data_root = FLOWDET_DATASETS_ROOT+'/syn_ycbv/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

syn_ycbv_cs_classes=['019_pitcher_base',
                      '021_bleach_cleaner',
                      '024_bowl',
                      '025_mug',
                      '035_power_drill',
                      '036_wood_block',
                      '037_scissors',
                      '040_large_marker',
                      '051_large_clamp',
                      '052_extra_large_clamp',
                      '061_foam_brick']
syn_ycbv_os_classes=['019_pitcher_base',
                      '021_bleach_cleaner',
                      '024_bowl',
                      '025_mug',
                      '035_power_drill',
                      '036_wood_block',
                      '037_scissors',
                      '040_large_marker',
                      '051_large_clamp',
                      '052_extra_large_clamp',
                      '061_foam_brick',
                      'OOD']

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    trainCS=dict(
        type="CocoDataset",
        classes=syn_ycbv_cs_classes,
        ann_file=data_root + 'train_5Inst_ID/annos/annotations.json',
        img_prefix=data_root + 'train_5Inst_ID/images',
        pipeline=test_pipeline),
    val=dict(
        type="CocoDataset",
        classes=syn_ycbv_cs_classes,
        ann_file=data_root + 'test_5Inst_ID/annos/annotations_original.json',
        img_prefix=data_root + 'test_5Inst_ID/images',
        pipeline=test_pipeline),
    test=dict(
        type="CocoDataset",
        classes=syn_ycbv_os_classes,
        ann_file=data_root + 'test_5Inst_ID/annos/annotations_original.json',
        img_prefix=data_root + 'test_5Inst_ID/images',
        pipeline=test_pipeline),
    testOS=dict(
        type="CocoDataset",
        classes=syn_ycbv_os_classes,
        ann_file=data_root + 'test_5Inst_OOD/annos/annotations.json',
        img_prefix=data_root + 'test_5Inst_OOD/images',
        pipeline=test_pipeline),
    )
evaluation = dict(interval=1, metric='bbox')
