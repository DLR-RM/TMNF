import os

from torch import classes
from flowDet_mmdet.flowDet_env import FLOWDET_DATASETS_ROOT
dataset_type = 'CocoDataset'
data_root = FLOWDET_DATASETS_ROOT+'/coco/'
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

coco_cs_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 
        'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 
        'cat', 'dog', 'horse', 'sheep', 'cow', 
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 
        'bowl', 'banana', 'apple', 'sandwich', 'orange',]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=coco_cs_classes,
        ann_file=data_root + 'annotations/instances_trainCS2017.json',
        img_prefix=data_root + 'images/trainCS2017/',
        pipeline=train_pipeline),
    trainCS=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_trainCS2017.json',
        img_prefix=data_root + 'images/trainCS2017/',
        pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_valCS2017.json',
        img_prefix=data_root + 'images/valCS2017/',
        pipeline=test_pipeline),
    valCS=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_valCS2017.json',
        img_prefix=data_root + 'images/valCS2017/',
        pipeline=test_pipeline),
    testOS = dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline),
    testCS=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_testCS2017.json',
        img_prefix=data_root + 'images/testCS2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'images/val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
