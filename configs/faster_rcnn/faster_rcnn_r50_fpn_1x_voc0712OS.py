_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_vocOS.py', '../_base_/datasets/voc0712OS.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[4, 6])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=7) 