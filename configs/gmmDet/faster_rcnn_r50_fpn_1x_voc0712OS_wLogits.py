_base_ = [
    './faster_rcnn_r50_fpn_vocOS_wLogits.py',
    '../_base_/datasets/voc0712OS.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
