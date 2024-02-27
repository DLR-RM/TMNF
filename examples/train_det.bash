#!/bin/sh
path=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $path
echo 'In directory:'$path
DATASET="voc" # coco, voc
CONFIG_ROOT=$path"../configs/gmmDet"

if [ $DATASET = "voc" ]
then 
    echo "Training Faster RCNN on closed-set VOC"
    CONFIG="faster_rcnn_r50_fpn_1x_voc0712OS_wLogitNorm" # "faster_rcnn_r50_fpn_1x_voc0712OS_Anchor"
    RESUME="None"
    WEIGHT_DIR=$path"../data/FRCNN/frcnn_CACCE_Voc_A01"
elif [ $DATASET = "coco" ]
then 
    echo "Training Faster RCNN on closed-set COCO"
    # CONFIG="faster_rcnn_r50_fpn_1x_cocoOS_Anchor"
    CONFIG="faster_rcnn_r50_fpn_1x_cocoOS_Anchor"
    WEIGHT_DIR=$path"../data/FRCNN/frcnn_CACCE_Coco_A005"
    RESUME="None" # "$CONFIG_ROOT/faster_rcnn_r50_fpn_1x_cocoOS_Anchor_${ID}/latest.pth" 
fi

if [ $RESUME == "None" ]
then
    python tools/train.py $CONFIG_ROOT/$CONFIG".py" --gpus 1 --work-dir $WEIGHT_DIR
else
    python tools/train.py $CONFIG_ROOT/$CONFIG".py" --gpus 1 --work-dir $WEIGHT_DIR --resume-from $RESUME
fi