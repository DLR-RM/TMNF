#!/bin/bash 
path=$( dirname -- "$( readlink -f -- "$0"; )"; )

DATA="voc" # voc coco

if [ $DATA = "voc" ]
then 
    CONFIG="flowDet/feat_ext/faster_rcnn_r50_fpn_voc0712OS_msFeats.py"
    # CONFIG="flowDet/feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsFeats.py"
    WEIGHT_DIR="/frcnn_CACCE_Voc_A01" 
    SAVE_NAME="GMMDet_Voc_msFeats" # GMMDet_Voc_msFeats, CE_Voc_msFeats, FlowDet_Voc_msFeats
elif [ $DATA = "coco" ]
then
    # coco flowDet/faster_rcnn_r50_fpn_1x_ycbvOS/
    CONFIG="flowDet/feat_ext/faster_rcnn_r50_fpn_cocoOS_clsLogits.py"
    WEIGHT_DIR="/frcnn_CACCE_Coco_A005" 
    SAVE_NAME="frcnn_GMM_Coco"
fi 

DATASPLIT_TYPE="flowDet" #  "GMMDet" or "flowDet"
CONF_THR=0.2 # 0.2 in GMM paper
IOU_THR=0.5  # 0.5 in GMM paper
maxOneDetOneRP=True # default is True
CKP="latest.pth" # latest.pth

cd $path
echo 'In directory:'$path

########################################## Feature Extraction ##########################################
ext_feat_cmd="python feat_extraction.py --checkpoint $CKP --datasplit_type $DATASPLIT_TYPE --config $CONFIG --weights_dir $WEIGHT_DIR --saveNm $SAVE_NAME --confThresh $CONF_THR --maxOneDetOneRP $maxOneDetOneRP"
echo "Extracting features from training set"
cmd=$ext_feat_cmd" --subset train "
echo "executing $cmd"
$cmd

echo "Extracting features from VAL set"
cmd=$ext_feat_cmd" --subset val "
echo "executing $cmd"
$cmd

echo "Extracting features from TEST set"
cmd=$ext_feat_cmd" --subset testOS "
echo "executing $cmd"
$cmd

########################################## Prediction Assignment ##########################################
ass_cmd="python pred_assignment.py --datasplit_type $DATASPLIT_TYPE --config $CONFIG --confThresh $CONF_THR --iouThresh $IOU_THR --saveNm $SAVE_NAME"
echo "assigning types to training set"
cmd=$ass_cmd" --subset train "
echo "executing $cmd"
$cmd

echo "assigning types to val set"
cmd=$ass_cmd" --subset val "
echo "executing $cmd"
$cmd

echo "assigning types to test set"
cmd=$ass_cmd" --subset test "
echo "executing $cmd"
$cmd