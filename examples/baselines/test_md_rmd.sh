#!/bin/bash 
path="/home/feng_ji/Documents/Projects/flowDet_mmdet/examples/baselines" # $( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_SET="spirit" # voc or coco or ycbv or spirit

if [ $DATA_SET = "voc" ]
then
    NF_CONFIF="/home/feng_ji/Documents/Projects/flowDet_mmdet/configs/flowDet/feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsLogits.py"
    FEAT_FN="GMMDet_Voc_msFeats" # "GMMDet_Voc", "CE_Voc", "GMMDet_Voc_msFeats", "CE_Voc_msFeats"
    SCRIPT="train_test_md_rmd.py"
elif [ $DATA_SET = "coco" ]
then
    NF_CONFIF="/home/feng_ji/Documents/Projects/flowDet_mmdet/configs/flowDet/feat_ext/faster_rcnn_r50_fpn_cocoOS_clsLogits.py"
    FEAT_FN="frcnn_GMM_Coco" # "frcnn_GMM_Coco", "frcnn_CE_Coco"
    SCRIPT="train_test_md_rmd.py"
elif [ $DATA_SET = "spirit" ]
then
    NF_CONFIF="/home/feng_ji/Documents/Projects/flowDet_mmdet/configs/flowDet/train/onlyFlow_spirit/oF_frcnn_spirit_rnvp_logits.py"
    FEAT_FN="sim_real" # "CE_ycbv_low_mixed_msFeats", "CE_ycbv_mixed_msFeats", "CE_ycbv_msFeats", "GMM_ycbv_msFeats"
    SCRIPT="train_test_md_rmd.py"
fi

DATASPLIT_TYPE="flowDet" #  "GMMDet" or "flowDet",
cd $path
echo 'In directory:'$path

echo "Test (Relative) Mahalanobis Distance with dataset_split: ${DATASPLIT_TYPE}"
python $SCRIPT --config $NF_CONFIF --datasplit_type $DATASPLIT_TYPE --feat_fn $FEAT_FN 
