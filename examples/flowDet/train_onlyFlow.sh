#!/bin/bash 
path=$( dirname -- "$( readlink -f -- "$0"; )"; )
DATA_SET="voc" # voc or coco
CONFIG="rnvp_logits_rbf_cls_ib" # [rnvp,nsf,residual]_logits_["", gmm, gmm_cls, gmm_cls_ib, rbf, rbf_cls, rbf_cls_ib]
SCRIPT="train_onlyFlow.py"

if [ $DATA_SET = "voc" ]
then
    NF_CONFIF="flowDet/train/onlyFlow_voc/oF_frcnn_voc_$CONFIG.py"
    FEAT_FN="GMMDet_Voc_msFeats" # "GMMDet_Voc_msFeats", "CE_Voc_msFeats"
elif [ $DATA_SET = "coco" ]
then
    NF_CONFIF="flowDet/train/onlyFlow_coco/oF_frcnn_coco_$CONFIG.py"
    FEAT_FN="frcnn_GMM_Coco" # "frcnn_GMM_Coco", "frcnn_CE_Coco"
fi

FEAT_TYPE="logits" # 'logits'
ONLY_EVAL=True

cd $path
echo 'In directory:'$path

if [ $ONLY_EVAL = "True" ]
then
    echo "Only testing NF for $CONFIG ..."
    python $SCRIPT --config $NF_CONFIF --feat_fn $FEAT_FN --feat_type $FEAT_TYPE --only_eval 
else
    echo "Training and testing NF  for $CONFIG ..."
    python $SCRIPT --config $NF_CONFIF --feat_fn $FEAT_FN --feat_type $FEAT_TYPE
fi
