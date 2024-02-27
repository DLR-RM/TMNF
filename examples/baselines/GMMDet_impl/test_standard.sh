#!/bin/bash 
# cmd: ./test_standard.sh  frcnn_CACCE_Voc_A01 frcnn_GMMDet_Voc voc 
# cmd: ./test_standard.sh  frcnn_CACCE_Coco_A005 frcnn_GMMDet_Coco coco
# ./test_standard.sh /home_local/feng_ji/Experiments/flowDet_exp/weights/GMMDet/faster_rcnn_r50_fpn_1x_ycbvOS_2 frcnn_GMM_noAnchor_Ycbv ycbv

# Dataset="ycbv" # voc, coco, ycbv
echo 'Testing with base weights:' $1 
echo 'Save name:' $2 
echo 'Dataset:' $3 
path=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $path
echo 'In directory:'$path

echo Testing training data set
if [ $3 = "voc" ]
then
    python test_data.py --subset train07 --dir "${1}" --saveNm "${2}" --dataset $3
    python test_data.py --subset train12 --dir "${1}" --saveNm "${2}" --dataset $3
else 
    python test_data.py --subset train --dir "${1}" --saveNm "${2}" --dataset $3
fi

echo Testing val and test data set
python test_data.py --subset val --dir "${1}" --saveNm "${2}" --dataset $3

if [ $3 = "ycbv" ]
then
    python test_data.py --subset test --dir "${1}" --saveNm "${2}" --dataset $3
    python test_data.py --subset testOS --dir "${1}" --saveNm "${2}" --dataset $3
else
    python test_data.py --subset test --dir "${1}" --saveNm "${2}" --dataset $3
fi

echo Associating data
python associate_data.py FasterRCNN --saveNm "${2}" --dataset $3 

echo Getting Results
python get_results.py FasterRCNN --saveNm "${2}" --dataset $3 # --saveResults True
