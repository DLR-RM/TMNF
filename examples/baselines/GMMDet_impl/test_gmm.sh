#!/bin/bash 
path=$( dirname -- "$( readlink -f -- "$0"; )"; )
FEAT_FN="frcnn_GMMDet_Voc" # "frcnn_GMMDet_Voc"
DETECTOR_TYPE="FasterRCNN"
DATASET="voc" #  

cd $path
echo 'In directory:'$path
echo "Getting Results from DATASET: ${DATASET}"
python get_results.py $DETECTOR_TYPE --dataset $DATASET --saveNm $FEAT_FN --numComp 1