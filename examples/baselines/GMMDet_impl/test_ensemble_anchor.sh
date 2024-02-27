#!/bin/bash 

echo 'Testing ensembles with base weights:' $1 
echo 'Save names:' $2 
echo 'Dataset:' $3 
Dataset=$3
save_fn=$2
weight_dir=$1
path=$( dirname -- "$( readlink -f -- "$0"; )"; )
cd $path
echo 'In directory:'$path

for NUM in 1 2 3 4 5
do
    echo Testing ensemble number $NUM
    if [ $Dataset = "voc" ]
    then
        python test_data.py --subset train07 --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset
        python test_data.py --subset train12 --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset
    else 
        python test_data.py --subset train --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset
        
    fi

    python test_data.py --subset val --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset

    if [ $Dataset = "ycbv" ]
    then
        python test_data.py --subset test --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset
        python test_data.py --subset testOS --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset
    else
        python test_data.py --subset test --dir "${weight_dir}${NUM}" --saveNm "${save_fn}${NUM}" --dataset $Dataset
    fi
done

echo Merging detections
if [ $Dataset = "voc" ]
then
    python merge_ensemble.py --subset train07 --saveNm "${save_fn}" --dataset $Dataset
    python merge_ensemble.py --subset train12 --saveNm "${save_fn}" --dataset $Dataset
else 
    python merge_ensemble.py --subset train --saveNm "${save_fn}" --dataset $Dataset
fi

python merge_ensemble.py --subset val --saveNm "${save_fn}" --dataset $Dataset

python merge_ensemble.py --subset test --saveNm "${save_fn}" --dataset $Dataset
if [ $Dataset = "ycbv" ]
then
    python merge_ensemble.py --subset testOS --saveNm "${save_fn}" --dataset $Dataset
fi

python associate_data.py FasterRCNN --saveNm "${save_fn}"Ensemble08 --dataset $Dataset

python get_results.py FasterRCNN --saveNm "${save_fn}"Ensemble08 --dataset $Dataset # --saveResults True
