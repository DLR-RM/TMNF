## Topology-matching Normalizing Flows for OOD Detection in Robot Learning  [[paper](https://openreview.net/forum?id=BzjLaVvr955)]

This implementation includes the code base of conditional Resampled base distribution with information bottleneck for synthetic density estimation and OOD detection in object detection.

### Overview
To facilitate reliable deployments of autonomous robots in the real world, Out-of-Distribution (OOD) detection capabilities are often required. A
powerful approach for OOD detection is based on density estimation with Normalizing Flows (NFs). However, we find that prior work with NFs attempts to
match the complex target distribution topologically with na ̈ıve base distributions leading to adverse implications. In this work, we circumvent this topological
mismatch using expressive class-conditional base distributions that we train with an information-theoretic objective to match the required topology. The proposed
method enjoys the merits of wide compatibility with existing learned models, efficient runtime, and low memory overhead while enhancing the OOD detection
performance. We demonstrate the benefits of our method in density estimation, 2D object detection benchmarks and in particular, showcase the applicability in a
real-robot deployment.
![method](https://github.com/JianxiangFENG/TMNF_mmdet/assets/26474993/3a672c16-0fec-4f39-a556-bfc2219363ec)


### Folder Structure
    ├── configs     
    │   ├── _base_
    │   ├── faster_rcnn
    │   ├── gmmDet          # config files for training FasterRCNN
    │   ├── flowDet         
    │   |   ├── feat_ext    # config files for feature extraction
    |   |   └── train       # config files for different flow variants
    ├── data                # needs to be created for saving the extracted features and model weights
    │   ├── datasets        # for extracted features
    |   |── figures         # for result figures
    |   |── FRCNN           # for saving models of new detectors
    │   └── trained_weights # for normalizing flow model weights
    ├── dependencies        # specific dependencies to be installed
    │   ├── mmdetection
    │   └── resampled-base-flows
    ├── environment.yml     # conda environment file
    ├── examples            # experiment scripts        
    │   ├── baselines       # inlcuding (relative) mahalanobis distance and GMMDet
    |       └──GMMDet_impl  # original implementation of GMMDet
    │   ├── flowDet         # topology-matching flows for OOD detection in Object Detection
    |   ├── Topologically-Matched-NFs-Density-Estimation.ipynb      # topology-matching flows for 2D density estimation
    │   └── train_det.bash  # to train a FasterRCNN from scratch
    ├── src                 # topology-matching flows implementation
    │   ├── flowDet_mmdet



### Installation
- Git clone with the argument `--recurse-submodules` for the submodule: resampled-base-flows. Assume that the working machine has GPUs available.

   ` git clone --recurse-submodules https://github.com/DLR-RM/TMNF.git`

- Install Conda environment: `conda env create -f environment.yml`

- Install FlowDet (implementation of Topology-matching Normalizing Flows for OOD detection): `pip install -e .`.

- Install mmdet: `cd dependencies/mmdetection/ & pip install -e .`.

- Install a [forked version](https://github.com/JianxiangFENG/resampled-base-flows) of resampled-base-flows: `cd ../dependencies/resampled-base-flows/ & pip install -e .`.

- Install mmcv-full with version 1.6.0: `pip install openmim & mim install mmcv-full==1.6.0`.

### Downloading Extracted Features and Flow Model Weights

- Download the extracted logits and weights above to the `data` folder from this [link](https://drive.google.com/file/d/1RnzzXmGPoRldrJsOKK2h4A8rNRS9sWmj/view?usp=sharing).

- The `data` folder includes extracted logits of the object detector and pre-trained flow models for both datasets (Pascal-Voc-OS and MS-Coco-OS).

- Modify the `FILE_ROOT` in `src/flowDet_mmdet/flowDet_env.py` to the absolute path of the current folder.


### Running Experiments

-  `cd examples/flowDet`

- Reproducing Results (test only): In `train_onlyFlow.sh`, provide the config file name that has been trained, set `ONLY_EVAL` to True and run `./train_onlyFlow.sh`. Config files are stored in the `configs/flowDet`. The pre-trained flow weights for the results in the paper are provided in the folder `data/trained_weights`.

- Train and test:  set `ONLY_EVAL` to False and run `./train_onlyFlow.sh`


### Training Faster-RCNN and Extracting Features
- training: `cd examples & ./train_det.bash `, the training dataset can be decided inside the script

- feature extraction: `cd examples/flowDet & ./extract_feat.sh`

### Citation 
If you find that this work is helpful for your research, welcome to cite with the following Bibtex:
```
@inproceedings{
feng2023topologymatching,
title={Topology-Matching Normalizing Flows for Out-of-Distribution Detection in Robot Learning},
author={Jianxiang Feng and Jongseok Lee and Simon Geisler and Stephan G{\"u}nnemann and Rudolph Triebel},
booktitle={7th Annual Conference on Robot Learning},
year={2023},
url={https://openreview.net/forum?id=BzjLaVvr955}
}
```


### Acknowledgement

Credits to the repositories: [Uncertainty for Identifying Open-Set Errors in Visual Object Detection](https://github.com/dimitymiller/openset_detection) and [Resampled-base-flows](https://github.com/VincentStimper/resampled-base-flows), upon which the code base of this work is established.
