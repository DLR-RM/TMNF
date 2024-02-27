import argparse
import os
from mmcv import Config
# from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)

# from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

# for new modules in mmdet
from flowDet_mmdet.utils.gmmDet_utils import run_inference
from flowDet_mmdet.utils.helper_utils import save_results_json, save_results_pickle
# from flowDet_mmdet.utils.functional_utils import get_feat_ext_end_idx
from flowDet_mmdet.flowDet_env import FLOWDET_WEIGHTS_ROOT, FLOWDET_FEAT_ROOT, FLOWDET_CONFIG_ROOT
BASE_WEIGHTS_FOLDER = FLOWDET_WEIGHTS_ROOT
CONFIG_ROOT = FLOWDET_CONFIG_ROOT

def parse_args():
    parser = argparse.ArgumentParser(description='A script to extract corresponding features and save the raw detections')
    parser.add_argument('--datasplit_type', type=str, help='to which data split produced by the code of which version: GMMDet or flowDet, flowDet_msfeats')
    parser.add_argument('--config', default = '/flowDet/feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsLogits.py', help='config of voc or coco')
    parser.add_argument('--subset', default = None, help='voc: trainCS12/07 or val or testOS, coco: trainCS, val, testOS')
    parser.add_argument('--maxOneDetOneRP', default=True, 
                        help=("the constraint applied on filtering raw predictions from a detector:"
                        "to get the detection with highest scores among those from the same region proposal."))
    parser.add_argument('--confThresh', type=float, default= 0.2, help=' only detections with a max softmax above this score are considered valid')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    parser.add_argument('--weights_dir', default = None, help='directory of object detector weights')
    parser.add_argument('--checkpoint', default = 'latest.pth', help='what is the name of the object detector weights')
    args = parser.parse_args()
    return args

def build_eval_model(cfg, weights_path):
    print("Building model")
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    # model = build_detector(cfg.model)
    # method_list = [func for func in dir(model) if callable(getattr(model, func))]
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, weights_path, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model

def main(argparse):
    confThresh = argparse.confThresh
    config_path = os.path.join(CONFIG_ROOT, argparse.config)
    cfg = Config.fromfile(config_path)
    distributed = False
    samples_per_gpu = cfg.data.testOS.pop('samples_per_gpu', 1)
    # feat_ext_end_idx = get_feat_ext_end_idx(cfg)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.testOS.pipeline = replace_ImageToTensor(cfg.data.testOS.pipeline)

    weights_path = os.path.join(BASE_WEIGHTS_FOLDER, f"{argparse.weights_dir}/{argparse.checkpoint}")

    model = build_eval_model(cfg, weights_path)
    ds_subsets = [argparse.subset]
    if "Ycbv" in cfg.dataset_type: 
        if argparse.subset == "train":
            ds_subsets = ["trainCS"]
        elif argparse.subset == "testOS":
            ds_subsets = ["test", "testOS"]
    elif "Coco" in cfg.dataset_type and argparse.subset == "train":
        ds_subsets = ["trainCS"]
    elif "VOC" in cfg.dataset_type and argparse.subset == "train":
        ds_subsets = ["trainCS07", "trainCS12"]
        
    for ds_subset in ds_subsets:
        save_path = f"{FLOWDET_FEAT_ROOT}{argparse.datasplit_type}/{cfg.model.type}/raw/{cfg.dataset_type}/{ds_subset}" #_conf{confThresh}"
        os.makedirs(save_path, exist_ok=True)
        print(f"Running inference on {cfg.dataset_type}/{ds_subset} with samples_per_gpu {samples_per_gpu}")
        dataset = build_dataset(cfg.data[ds_subset])
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        results = run_inference(model, data_loader, confThresh, argparse.maxOneDetOneRP)
        
        save_results_json(results, save_path+f"/{argparse.saveNm}")
        # save_results_pickle(results, save_path)

if __name__ == "__main__":
    argparse = parse_args()
    main(argparse)