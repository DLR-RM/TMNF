import os
import argparse
import numpy as np
from mmcv import Config

from flowDet_mmdet.utils.gmmDet_utils import assign_train, assign_eval
from flowDet_mmdet.utils.helper_utils import save_results_json
from flowDet_mmdet.flowDet_env import FLOWDET_WEIGHTS_ROOT, FLOWDET_FEAT_ROOT, FLOWDET_CONFIG_ROOT
BASE_WEIGHTS_FOLDER = FLOWDET_WEIGHTS_ROOT
CONFIG_ROOT = FLOWDET_CONFIG_ROOT

def parse_args():
    parser = argparse.ArgumentParser(description='A script to extract corresponding features and save the raw detections')
    parser.add_argument('--datasplit_type', type=str, help='to which data split produced by the code of which version: GMMDet or flowDet, flowDet_msfeats')
    parser.add_argument('--subset', default = None, help='voc: trainCS12/07 or val or testOS, coco: trainCS, val, testOS')
    parser.add_argument('--config', default = '/flowDet/feat_ext/faster_rcnn_r50_fpn_voc0712OS_clsLogits.py', help='config of voc or coco')
    parser.add_argument('--confThresh', type=float, default= 0.2, help=' only detections with a max softmax above this score are considered valid')
    parser.add_argument('--iouThresh', type=float, default= 0.5, help=' only detections with a max softmax above this score are considered valid')
    parser.add_argument('--saveNm', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

def main(argparse):
    config_path = os.path.join(CONFIG_ROOT, argparse.config)
    cfg = Config.fromfile(config_path)
    num_classes = cfg.model.roi_head.bbox_head.num_classes
    num_logits = num_classes+1
    iouThresh = argparse.iouThresh
    confThresh = argparse.confThresh
    saveFileName = argparse.saveNm
    # print(f"argparse.datasplit_type: {argparse.datasplit_type}")
    # FLOWDET_FEAT_ROOT += f"/{FLOWDET_FEAT_ROOT}"
    datasplit_type = argparse.datasplit_type
    save_dir = f"{FLOWDET_FEAT_ROOT}/{datasplit_type}/{cfg.model.type}/associated/{cfg.dataset_type}"
    raw_res_dir = f"{FLOWDET_FEAT_ROOT}/{datasplit_type}/{cfg.model.type}/raw/{cfg.dataset_type}"

    # process train data
    if argparse.subset == "train":
        if "Ycbv" in cfg.dataset_type:
            ds_subsets = [cfg.data.trainCS]
            raw_res_pth_list = [f"{raw_res_dir}/trainCS/{saveFileName}.json"]
        if "Coco" in cfg.dataset_type:
            ds_subsets = [cfg.data.trainCS]
            raw_res_pth_list = [f"{raw_res_dir}/trainCS/{saveFileName}.json"]
        if "VOC" in cfg.dataset_type:
            ds_subsets = [cfg.data.trainCS07, cfg.data.trainCS12]
            raw_res_pth_list = [f"{raw_res_dir}/trainCS07/{saveFileName}.json",
                                f"{raw_res_dir}/trainCS12/{saveFileName}.json"]

        print(f'Assigning train data on {cfg.dataset_type}')
        print(f"Loading outputs from {raw_res_pth_list}.")
        trainDict = assign_train(ds_subsets, raw_res_pth_list, num_logits, iouThresh)
        print(f"detScores: {len(trainDict['scores'])} with avg {np.mean(trainDict['scores'])}")
        print(f"detLogits: {len(trainDict['logits'])}")
        # print(f"detFeats: {len(trainDict['feats'])}")
        print(f"all_filenames: {len(trainDict['filenames'])}")

        save_results_json(trainDict, os.path.join(save_dir, f'train/{saveFileName}'))
    
    # process eval data
    if argparse.subset == "val":
        print(f'Assigning val data on {cfg.dataset_type}.')
        raw_output_file =  f"{raw_res_dir}/val/{saveFileName}.json"
        print(f"Loading outputs from {raw_output_file}.")
        evalDict = assign_eval(cfg.data.val, raw_output_file, num_classes, confThresh, iouThresh)

        print(f"detScores: {len(evalDict['scores'])} with avg {np.mean(evalDict['scores'])}")
        print(f"detLogits: {len(evalDict['logits'])}")
        print(f"type: {len(evalDict['type'])}")
        print(f"detFeats: {len(evalDict['feats'])}")
        print(f"all_filenames: {len(evalDict['filenames'])}")
        print(f"#type 0: {np.sum(np.array(evalDict['type'])==0)}")
        print(f"#type 1: {np.sum(np.array(evalDict['type'])==1)}")
        print(f"#type 2: {np.sum(np.array(evalDict['type'])==2)}")
	
        save_results_json(evalDict, os.path.join(save_dir, f'val/{saveFileName}'))

    if argparse.subset == "test":
        gtLabel_mapping = None
        if "Ycbv" in cfg.dataset_type:
            # label remapping only for ycbv ood dataset
            def gtLabel_mapping(gtLabel_list):
                # because during testing, OOD object labels in OS test set are mapped to 0
                re_map_dict = {0:12, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10}
                for id in range(len(gtLabel_list)):
                    if gtLabel_list[id] in re_map_dict.keys():
                        gtLabel_list[id] = re_map_dict[gtLabel_list[id]]
                return gtLabel_list

        evalDict = {}
        print(f'Assigning Open-Set test data on {cfg.dataset_type}.')
        raw_output_file =  f"{raw_res_dir}/testOS/{saveFileName}.json"
        print(f"Loading outputs from {raw_output_file}.")
        evalDict = assign_eval(cfg.data.testOS, raw_output_file, num_classes, confThresh, iouThresh, gtLabel_mapping=gtLabel_mapping)

        print(f"detScores: {len(evalDict['scores'])} with avg {np.mean(evalDict['scores'])}")
        print(f"detScores for type 2 avg: {np.mean(np.array(evalDict['scores'])[np.array(evalDict['type'])==2])}")
        print(f"detScores for type 1 avg: {np.mean(np.array(evalDict['scores'])[np.array(evalDict['type'])==1])}")
        print(f"detLogits: {len(evalDict['logits'])}")
        print(f"type: {len(evalDict['type'])}")
        print(f"detFeats: {len(evalDict['feats'])}")
        print(f"all_filenames: {len(evalDict['filenames'])}")
        print(f"#type 0: {np.sum(np.array(evalDict['type'])==0)}")
        print(f"#type 1: {np.sum(np.array(evalDict['type'])==1)}")
        print(f"#type 2: {np.sum(np.array(evalDict['type'])==2)}")
        # save_results_json(evalDict, os.path.join(save_dir, f'test/{saveFileName}'))

        if "Ycbv" in cfg.dataset_type:        
            print(f'Assigning test ID data on {cfg.dataset_type}.')
            raw_output_file =  f"{raw_res_dir}/test/{saveFileName}.json"
            print(f"Loading outputs from {raw_output_file}.")
            evalDict_ = assign_eval(cfg.data.test, raw_output_file, num_classes, confThresh, iouThresh, gtLabel_mapping=None)
            for key, val in evalDict_.items():
                evalDict[key].extend(val)

            print(f"detScores: {len(evalDict_['scores'])} with avg {np.mean(evalDict_['scores'])}")
            print(f"detScores for type 2 avg: {np.mean(np.array(evalDict_['scores'])[np.array(evalDict_['type'])==2])}")
            print(f"detScores for type 1 avg: {np.mean(np.array(evalDict_['scores'])[np.array(evalDict_['type'])==1])}")
            print(f"detScores for type 0 avg: {np.mean(np.array(evalDict_['scores'])[np.array(evalDict_['type'])==0])}")
            print(f"detLogits: {len(evalDict_['logits'])}")
            print(f"type: {len(evalDict_['type'])}")
            print(f"detFeats: {len(evalDict_['feats'])}")
            print(f"all_filenames: {len(evalDict_['filenames'])}")
            print(f"#type 0: {np.sum(np.array(evalDict_['type'])==0)}")
            print(f"#type 1: {np.sum(np.array(evalDict_['type'])==1)}")
            print(f"#type 2: {np.sum(np.array(evalDict_['type'])==2)}")
	
        
        save_results_json(evalDict, os.path.join(save_dir, f'test/{saveFileName}'))

if __name__ == "__main__":
    argparse = parse_args()
    main(argparse)