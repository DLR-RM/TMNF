# based on: https://github.com/stanislavfort/exploring_the_limits_of_OOD_detection/blob/main/ViT_for_strong_near_OOD_detection.ipynb
import json
from posixpath import basename
import numpy as np
import argparse
import random
import time
import os
import os.path as osp
import torch
from mmcv import Config
from flowDet_mmdet.utils.gmmDet_utils import summarise_performance
from flowDet_mmdet.utils.functional_utils import (get_latest_checkpoint,
												test_flow_NLL_v2,
												init_flow_module_v2,
												setup_dataloader)
from flowDet_mmdet.utils.training_utils import (train_flow_epoch_v2,
												setup_flow_opt)
from flowDet_mmdet.utils.helper_utils import (load_json_data,
					      					append_to_json,
					      					get_logger,
											draw_histogram, 
											save_flow_model)
from flowDet_mmdet.flowDet_env import FLOWDET_FEAT_ROOT, FLOWDET_CONFIG_ROOT, FLOWDET_EXP_ROOT	

from sklearn.metrics import roc_auc_score

def maha_distance(xs,cov_inv_in,mean_in,norm_type=None):
  diffs = xs - mean_in.reshape([1,-1])

  second_powers = np.matmul(diffs,cov_inv_in)*diffs

  if norm_type in [None,"L2"]:
    return np.sum(second_powers,axis=1)
  elif norm_type in ["L1"]:
    return np.sum(np.sqrt(np.abs(second_powers)),axis=1)
  elif norm_type in ["Linfty"]:
    return np.max(second_powers,axis=1)
  
def get_rmd_md_scores(
    indist_train_embeds_in,
    indist_train_labels_in,
    indist_test_embeds_in,
    outdist_test_embeds_in,
    subtract_mean = True,
    normalize_to_unity = True,
    indist_classes = 100,
    norm_name = "L2",
    ):
  
	# storing the replication results
	maha_intermediate_dict = dict()
	
	description = ""
	
	all_train_mean = np.mean(indist_train_embeds_in,axis=0,keepdims=True)

	indist_train_embeds_in_touse = indist_train_embeds_in
	indist_test_embeds_in_touse = indist_test_embeds_in
	outdist_test_embeds_in_touse = outdist_test_embeds_in

	if subtract_mean:
		indist_train_embeds_in_touse -= all_train_mean
		indist_test_embeds_in_touse -= all_train_mean
		outdist_test_embeds_in_touse -= all_train_mean
		description = description+" subtract mean,"

	if normalize_to_unity:
		indist_train_embeds_in_touse = indist_train_embeds_in_touse / np.linalg.norm(indist_train_embeds_in_touse,axis=1,keepdims=True)
		indist_test_embeds_in_touse = indist_test_embeds_in_touse / np.linalg.norm(indist_test_embeds_in_touse,axis=1,keepdims=True)
		outdist_test_embeds_in_touse = outdist_test_embeds_in_touse / np.linalg.norm(outdist_test_embeds_in_touse,axis=1,keepdims=True)
		description = description+" unit norm,"

	#full train single fit
	mean = np.mean(indist_train_embeds_in_touse,axis=0)
	cov = np.cov((indist_train_embeds_in_touse-(mean.reshape([1,-1]))).T)

	eps = 1e-8
	cov_inv = np.linalg.inv(cov)

	#getting per class means and covariances
	class_means = []
	class_cov_invs = []
	class_covs = []
	for c in range(indist_classes):
		num_each_class = np.sum(indist_train_labels_in == c)
		print(f"there are {num_each_class} data points for class {c}")
		if num_each_class == 0:
			mean_now = np.zeros_like(mean)
			cov_now = np.identity(mean.shape[0])
		else:
			mean_now = np.mean(indist_train_embeds_in_touse[indist_train_labels_in == c],axis=0)
			cov_now = np.cov((indist_train_embeds_in_touse[indist_train_labels_in == c]-(mean_now.reshape([1,-1]))).T)
		class_covs.append(cov_now)
		# print(c)

		eps = 1e-8
		cov_inv_now = np.linalg.inv(cov_now+eps)

		class_cov_invs.append(cov_inv_now)
		class_means.append(mean_now)

	#the average covariance for class specific
	mean_cov = np.mean(np.stack(class_covs,axis=0),axis=0)
	class_cov_invs = [np.linalg.inv(mean_cov+eps)]*len(class_covs)

	maha_intermediate_dict["class_cov_invs"] = class_cov_invs
	maha_intermediate_dict["class_means"] = class_means
	maha_intermediate_dict["cov_inv"] = cov_inv
	maha_intermediate_dict["mean"] = mean

	out_totrain = maha_distance(outdist_test_embeds_in_touse,cov_inv,mean,norm_name)
	in_totrain = maha_distance(indist_test_embeds_in_touse,cov_inv,mean,norm_name)

	out_totrainclasses = [maha_distance(outdist_test_embeds_in_touse,class_cov_invs[c],class_means[c],norm_name) for c in range(indist_classes)]
	in_totrainclasses = [maha_distance(indist_test_embeds_in_touse,class_cov_invs[c],class_means[c],norm_name) for c in range(indist_classes)]

	out_scores = np.min(np.stack(out_totrainclasses,axis=0),axis=0)
	in_scores = np.min(np.stack(in_totrainclasses,axis=0),axis=0)

	# md_scores = np.concatenate([out_scores,in_scores],axis=0)

	rmd_out_scores = out_scores - out_totrain
	rmd_in_scores = in_scores - in_totrain


	onehots = np.array([1]*len(out_scores) + [0]*len(in_scores))
	# rmd_scores = np.concatenate([rmd_out_scores,rmd_in_scores],axis=0)

	return out_scores, in_scores, rmd_out_scores, rmd_in_scores

def parse_args():
	parser = argparse.ArgumentParser(description='train only flow and test')
	parser.add_argument('--config', help='flow training config')
	parser.add_argument('--datasplit_type', type=str,  help='to which data split produced by the code of which version: GMMDet or flowDet, flowDet_msfeats')
	parser.add_argument('--feat_fn', default = None, help='file name of saved features, used for GMMs or NFs.')
	args = parser.parse_args()
	return args

def read_data(cfg, saved_feat_dir, feat_fn):
	if cfg.dataset_type == "SpiritDataset":
		test_feat_fn = ["real_eval_tst.json", "sim_eval_TMNF_tst.json"]
		trn_feat_fn = "sim_eval_trn.json"
		testData = {'type': [], 'logits': [], 'scores': [], "ious": []}
		for fn in test_feat_fn:
			with open(f'{saved_feat_dir}/test/{fn}', 'r') as f:
				json_data = json.load(f)
			
			testData['logits'].extend(json_data['logits'])
			testData['scores'].extend(json_data['scores'])
			testData['type'].extend(json_data['type'])
			testData['ious'].extend(json_data['ious'])
			
		testLogits = np.array([np.array(a) for a in testData['logits']])
		testLogits = testLogits[:, -2:]
		testTypes = np.asarray(testData['type'])
		with open(f'{saved_feat_dir}/train/{trn_feat_fn}', 'r') as f:
			trainData = json.load(f)
		trainLogits = np.array(trainData['logits']).squeeze(1)
		trainLogits = trainLogits[:, -2:]
		trainLabels = np.array(trainData['predictions'])
		num_cls = trainLogits.shape[1] 
	else:
		# test set
		tst_res_path = f"{saved_feat_dir}/test/{feat_fn}.json"
		testTypes, testLogits, Scores, Ious, testFeats, Filenames, bboxes = load_json_data(tst_res_path, data_type="eval")
		print(f"testTypes: {testTypes.shape}")
		print(f"testLogits: {testLogits.shape}")
		# training set
		trn_res_path = f"{saved_feat_dir}/train/{feat_fn}.json"
		trainLabels, trainLogits, trnScores, trnIoUs, trainFeats = load_json_data(trn_res_path, data_type="train")
		filter_dict = dict(iouThresh=0.6, scoreThresh=0.7) # to ensure high quality training data: iouThresh=0.6, scoreThresh=0.7, similar to GMMdet impl
		if filter_dict is not None:
			#mask for high iou and high conf
			mask = (trnIoUs >= filter_dict['iouThresh'])*(trnScores >= filter_dict['scoreThresh'])
			trainLogits = trainLogits[mask]
			trainLabels = trainLabels[mask]
		print(f"trainLogits: {trainLogits.shape}")
		print(f"trainLabels: {trainLabels.shape}")
		num_cls = trainLogits.shape[1] - 1
	
	return testLogits, testTypes, trainLogits, trainLabels, num_cls

def main(args):
	config_path = osp.join(FLOWDET_CONFIG_ROOT, args.config)
	cfg = Config.fromfile(config_path)
	# to decide features extracted based on code of flowDet or GMMDet
	BASE_FEAT_FOLDER = osp.join(FLOWDET_FEAT_ROOT, args.datasplit_type) 

	detector_type = cfg.model.type
	feat_fn = args.feat_fn
	torch.manual_seed(0)
	np.random.seed(0)
	random.seed(0)

	# get data and prepare data set
	saved_feat_dir = f"{BASE_FEAT_FOLDER}/{detector_type}/associated/{cfg.dataset_type}"
	print(f"saved_feat_dir: {saved_feat_dir}")
		
	testLogits, testTypes, trainLogits, trainLabels, num_cls = read_data(cfg, saved_feat_dir, feat_fn)

	tp_fp_idxes = [0, 2]
	indist_test_embeds_in = testLogits[testTypes == tp_fp_idxes[0]]
	outdist_test_embeds_in = testLogits[testTypes == tp_fp_idxes[1]]
	md_out_scores, md_in_scores, rmd_out_scores, rmd_in_scores = \
	get_rmd_md_scores(
				trainLogits,
				trainLabels,
				indist_test_embeds_in,
				outdist_test_embeds_in,
				subtract_mean = False,
				normalize_to_unity = False,
				indist_classes = num_cls,
				norm_name = "L2",
				)
	log_folder = "/home_local/feng_ji/Experiments/flowDet_exp/exp_logs/flowDet/Yolov7/SpiritDataset/mahalanobis_distance"
	scores_dict = {"md":[md_in_scores, md_out_scores], "rmd":[rmd_in_scores, rmd_out_scores]}
	for method_name in ["md", "rmd"]:
		# results file
		res_json_file = osp.join(log_folder, f"{method_name}.json")
		json_key = "training_logging_res"
		with open(res_json_file, "w") as json_file:
			init_dict = {json_key:[]}
			json.dump(init_dict, json_file)
		fprRates = [0.05, 0.1, 0.2]
		id_dist, ood_dist = scores_dict[method_name]
		# filter out nan
		id_nan_flag = np.isnan(id_dist) 
		ood_nan_flag = np.isnan(ood_dist) 
		id_dist = id_dist[np.logical_not(id_nan_flag)]
		ood_dist = ood_dist[np.logical_not(ood_nan_flag)]
		print(f"id_dist after filtering out {np.sum(id_nan_flag)} Nans: {np.mean(id_dist):.3f}")
		print(f"ood_dist after filtering out {np.sum(ood_nan_flag)} Nans: {np.mean(ood_dist):.3f}")
		scoreResults = summarise_performance(-id_dist, -ood_dist, fprRates=fprRates, printRes=True, methodName=method_name)
		scoreResults["inData_nll"] = id_dist
		scoreResults["outData_nll"] = ood_dist
		append_to_json(scoreResults, root_key=json_key, new_key=f"final_test_ep199", filename=res_json_file)
		# draw_histogram(scoreResults, new_fn=log_folder+f"/test_histo_auroc{scoreResults['auroc']:.4f}.png", save=True, density=True)

if __name__ == "__main__":
	args = parse_args()
	main(args)