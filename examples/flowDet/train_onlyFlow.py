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
# from mmengine.config import Config
from flowDet_mmdet.utils.functional_utils import (get_latest_checkpoint,
												test_flow_NLL_v2,
												init_flow_module_v2,
												setup_dataloader)
from flowDet_mmdet.utils.training_utils import (train_flow_epoch_v2,
												setup_flow_opt)
from flowDet_mmdet.utils.helper_utils import (append_to_json,
					      					get_logger,
											draw_histogram, 
											save_flow_model)
from flowDet_mmdet.flowDet_env import FLOWDET_FEAT_ROOT, FLOWDET_CONFIG_ROOT, FLOWDET_EXP_ROOT

# s_handler = logging.StreamHandler(sys.stdout)
# s_handler.setLevel(logging.DEBUG)  # logging.{DEBUG, INFO, WARNING, ERROR, CRITICAL}
# formatter = logging.Formatter("%(asctime)s: %(levelname)8s - %(name)s:  %(message)s")
# s_handler.setFormatter(formatter)
# logger = logging.getLogger(__name__)
# logger.addHandler(s_handler)
# logger.setLevel(logging.DEBUG)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def parse_args():
	parser = argparse.ArgumentParser(description='train only flow and test')
	parser.add_argument('--config', help='flow training config')
	parser.add_argument('--feat_fn', default = None, help='file name of saved features, used for GMMs or NFs.')
	parser.add_argument('--feat_type', default = None, help='type of features for training the flow, logits or msfeats')
	parser.add_argument('--only_eval', action='store_true', help='whether to ONLY eval the trained model on test set.')
	args = parser.parse_args()
	return args

def flow_folder_name_generator(params_dict):
	prior_params = ""
	if params_dict['prior_type'] in ["gmm", "gmm_cls", "gmm_cls_ib"]:
		prior_params += f"{params_dict['base_n_modes']}*{params_dict['base_loc_scale']}"
		prior_params += "_"
	elif params_dict['prior_type'] in ["resampled", "resampled_cls", "resampled_v2_cls", "resampled_v3_cls", "resampled_cls_ib", "resampled_v2_cls_ib", "resampled_v3_cls_ib"]:
		prior_params += f"T{params_dict['base_T']}_eps{params_dict['base_eps']}_{params_dict['base_a_hidden_layers']}*{params_dict['base_a_hidden_units']}_Ini0{params_dict['base_init_zeros']}_dp{params_dict['base_dropout']}"
		prior_params += "_"

	if "_ib" in params_dict['prior_type']:
		if prior_params == "": prior_params = "_"
		else:
			prior_params = f"beta{params_dict['ib_beta']}_sig{params_dict['ib_sigma']}_" + prior_params

	if params_dict['early_stop_tolerance'] == 0:
		stopping_criteria = f"max{params_dict['max_epochs']}"
		params_dict['val_freq'] = int(params_dict['max_epochs']/2.)
	else:
		stopping_criteria = f"es{params_dict['early_stop_tolerance']}*inter{params_dict['val_freq']}*max{params_dict['max_epochs']}"

	if params_dict['flow_type'] == "residual":
		flow_params = f"LipConst{params_dict['lipschitz_const']}_"
		flow_params += f"{params_dict['blocks']}*[{params_dict['num_layers_st_net']}*{params_dict['hidden_dim']}]_"

	elif params_dict['flow_type'] == "rnvp":
		flow_params = f"{params_dict['permutation']}_{params_dict['scale_map']}_"
		flow_params += f"{params_dict['blocks']}*[{params_dict['num_layers_st_net']}*{params_dict['hidden_dim']}]_"
	
	elif params_dict['flow_type'] == "nsf_ar":
		flow_params = f"bins{params_dict['num_bins']}_"
		flow_params += f"{params_dict['blocks']}*[{params_dict['num_layers_st_net']}*{params_dict['hidden_dim']}]_"

	folder_suffix = (f"{prior_params}"
					f"{flow_params}"
					f"{params_dict['optm']}_"
					f"{params_dict['lr']}_"
					f"bs{params_dict['batch_size']}_"
					f"{stopping_criteria}_"
					f"gnorm{params_dict['max_norm']}"
					)
	return folder_suffix

def crete_logging_file(log_folder, only_eval):

	# time_str = time.ctime().replace(" ", "")
	time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime())
	if only_eval:
		log_file = f"{log_folder}/{time_str}_eval_only.txt"
	else:
		log_file = f"{log_folder}/{time_str}.txt"
	os.system(f"echo 'Host:' $(hostname) | tee -a {log_file}")
	os.system(f"echo 'Conda:' $(which conda) | tee -a {log_file}")
	os.system(f"echo $(pwd) | tee -a {log_file}")
	os.system(f"echo 'Version:' $(VERSION) | tee -a {log_file}")
	os.system(f"echo 'Git diff:'| tee -a {log_file}")
	os.system(f"git diff | tee -a {log_file}")
	os.system(f"nvidia-smi| tee -a {log_file}")

	log_file = f"{log_folder}/{time_str}.log"
	return log_file

def main(args):
	print(f"FLOWDET_CONFIG_ROOT: {FLOWDET_CONFIG_ROOT}")
	config_path = osp.join(FLOWDET_CONFIG_ROOT, args.config)
	print(f"Loading config file: {config_path}")
	cfg = Config.fromfile(config_path)
	# to set up as a flowDet exp logging folder
	BASE_EXP_FOLDER = FLOWDET_EXP_ROOT
	# to decide features extracted based on code of flowDet or GMMDet
	BASE_FEAT_FOLDER = FLOWDET_FEAT_ROOT

	# prepare folders
	params_dict = cfg.flow_params
	flow_type = params_dict["flow_type"]

	detector_type = cfg.model.type
	feat_fn = args.feat_fn
	torch.manual_seed(params_dict['random_seed'])
	np.random.seed(params_dict['random_seed'])
	random.seed(params_dict['random_seed'])
	# for logging the training and saving the model
	log_folder_suffix = flow_folder_name_generator(params_dict)
	if params_dict['base_learn_mean_var']:
		prior_status = "learned_prior"
	else:
		prior_status = "fixed_prior"
	log_folder = f"{BASE_EXP_FOLDER}/{detector_type}/{cfg.dataset_type}/{flow_type}/{feat_fn}/{params_dict['prior_type']}/{prior_status}/{log_folder_suffix}_rnd{params_dict['random_seed']}"
	if "msFeats" in feat_fn:
		if args.feat_type == "logits":
			log_folder = f"{BASE_EXP_FOLDER}/{detector_type}/{cfg.dataset_type}/{flow_type}/{feat_fn}_logits/{params_dict['prior_type']}/{prior_status}/{log_folder_suffix}_rnd{params_dict['random_seed']}"
	
	if args.only_eval:
		assert os.path.isdir(log_folder), f"{log_folder} doesn't exist!"

	os.makedirs(log_folder, exist_ok=True)
	log_file = crete_logging_file(log_folder, args.only_eval)
	logger = get_logger("train_onlyFLow", log_file)
	# for saving the model
	save_model_dir = osp.join(log_folder, "saved_model")
	os.makedirs(save_model_dir, exist_ok=True)
	# fig_save_dir = osp.join(log_folder, "results_figs")
	# os.makedirs(fig_save_dir, exist_ok=True)
	# for recording validation results during training
	val_json_file = osp.join(log_folder, "training_logging_res.json")
	json_key = "training_logging_res"
	with open(val_json_file, "w") as json_file:
		init_dict = {json_key:[]}
		json.dump(init_dict, json_file)

	# dump config
	# config_fn = osp.join(log_folder, osp.basename(args.config))
	# print(f"dumping: {config_fn}")
	# cfg.dump(config_fn)

	# get data and prepare data set
	# for loading saved features extracted from the detector
	saved_feat_dir = f"{BASE_FEAT_FOLDER}/{detector_type}/associated/{cfg.dataset_type}"
	tst_res_path = f"{saved_feat_dir}/test/{feat_fn}.json"
	test_dataloader, testTypes = setup_dataloader(tst_res_path, batch_size=params_dict["batch_size"], data_type="eval", feat=args.feat_type, log_file=log_file)
	logger.info(f"test #type 0: {np.sum(testTypes==0)}")
	logger.info(f"test #type 1: {np.sum(testTypes==1)}")
	logger.info(f"test #type 2: {np.sum(testTypes==2)}")
	
	if not args.only_eval:
		val_res_path = f"{saved_feat_dir}/val/{feat_fn}.json"
		val_dataloader, valTypes = setup_dataloader(val_res_path, batch_size=params_dict["batch_size"], data_type="eval", feat=args.feat_type, log_file=log_file)
		logger.info(f"val #type 0: {np.sum(valTypes==0)}")
		logger.info(f"val #type 1: {np.sum(valTypes==1)}")
		logger.info(f"val #type 2: {np.sum(valTypes==2)}")
		trn_res_path = f"{saved_feat_dir}/train/{feat_fn}.json"
		filter_dict = dict(iouThresh=0.6, scoreThresh=0.7) # to ensure high quality training data: iouThresh=0.6, scoreThresh=0.7, similar to GMMdet impl
		train_dataloader = setup_dataloader(trn_res_path, 
											batch_size=params_dict["batch_size"], 
											data_type="train", 
											feat=args.feat_type, 
											log_file=log_file,
											filter_dict=filter_dict)

	# init flow module
	# flow_module = init_flow_module(params_dict, device=device)
	flow_module = init_flow_module_v2(params_dict)
	# Move model on GPU if available
	use_gpu = torch.cuda.is_available()
	device = torch.device('cuda' if use_gpu else 'cpu')
	flow_module = flow_module.to(device)
	flow_module = flow_module.float()
	
	# init optimizer
	optimizer = setup_flow_opt(params_dict, flow_module)

	# start training
	print("hyper-params:", log_folder_suffix)
	prior_type = params_dict['prior_type']
	grad_max_norm = params_dict['max_norm']
	early_stop_iter = params_dict['early_stop_tolerance']
	m_name = f"{params_dict['flow_type']}_{prior_type}"
	best_auroc = 0.0
	best_auroc_counter = 0
	if not args.only_eval:
		for epoch in range(0, params_dict['max_epochs']):
			# scoreResults dict has fields: "inData_nll", "outData_nll", "auroc", "epoch", "val_auroc"
			logger.info('Epoch: %d' % epoch)

			# Save checkpoint
			if (epoch % params_dict['val_freq'] == 0):
				logger.info("##### On Val Set: #####")
				val_scoreResults = test_flow_NLL_v2(flow_module, 
													val_dataloader, 
													valTypes, 
													method_name=m_name, 
													return_nlls=False, 
													test_or_eval='eval',
													log_file=log_file,
													tp_fp_idxes=[0, 1]) 
				
				if early_stop_iter != 0:
					if val_scoreResults['auroc'] > best_auroc or val_scoreResults['auroc'] == best_auroc:
						best_auroc_counter = 0
						best_auroc = val_scoreResults['auroc'] 
					else:
						best_auroc_counter += 1

					append_to_json(val_scoreResults, root_key=json_key, new_key=str(epoch), filename=val_json_file)
					logger.info(f"saving val results to {val_json_file} with current best_auroc: {best_auroc:.3f}"
							f" and best_auroc_counter: {best_auroc_counter}/{early_stop_iter}.")
					if best_auroc_counter > early_stop_iter:
						save_flow_model(save_model_dir, flow_module, epoch)
						break
				else:
					logger.info(f"Because early_stop_iter is {early_stop_iter}, we train the model with epochs {params_dict['max_epochs']}.")
			
			# train_flow_epoch(flow_module, train_dataloader, optimizer, prior_type, grad_max_norm)
			train_flow_epoch_v2(flow_module, params_dict, train_dataloader, optimizer, grad_max_norm, log_file=log_file)
			if early_stop_iter == 0 and epoch == (params_dict['max_epochs'] - 1):
				save_flow_model(save_model_dir, flow_module, epoch)
	
	else:
		weights_folder = save_model_dir
		latest_cp = get_latest_checkpoint(weights_folder)
		assert latest_cp is not None, f"latest_cp from {weights_folder} should be None, now is {latest_cp}"
		logger.info(f"Loading weights from {latest_cp}")
		flow_module.load_state_dict(torch.load(latest_cp)['net'])
		epoch = torch.load(latest_cp)['epoch']

	logger.info("Evaluation on Test set after training:")
	logger.info("##### On Test Set: #####")
	scoreResults = test_flow_NLL_v2(flow_module, 
									test_dataloader, 
									testTypes, 
									method_name=m_name, 
									prior_type=prior_type,
									test_or_eval='test',
									return_nlls=True, 
									log_file=log_file,
									tp_fp_idxes=[0, 2]) 
	append_to_json(scoreResults, root_key=json_key, new_key=f"final_test_ep{epoch}", filename=val_json_file)
	draw_histogram(scoreResults, new_fn=log_folder+f"/test_histo_auroc{scoreResults['auroc']:.4f}_ep{epoch}.png", save=True, density=True)
	# print(f"saving histogram to {fig_save_dir}")

if __name__ == "__main__":
	args = parse_args()
	main(args)