import os, sys
import numpy as np
import json
import logging
import pickle
import tqdm
import matplotlib.pyplot as plt
import torch

# logger = logging.getLogger(__name__)

def get_logger(name, file):
    if name in logging.root.manager.loggerDict:
        return logging.getLogger(name)

    s_handler = logging.StreamHandler(sys.stdout)
    s_handler.setLevel(logging.DEBUG)  # logging.{DEBUG, INFO, WARNING, ERROR, CRITICAL}
    formatter = logging.Formatter("%(asctime)s: %(levelname)8s - %(name)s:  %(message)s")
    s_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(s_handler)
    logger.setLevel(logging.DEBUG)

    if file is not None:
        print(f"adding logging file {file}")
        fileHandler = logging.FileHandler(file)
        logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)
    return logger

def unfreeze(model, part_name_lst, logger):
    for part in part_name_lst:
        logger.info(f"##### Setting sub-module({part}) as Trainbale!")
        if '.' in part:
            sub_part_hierarchy_list = part.split('.')
            module = getattr(model, sub_part_hierarchy_list[0])
            for hier_idx, sub_part in enumerate(sub_part_hierarchy_list[1:]):
                logger.info(f"##### Getting sub-module({sub_part}) from {sub_part_hierarchy_list[hier_idx]}!")
                module = getattr(module, sub_part)
                if hier_idx == len(sub_part_hierarchy_list[1:]) - 1:
                    for name, param in module.named_parameters():
                        param.requires_grad = True
                        logger.info(f"##### Setting sub-module({name}).requires_grad to {param.requires_grad}!")
        else:
            module = getattr(model, part)
            for name, param in module.named_parameters():
                param.requires_grad = True
                logger.info(f"##### Setting sub-module({name}).requires_grad to {param.requires_grad}!")

def setup_frcnn_trainable_model(model, trainable_parts, logger):
    if trainable_parts != "all":
        # freeze the whole model
        logger.info(f"##### Setting all params' requires_grad to False!")
        for name, param in model.named_parameters():
            param.requires_grad = False
            logger.info(f"##### Setting {name}.requires_grad to {param.requires_grad}!")

        # unfreeze specfic parts
        frcnn_part_dict = {"roi_head":["roi_head"], 
                           "bbox_head":["roi_head.bbox_head"], 
                           "fc_cls":["roi_head.bbox_head.fc_cls"], 
                           "fc_cls_reg":["roi_head.bbox_head.fc_cls", "roi_head.bbox_head.fc_reg"]}
        unfreeze(model, frcnn_part_dict[trainable_parts], logger)
    else:
        return 
    # else:
    #     raise NotImplementedError(f"trainable_parts ({trainable_parts}) are not implemented in the current version!")

def get_cls_counts(dataset):
    dim_logits = len(dataset.CLASSES) + 1
    cls_counter = np.zeros(dim_logits)
    print("Counting class num distribution:")
    for imIdx in tqdm.tqdm(range(len(dataset))):
        # imName = dataset.data_infos[imIdx]['filename']	
        cat_ids = dataset.get_cat_ids(imIdx)
        for cat in cat_ids:
            cls_counter[cat] += 1
    cls_counter[-1] = np.sum(cls_counter)/2. 
    return cls_counter

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def draw_histogram(scores_dict, suffix=None, new_fn=None, save=False, density=False):
	_, axs = plt.subplots(1, 1, figsize=(12, 10))
	# for idx, s_type in enumerate(scores_dict[0].keys()):
	axs.hist(scores_dict['inData_nll'], bins=30, histtype='barstacked', density=density, alpha=0.5, label="ID")
	axs.hist(scores_dict['outData_nll'], bins=30, histtype='barstacked', density=density, alpha=0.5, label="OOD")
	axs.set_title(f"Open Set Detection Histogram with AUROC {scores_dict['auroc']:.3f},\n" 
		 f"#ID({len(scores_dict['inData_nll'])}) with mean {np.mean(scores_dict['inData_nll']):.2f}, median {np.median(scores_dict['inData_nll']):.2f}\n"
		 f"#OOD({len(scores_dict['outData_nll'])}) with mean {np.mean(scores_dict['outData_nll']):.2f}, median {np.median(scores_dict['outData_nll']):.2f}")
	axs.grid()
	plt.ylabel('Counts of detections')
	plt.xlabel('NLLs of detections')
	leg = axs.legend(loc='best', fontsize="medium")
	leg.get_frame().set_alpha(0.5)
	
	if save:
		new_fn = new_fn.replace(".png", f"{suffix}.png") if suffix is not None else new_fn
		plt.savefig(new_fn, dpi=300)
		print()


def save_results_pickle(results, save_path):
    save_dir = os.path.split(save_path)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_path}.pickle", 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    print(f"Saving to {save_path}.pickle")


def save_results_json(results, save_path):
    jsonRes = json.dumps(results, cls=NumpyEncoder)
    save_dir = os.path.split(save_path)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(f"{save_path}.json", 'w')
    f.write(jsonRes)
    print(f"Saving to {save_path}.json")
    f.close()

def load_json_data(json_path, data_type="eval"):
    with open(json_path, 'r') as f:
        loaded_json_dict = json.load(f)

    Logits = np.array(loaded_json_dict['logits'])
    Scores = np.asarray(loaded_json_dict['scores'])
    IoUs = np.asarray(loaded_json_dict['ious'])
    if "feats" in loaded_json_dict.keys():
        Feats = np.array(loaded_json_dict["feats"])
    else:
        Feats = None
    if "filenames" in loaded_json_dict.keys():
        Filenames = np.array(loaded_json_dict["filenames"])
    else:
        Filenames = None
        
    if data_type == "eval":
        if "bboxes" in loaded_json_dict.keys():
            bboxes = np.array(loaded_json_dict["bboxes"])
        else:
            bboxes = None
        Type = np.asarray(loaded_json_dict['type'])
        return Type, Logits, Scores, IoUs, Feats, Filenames, bboxes
    elif data_type == "train": 
        Labels = np.asarray(loaded_json_dict['labels'])

        return Labels, Logits, Scores, IoUs, Feats

# function to append data to JSON
def append_to_json(new_data, root_key, new_key, filename):
    with open(filename,'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        file_data[root_key].append({new_key: new_data})
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, cls=NumpyEncoder)

def save_flow_model(model_save_dir, flow_module, epoch):
	print(f'Saving model to {model_save_dir}...')
	state = {
		'net': flow_module.state_dict(),
		'epoch': epoch,
		# 'means': means
	}
	os.makedirs(model_save_dir, exist_ok=True)
	torch.save(state, os.path.join(model_save_dir, str(epoch)+'.pt'))

def process_raw_ouput(dets, feat_ext_end_idx, num_logits):
    """
    function to process raw output from detector, which is used in feature extraction and detector testing
    """
    bboxes = dets[:, :4]
    scores = dets[:, 4]
    logits = dets[:, 5: 5+num_logits]
    scoresT = np.expand_dims(scores, axis=1)
    if feat_ext_end_idx == 5+num_logits:
        imDets = np.concatenate((logits, bboxes, scoresT), 1) # N*(#logits+4+1)
    else:
        feat = dets[:, 5+num_logits: feat_ext_end_idx]
        imDets = np.concatenate((logits, feat, bboxes, scoresT), 1) # N*(#logits+#feat+4+1)

    return imDets

def decompose_detData(detData, num_logits):
    det_vec_len = detData.shape[1] # (#logits+4+1) or (#logits+#feat+4+1)
    detBoxes = detData[:, -5:-1]
    detScores = detData[:, -1]
    detLogits = detData[:, :num_logits]
    if det_vec_len > 5+num_logits:
        detFeat = detData[:, num_logits: -5]
    else:
        detFeat = None

    return detBoxes, detScores, detLogits, detFeat