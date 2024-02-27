import json
import numpy as np
import torch
import argparse
import scipy.stats as st
import tqdm
import os
import sys
from flowDet_mmdet.utils.gmmDet_utils import aurocScore, fit_gmms, gmm_uncertainty, summarise_performance
from flowDet_mmdet.utils.helper_utils import NumpyEncoder, draw_histogram
from flowDet_mmdet.flowDet_env import FLOWDET_FEAT_ROOT, FLOWDET_EXP_ROOT

def parse_args():
	parser = argparse.ArgumentParser(description='Test with Distance')
	# parser.add_argument('datasplit_type', help='to which data split produced by the code of which version: GMMDet or flowDet')
	parser.add_argument('detectorType', default = 'Yolov7', help='FRCNN or retinanet, Yolov7')
	parser.add_argument('--dataset', default = 'SpiritDataset', help='VOCDataset or CocoDataset, SpiritDataset')
	parser.add_argument('--unc', default = 'all', help='how to measure uncertainty? score, entropy, gmm, simple or all?')
	parser.add_argument('--saveNm', default = None, help='file name of saved features.')
	parser.add_argument('--iouThresh', default = 0.6, type = float, help='what is the cutoff iou for logits used to estimate class centres?')
	parser.add_argument('--scoreThresh', default = 0.7, type = float, help='what is the cutoff score for logits used to estimate class centres?')
	parser.add_argument('--numComp', default = None, type = int, help='do you want to test with a specific number of GMM components?')
	args = parser.parse_args()
	return args

args = parse_args()
datasplit_type = "GMMDet"
BASE_EXP_FOLDER = os.path.join(FLOWDET_EXP_ROOT, "GMMDet")

if args.unc == 'all':
	uncTypes = ['score', 'simple', 'gmm'] # 'entropy', 
else:
	uncTypes = [args.unc]

num_classes = 2

#load in the extracted feature data
BASE_FEAT_FOLDER = os.path.join(FLOWDET_FEAT_ROOT, "GMMDet")
feat_dir = f'{BASE_FEAT_FOLDER}/{args.detectorType}/associated/{args.dataset}'
test_feat_fn = ["real_eval_tst.json", "sim_eval_TMNF_tst.json"]
trn_feat_fn = "sim_eval_trn.json"
testData = {'type': [], 'logits': [], 'scores': [], "ious": []}
for fn in test_feat_fn:
	with open(f'{feat_dir}/test/{fn}', 'r') as f:
		json_data = json.load(f)
	
	# print(f"json_data['logits']: {len(json_data['logits'])}")
	# print(f"json_data['logits'][0]: {len(json_data['logits'][0])}")
	# print(f"json_data['logits'][0]: {json_data['logits'][0]}")
	# for logit_list in json_data['logits']:
	# 	print("logit_list: ", logit_list)
	testData['logits'].extend(json_data['logits'])
	# print("testData['logits']: ", len(testData['logits']))
		
	score = json_data['scores']
	testData['scores'].extend(score)
	t = json_data['type']
	testData['type'].extend(t)
	iou = json_data['ious']
	testData['ious'].extend(iou)
	
testLogits = np.array([np.array(a) for a in testData['logits']])
testLogits = testLogits[:, -2:]
print(f"testLogits: {testLogits.shape}")
testType = np.asarray(testData['type'])
print(f"testType: {testType.shape}")
testScores = np.asarray(testData['scores'])
print(f"testScores: {testScores.shape}")
testIoUs = np.asarray(testData['ious'])
print(f"testIoUs: {testIoUs.shape}")

#we want results in terms of AUROC, and TPR at 5%, 10% and 20% FPR
fprRates = [0.05, 0.1, 0.2]

allResults = {}
for unc in uncTypes:
	# create folder for saving results
	exp_dir = f'{BASE_EXP_FOLDER}/{args.detectorType}/{args.dataset}/{unc}/'	
	if not os.path.exists(exp_dir):
		os.makedirs(exp_dir)
	if unc == 'score':
		#correctly classified detections of known objects
		tpKnown = testScores[testType == 0]
		
		#open-set errors
		fpUnknown = testScores[testType == 2]

	
	elif unc == 'entropy':
		#faster r-cnn uses softmax
		if args.detectorType in ['FRCNN', 'FasterRCNN']:
			softmaxLayer = torch.nn.Softmax(dim = 1)
			tensorLogits = torch.Tensor(testLogits)
			softmaxScores = softmaxLayer(tensorLogits).cpu().detach().tolist()

		#retinanet uses sigmoid
		else:
			print('implement')
			exit()
		
		entropy = st.entropy(softmaxScores, axis = 1)
		entropy = np.asarray(entropy)

		#for entropy, a higher score means greater uncertainty. therefore we use the negative entropy, so that a lower score means greater uncertainty
		tpKnown = -entropy[testType == 0]
		fpUnknown = -entropy[testType == 2]

	#load in training and val logits for distance-based measures
	elif unc == 'simple' or unc == 'gmm':
		with open(f'{feat_dir}/train/{trn_feat_fn}', 'r') as f:
			trainData = json.load(f)

		# with open(f'{feat_dir}/val/{args.saveNm}.json', 'r') as f:
		# 	valData = json.load(f)

		trainLogits = np.array(trainData['logits']).squeeze(1)
		trainLogits = trainLogits[:, -2:]
		trainLabels = np.array(trainData['predictions'])
		trainScores = np.array(trainData['scores'])
		trainIoUs = np.array(trainData['ious'])

		valLogits = np.array(testData['logits'])
		valLogits = valLogits[:, -2:]
		valTypes = np.array(testData['type'])

		#fit distance-based models
		if unc == 'gmm':
			#find the number of components that gives best performance on validation data, unless numComp argument specified
			if args.numComp != None:
				gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = args.numComp)
			else:
				allAurocs = []
				nComps = [nI for nI in range(3, 16)]
				print('Finding optimal component number for the GMM')
				for nComp in tqdm.tqdm(nComps, total = len(nComps)):
					gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = nComp)
			
					gmmScores = gmm_uncertainty(valLogits, gmms)
					valTP = gmmScores[valTypes == 0]
					valFP = gmmScores[valTypes == 1]
					_, _, auroc = aurocScore(valTP, valFP)
					allAurocs += [auroc]

				allAurocs = np.array(allAurocs)
				bestIdx = np.argmax(allAurocs)
				preferredComp = nComps[bestIdx]

				print(f'Testing GMM with {preferredComp} optimal components')
				gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = preferredComp)

		else:
			print(f"trainLogits: {trainLogits.shape}")
			print(f"testLogits: {testLogits.shape}")
			gmms = fit_gmms(trainLogits, trainLabels, trainIoUs, trainScores, args.scoreThresh, args.iouThresh, num_classes, components = 1, covariance_type = 'spherical')
		
		gmmScores = gmm_uncertainty(testLogits, gmms)
		tpKnown = gmmScores[testType == 0]
		fpUnknown = gmmScores[testType == 2]

	else:
		print('That uncertainty measure has not been implemented. Check the args.unc input argument.')
		exit()
	
	scoreResults = summarise_performance(tpKnown, fpUnknown, fprRates=fprRates, printRes=True, methodName=f' with uncertainty {unc}')
	scoreResults["inData_nll"] = -tpKnown
	scoreResults["outData_nll"] = -fpUnknown
	print(f"saving histogram to {exp_dir}")
	draw_histogram(scoreResults, new_fn=f"{exp_dir}histo_{unc}_density.png", save=True, density=True)
	# allResults[unc] = scoreResults

	# save results
	with open(f'{exp_dir}real_sim_sep.json', 'w') as outFile:
		json.dump(scoreResults, outFile, cls=NumpyEncoder)
	