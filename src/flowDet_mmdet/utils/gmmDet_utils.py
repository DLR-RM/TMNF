# copied from https://github.com/dimitymiller/openset_detection
import sklearn.mixture as sm
import numpy as np
import sklearn.metrics
import scipy.stats
import json 
import tqdm
import torch
from mmdet.datasets import build_dataset
from flowDet_mmdet.utils.helper_utils import process_raw_ouput, decompose_detData, get_logger

def run_inference(model, data_loader, confThresh, maxOneDetOneRP=True):    
    """
    function to process raw output from detector, which is used in feature extraction and detector testing
    """
    allResults = {}

    for i, data in enumerate(tqdm.tqdm(data_loader, total=len(data_loader.dataset))):   
        imgName = data_loader.dataset.data_infos[i]['filename']
        allResults[imgName] = []
        all_detections = []
        
        with torch.no_grad():
            # see format from bbox2result() in flowDet_mmdet/mmdetection/mmdet/core/bbox/transforms.py
            results = model(return_loss=False, rescale=True, **data)[0]

        num_cls = len(results) # class number WITHOUT bg cls
        for cls_idx in range(num_cls):
            imDets = results[cls_idx] # det_bboxes of class cls_idx after NMS
            if len(imDets) > 0:
                num_logits = model.module.roi_head.bbox_head.num_classes+1
                # imDets = process_raw_ouput(dets, feat_ext_end_idx, num_logits)

                # winning class must be class j for this detection to be considered valid
                # this will filter out the cases where multiple bboxes valid within one ROI
                logits = imDets[:, 5:5+num_logits]
                scores = imDets[:, 4]
                mask = None
                if maxOneDetOneRP:
                    mask = np.argmax(logits, axis=1)==cls_idx
                    if np.sum(mask) > 0:
                        imDets = imDets[mask]
                        scores = scores[mask]
                    else:
                        continue
                
                # filter out det with score < confThresh
                if confThresh > 0.:
                    mask = scores >= confThresh
                    if np.sum(mask) > 0:
                        imDets = imDets[mask]
                    else:
                        continue
                
                all_detections.append(imDets)

        if len(all_detections) == 0:
            continue
        else:
            #remove doubled-up detections -- this shouldn't really happen
            all_detections = np.concatenate(all_detections, axis=0)
            detections, _ = np.unique(all_detections, return_index = True, axis = 0)

        allResults[imgName] = detections.tolist()
        
    return allResults

#function used to calculate IoU between boxes, taken from: https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
def iouCalc(boxes1, boxes2):
	def run(bboxes1, bboxes2):
		x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
		x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
		xA = np.maximum(x11, np.transpose(x21))
		yA = np.maximum(y11, np.transpose(y21))
		xB = np.minimum(x12, np.transpose(x22))
		yB = np.minimum(y12, np.transpose(y22))
		interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
		boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
		boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
		iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
		return iou
	return run(boxes1, boxes2)

def assign_ID(dataHolder,
                detGTIous, 
                sorted_idxes, 
                detAssociated, 
                gtKnownAssociated=None,
                knownLabels=None, 
                detPredict=None, 
                iouThresh=0.5):
    detIdx_list = []
    for detIdx in sorted_idxes:
        #if all gt have been associated, move on
        if np.sum(gtKnownAssociated) == len(gtKnownAssociated):
            break

        ious = detGTIous[detIdx]
        #sort from greatest to lowest overlap
        sorted_iouIdxs = np.argsort(ious)[::-1]
        
        for iouIdx in sorted_iouIdxs:
            #check this gt object hasn't already been detected
            if gtKnownAssociated[iouIdx] == 1:
                continue

            if ious[iouIdx] >= iouThresh:
                #associating this detection and gt object
                gtKnownAssociated[iouIdx] = 1
                detAssociated[detIdx] = 1
                detIdx_list.append(detIdx)

                dataHolder['ious'] += [ious[iouIdx]]
                gtlabel = knownLabels[iouIdx]
                if detPredict[detIdx] == gtlabel:
                    #known class was classified correctly
                    dataHolder['type'] += [0]
                else:
                    #known class was misclassified
                    dataHolder['type'] += [1]
                break
            else:
                #doesn't have an iou greater than 0.5 with anything
                break
    return detIdx_list

def assign_OOD(dataHolder,
                detGTIous, 
                sorted_idxes, 
                detAssociated, 
                iouThresh=0.5):
    detIdx_list = []
    for detIdx in sorted_idxes:
        #if the detection has already been associated, skip it
        if detAssociated[detIdx] == 1:
            continue

        ious = detGTIous[detIdx]
        #sort from greatest to lowest overlap
        sorted_iouIdxs = np.argsort(ious)[::-1]
        for iouIdx in sorted_iouIdxs:
            if ious[iouIdx] >= iouThresh:
                #associating this detection and gt object
                detAssociated[detIdx] = 1
                detIdx_list.append(detIdx)
                dataHolder['ious'] += [ious[iouIdx]]
                dataHolder['type'] += [2]
                break
            else:
                #doesn't have an iou greater than 0.5 with anything
                break
    return detIdx_list

# used to assign types to detections: background (3), known class correctly predicted (0), known class incorrectly predicted (1), unknown class (2)
def assign_types(dataHolder, dets, gt, clsCutoff, iouThresh=0.5):
    # gt
    gtBoxes = gt['bboxes']
    gtLabels = gt['labels']
    
    # pred
    img_fns = dets['filenames']
    detPredict = dets['predictions']
    detBoxes = dets['boxes']
    detScores = dets['scores']
    detLogits = dets['logits']
    if "feats" in dets.keys():
        detFeats = dets['feats']
    else:
        detFeats = None

    # set up Openset condition
    knownBoxes = gtBoxes[gtLabels < clsCutoff]
    knownLabels = gtLabels[gtLabels < clsCutoff]
    unknownBoxes = gtBoxes[gtLabels > clsCutoff]
    unknownLabels = gtLabels[gtLabels > clsCutoff]
    # print(f"#GTknownBoxes: {len(knownBoxes)}")
    # print(f"#GTunknownBoxes: {len(unknownBoxes)}")

    #sort from most confident to least
    # sorted_scores = np.sort(detScores)[::-1]
    sorted_idxes = np.argsort(detScores)[::-1]
    detScores = np.expand_dims(detScores, axis=1)
    detAssociated = [0]*len(detScores)
    gtKnownAssociated = [0]*len(knownBoxes)

    #first, we check if the detection has fallen on a known class
    #if an IoU > iouThresh with a known class --> it is detecting that known class
    if len(knownBoxes) > 0:
        detKnownGTIous = iouCalc(detBoxes, knownBoxes)
        detIdx_list = assign_ID(dataHolder,
                                detKnownGTIous, 
                                sorted_idxes, 
                                detAssociated, 
                                gtKnownAssociated=gtKnownAssociated,
                                knownLabels=knownLabels, 
                                detPredict=detPredict, 
                                iouThresh=iouThresh)
        
        # print(f"#detIdx_list: {len(detIdx_list)}")
        # print(f"detIdx_list: {detIdx_list}")
        # print(f"detLogits: {len(detLogits)}")
        # print(f"detScores: {len(detScores)}")
        # print(f"img_fns: {len(img_fns)}")
        # print(f"detFeats: {len(detFeats)}")
        for detIdx in detIdx_list:
            dataHolder['logits'] += [list(detLogits[detIdx])]
            dataHolder['scores'] += [list(detScores[detIdx])]
            dataHolder['filenames'] += [img_fns[detIdx]]
            dataHolder['bboxes'] += [list(detBoxes[detIdx])]
            if detFeats is not None:
                dataHolder['feats'] += [list(detFeats[detIdx])]
            
    #all detections have been associated
    if np.sum(detAssociated) == len(detAssociated):
        return dataHolder

    ### Next, check if the detection overlaps an ignored gt known object - these detections are ignored
    #also check ignored gt known objects
    if len(gt['bboxes_ignore']) > 0:
        igBoxes = gt['bboxes_ignore']
        igIous = iouCalc(detBoxes, igBoxes)
        for detIdx in sorted_idxes:
            if detAssociated[detIdx] == 1:
                continue

            ious = igIous[detIdx]
            #sort from greatest to lowest overlap
            sorted_iouIdxs = np.argsort(ious)[::-1]

            for iouIdx in sorted_iouIdxs:
                if ious[iouIdx] >= iouThresh:
                    #associating this detection and gt object
                    detAssociated[detIdx] = 1
                break

    #all detections have been associated
    if np.sum(detAssociated) == len(detAssociated):
        return dataHolder

    #if an IoU > 0.5 with an unknown class (but not any known classes) --> it is detecting the unknown class
    newDetAssociated = detAssociated
    if len(unknownBoxes) > 0:
        unknownIous = iouCalc(detBoxes, unknownBoxes)
        detIdx_list = assign_OOD(dataHolder,
                                unknownIous, 
                                sorted_idxes, 
                                detAssociated,  
                                iouThresh=iouThresh)
        for detIdx in detIdx_list:
            dataHolder['logits'] += [list(detLogits[detIdx])]
            dataHolder['scores'] += [list(detScores[detIdx])]
            dataHolder['filenames'] += [img_fns[detIdx]]
            dataHolder['bboxes'] += [list(detBoxes[detIdx])]
            if detFeats is not None:
                dataHolder['feats'] += [list(detFeats[detIdx])]

    if np.sum(detAssociated) == len(detAssociated):
        return dataHolder

    #otherwise remaining detections are all background detections
    for detIdx, assoc in enumerate(detAssociated):
        if not assoc:
            dataHolder['scores'] += [list(detScores[detIdx])]
            dataHolder['logits'] += [list(detLogits[detIdx])]
            dataHolder['filenames'] += [img_fns[detIdx]]
            dataHolder['bboxes'] += [list(detBoxes[detIdx])]
            dataHolder['type'] += [3]
            dataHolder['ious'] += [0]
            detAssociated[detIdx] = 1
            if detFeats is not None:
                dataHolder['feats'] += [list(detFeats[detIdx])]

    if np.sum(detAssociated) != len(detAssociated):
        print("THERE IS A BIG ASSOCIATION PROBLEM")
        exit()
    
    # print("Associtating raw predictions")
    # print(f"detScores: {len(dataHolder['scores'])}")
    # print(f"detLogits: {len(dataHolder['logits'])}")
    # print(f"type: {len(dataHolder['type'])}")
    # print(f"detFeats: {len(dataHolder['feats'])}")
    # print(f"all_filenames: {len(dataHolder['filenames'])}")
    return dataHolder

def assign_eval(ds_dict, raw_res_pth, num_classes, scoreThresh=0.2, iouThresh=0.5, gtLabel_mapping=None):
    testDataset = build_dataset(ds_dict)
    print(ds_dict)
    lenDataset = len(testDataset)
    num_logits = num_classes + 1
    all_filenames = []

    with open(raw_res_pth, 'r') as f:
        testDets = json.load(f)

    allData = {'logits': [], 'bboxes': [], 'scores': [], 'ious': [], 'type': [], 'feats':[], 'filenames':[]}
    for imIdx, im in enumerate(tqdm.tqdm(testDataset, total = lenDataset)):
        imName = testDataset.data_infos[imIdx]['filename']
        detData = np.asarray(testDets[imName])
        gtData = testDataset.get_ann_info(imIdx)
        if gtLabel_mapping is not None:
            gtData['labels'] = gtLabel_mapping(gtData['labels'])
        # print(f"gtData: {gtData}")
        # print(f"{imName} gt['labels']: {gtData['labels']}")
        # print(f"gt['labels']>{num_classes}: {np.sum(np.array(gtData['labels'])>num_classes)}")
        # print(f"gt['labels']<{num_classes}: {np.sum(np.array(gtData['labels'])<num_classes)}")

        if len(detData) == 0: #no detections for this image
            continue

        # det
        # detBoxes, detScores, detLogits, detFeats = decompose_detData(detData, num_logits)
        detBoxes = detData[:, 0:4]
        detScores = detData[:, 4]
        detLogits = detData[:, 5:5+num_logits]
        if detData.shape[1] > 5+num_logits:
            detFeats = detData[:, 5+num_logits:]
        else:
            detFeats = None
        detPredict = np.argmax(detLogits, axis = 1)
        # if gtLabel_mapping is not None:
        #     detPredict = gtLabel_mapping(detPredict)

        #only consider detections that meet the score threshold
        mask = detScores >= scoreThresh
        detScores = detScores[mask]
        detBoxes = detBoxes[mask]
        detLogits = detLogits[mask]
        detPredict = detPredict[mask]
        if detFeats is not None:
            detFeats = detFeats[mask]
        all_filenames = [imName] * len(detScores)
        # print(detPredict)

        allDets = {'filenames':all_filenames, 'predictions': detPredict, 'scores': detScores, 'boxes': detBoxes, 'logits': detLogits}
        if detFeats is not None:
            allDets.update({'feats': detFeats })
        #assign prediction types (known or unknown) to detections
        allData = assign_types(allData, allDets, gtData, clsCutoff=num_classes, iouThresh=iouThresh)
        # print(f"#type: {len(allData['type'])}")
        # # print(f"allData['type']: {allData['type']}")
        # print(f"#type 0: {np.sum(np.array(allData['type'])==0)}")
        # print(f"#type 1: {np.sum(np.array(allData['type'])==1)}")
        # print(f"#type 2: {np.sum(np.array(allData['type'])==2)}")
        # print(f"#type 3: {np.sum(np.array(allData['type'])==3)}")

    return allData

def assign_train(ds_dict_list, raw_res_pth_list, num_logits, iouThresh=0.5):
    allLogits = []
    allLabels = []
    allScores = []
    allFeats = []
    allIoUs = []
    all_filenames = []
    
    trainDatasets = []
    for ds_dict in ds_dict_list:
        trainDatasets.append(build_dataset(ds_dict))
    
    allTrainDets = []
    for pth in raw_res_pth_list:
        with open(pth, 'r') as f:
            allTrainDets.append(json.load(f))

    for tIdx, trainDataset in enumerate(trainDatasets):
        data_set = ds_dict_list[tIdx]
        trainDets = allTrainDets[tIdx]
        lenDataset = len(trainDataset)
        for imIdx, _ in enumerate(tqdm.tqdm(trainDataset, total = lenDataset)):
            imName = trainDataset.data_infos[imIdx]['filename']	
            detData = np.asarray(trainDets[imName])
            gtData = trainDataset.get_ann_info(imIdx)

            #continue if no detections made
            if len(detData) == 0:
                continue
            
            # det
            detBoxes = detData[:, 0:4]
            detScores = detData[:, 4]
            detLogits = detData[:, 5:5+num_logits]
            if detData.shape[1] > 5+num_logits:
                detFeats = detData[:, 5+num_logits:]
            else:
                detFeats = None
            # detBoxes, detScores, detLogits, detFeats = decompose_detData(detData, num_logits)
            detPredict = np.argmax(detLogits, axis = 1)

            # gt
            gtBoxes = gtData['bboxes']
            gtLabels = gtData['labels']
            
            ious = iouCalc(detBoxes, gtBoxes)
            for detIdx, predcls in enumerate(detPredict):
                iou = ious[detIdx]
                mask = iou > iouThresh

                trueClasses = gtLabels[mask]
                gtMatches = np.where(predcls == trueClasses)[0]

                if len(gtMatches) > 0:
                    allLogits += [detLogits[detIdx].tolist()]
                    allLabels += [int(predcls)]
                    allScores += [detScores[detIdx]]
                    all_filenames += [data_set['img_prefix']+imName]
                    if detFeats is not None:
                        allFeats += [detFeats[detIdx].tolist()]

                    maxIoU = np.max(iou[mask][gtMatches])	
                    allIoUs += [maxIoU]

    trainDict = {'filenames':all_filenames, 'dataset':data_set, 'logits': list(allLogits), 'labels': list(allLabels), 'scores': list(allScores), 'ious': list(allIoUs)}
    if detFeats is not None:
        trainDict.update({'feats': list(allFeats) })
    return trainDict

def fit_gmms(logits, labels, ious, confs, scoreThresh, iouThresh, num_classes, components = 1, covariance_type = 'full'):
    gmms = [None for i in range(num_classes)]
    for i in range(num_classes):
        ls = logits[labels == i]
        iou = ious[labels == i]
        conf = confs[labels == i]
        
        if len(ls) < components+2: #no objects from this class were detected, or not enough given the components number
            continue	

        #mask for high iou and high conf
        mask = (iou >= iouThresh)*(conf >= scoreThresh)
        lsThresh = ls[mask]

        #only threshold if there is enough logits given the amount of components
        if len(lsThresh) < components+2: 
            lsThresh = ls
        
        gmms[i] = sm.GaussianMixture(n_components = components, random_state = 0, max_iter = 200, n_init = 2, covariance_type = covariance_type).fit(lsThresh)

    return gmms

def gmm_uncertainty(allLogits, gmms):
    gmmScores = []
    #test all data in 10 batches - not too slow, doesn't overload cpu
    intervals = np.ceil(len(allLogits)/10)
    sI = [int(i*intervals) for i in range(10)]
    eI = [int(s+intervals) for s in sI]
    for jj, inty in enumerate(sI):
        clsScores = []
        ls = allLogits[inty:eI[jj]]

        #find logit log likelihood for every class GMM
        for clsIdx, gmm in enumerate(gmms):
            if gmm == None:
                continue

            gmmLL = gmm.score_samples(ls)
            clsScores += [gmmLL]

        clsScores = np.array(clsScores)

        #we use the maximum likelihood to reperesent uncertainty
        maxScore =  np.max(clsScores, axis = 0)
        gmmScores += list(maxScore)
        
    gmmScores = np.array(gmmScores)
    return gmmScores

def aurocScore(inData, outData):
	allData = np.concatenate((inData, outData))
	labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
	fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, allData, pos_label = 0)  
	return fpr, tpr, sklearn.metrics.auc(fpr, tpr)

def auprScore(inData, outData):
	allData = np.concatenate((inData, outData))
	labels = np.concatenate((np.zeros(len(inData)), np.ones(len(outData))))
	precision, recall, thresholds = sklearn.metrics.precision_recall_curve(labels, allData, pos_label = 0)  
	return precision, recall, sklearn.metrics.auc(recall, precision)

def tprAtFpr(tpr, fpr, fprRate = 0.05):
	fprAdjust = np.abs(np.array(fpr)-fprRate)
	fprIdx = np.argmin(fprAdjust)
	tpratfpr = tpr[fprIdx]

	return tpratfpr, fpr[fprIdx]

def summarise_performance(inData, outData, fprRates = [], printRes = True, methodName = '', log_file=None):
    results = {}
    logger = get_logger("flowDet_mmdet.utils.gmmDet_utils", log_file)

    fpr, tpr, auroc = aurocScore(inData, outData)
    pc, rec, aupr = auprScore(inData, outData)
    results['auroc'] = auroc
    results['aupr'] = aupr
    # results['fpr'] = list(fpr)
    # results['tpr'] = list(tpr)

    specPoints = []
    for fprRate in fprRates:
        tprRate = tprAtFpr(tpr, fpr, fprRate)
        specPoints += [tprRate]

        results[f'tpr at fprRate {fprRate}'] = tprRate
    
    if printRes:
        logger.info(f'Results for Method: {methodName}')
        logger.info(f'------ AUPR: {round(aupr, 3)}')
        logger.info(f'------ AUROC: {round(auroc, 3)}')
        for point in specPoints:
            fp = point[1]
            tp = point[0]
            logger.info(f'------ TPR at {round((100.*fp), 1)}FPR: {round((100.*tp), 1)}')

    return results


class DisjointSet:
    def __init__(self):
        self.sets = []

    # ==================================
    def create_set(self, elem):
        if type(elem) == list:
            self.sets.append(elem)
        else:
            self.sets.append([elem])

    # ==================================
    def find(self,elem):
        for s in self.sets:
            if elem in s:
                return s
        # print('Could not find ', elem)
        return None


    # ==================================
    def merge(self, a, b):
        setA = self.find(a)
        setB = self.find(b)

        if setA is None or setB is None:
            return

        if setA != setB:
            setA.extend(setB)
            self.sets.remove(setB)
    # ==================================
    def __str__(self):
        string = ''

        for s in self.sets:
            string += str(s) + '\n'

        return string

class SamplingDetector():
    """Base for any sampling-based detector

    Args:
        iou: spatial affinity IoU cutoff during merging
        min_dets: minimum number of detections in each cluster
        label: label affinity measure to use, if any
        min_score: minimum known softmax score a detection must have to be considered

    """

    def __init__(self, min_dets = 2, iou = 0.8, label = None, vis = False):
        self.iou = iou
        self.label = label
        self.min_dets = min_dets
        self.visualise = vis

    def form_final(self, detections):
        #detections are in format of [logits, bbox]

        #cluster detections from each forward pass 
        det_set = DisjointSet()
        det_set = self.cluster(detections, det_set)

        #remove clusters not meeting the minimum number of detections
        det_set = self.remove_for_min(det_set)
        if (len(det_set.sets) == 0): 
            return []

        #form observations from cluster
        observations = self.form_observations(detections, det_set)

        final_detections = self.form_final_detections(observations)

        return final_detections


    def remove_for_min(self, det_set):
        #no minimum detections, just return set
        if self.min_dets == 0:
            return det_set

        remove = []
        #only keep observations that were detected to have the minimum number of detections
        for i in range(len(det_set.sets)):
            s = det_set.sets[i]
            if (len(s)) < self.min_dets:
                remove.append(s)
        for r in remove:
            det_set.sets.remove(r)

        return det_set

    
    def cluster(self, detections, det_set):
        bboxes = detections[:, -4:]

        distributions = detections[:, :-4]
        
        #create a set for every detection
        for idx in range(len(detections)):
            det_set.create_set([idx])

        #find affinity matrix
        spatial_matrix = np.asarray(self.spatial_association(bboxes, bboxes))

        if self.label == None:
           #no label association, only spatial 
            matrix = spatial_matrix
        else:
            label_matrix = np.asarray(self.label_association(distributions, distributions, self.label))
           
            #detections have same winning label
            matrix = (label_matrix) * spatial_matrix

        for i in range(len(detections)):
            #which other sets meet the threshold minimum?
            candidates = np.nonzero(matrix[i,] >= self.iou)[0]
            for c in candidates:
                if c != i:
                    det_set.merge(i, c)

        return det_set

    def spatial_association(self, old_bboxes, new_bboxes):
        #finds the IoU between bboxes
        assoc_matrix = []

        old_bboxes = np.asarray(old_bboxes)
        new_bboxes = np.asarray(new_bboxes)
       
        for idx in range(len(new_bboxes)):
            nx1 = new_bboxes[idx, -4] * np.ones(len(old_bboxes))
            ny1 = new_bboxes[idx, -3] * np.ones(len(old_bboxes))
            nx2 = new_bboxes[idx, -2] * np.ones(len(old_bboxes))
            ny2 = new_bboxes[idx, -1] * np.ones(len(old_bboxes))
           
            #find the iou with the detection bbox
            ox1 = old_bboxes[:, 0] 
            oy1 = old_bboxes[:, 1]
            ox2 = old_bboxes[:, 2]
            oy2 = old_bboxes[:, 3]

            #centroids, width and heights
            ncx = (nx1+nx2)/2
            ncy = (ny1+ny2)/2
            nw = nx2-nx1
            nh = ny2-ny1

            ocx = (ox1+ox2)/2
            ocy = (oy1+oy2)/2
            ow = ox2-ox1
            oh = oy2-oy1
            
            ### 1 is good, 0 is bad
            xx1 = np.max([nx1, ox1], axis = 0)
            yy1 = np.max([ny1, oy1], axis = 0)
            xx2 = np.min([nx2, ox2], axis = 0)
            yy2 = np.min([ny2, oy2], axis = 0)       

            w = xx2 - xx1
            h = yy2 - yy1
            w = w * (w > 0)
            h = h * (h > 0)  

            inter = w*h
            Narea = (nx2-nx1)*(ny2-ny1)
            Oarea = (ox2-ox1)*(oy2-oy1)
            union = Narea + Oarea - inter
            IoUs = inter/union
            assoc_matrix.append(IoUs)

        return assoc_matrix

    def label_association(self, old_dists, new_dists, method):
        assoc_matrix = []
        if method == None:
            return assoc_matrix

        if 'kl' in method:
            assoc_matrix = scipy.stats.entropy(new_dists.T[:, :, None], old_dists.T[:, None, :])
            return assoc_matrix

        for idx in range(len(new_dists)):
            new_dist = new_dists[idx]

            if 'label' in method:
                assoc_matrix.append(np.argmax(new_dist) == np.argmax(old_dists, axis = 1))           ### 1 is good, 0 is bad

        return assoc_matrix

    def form_observations(self, total_detections, det_set, img = None):
        #merge observations into final detections
        observations=[]

        for S in det_set.sets:
            D = []
            for detection_id in S:
                D.append(total_detections[detection_id])
            observations += [np.asarray(D)]

        return observations

    def form_final_detections(self, observations):
        detections = []
        for ob_individ in observations:
            distribution = np.mean(ob_individ[:, :-4], axis = 0)
            bbox = np.mean(ob_individ[:, -4:], axis = 0) 

            detections += [distribution.tolist() + bbox.tolist()]

        return np.asarray(detections)
