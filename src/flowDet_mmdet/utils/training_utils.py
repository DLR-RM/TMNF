import os, logging
from typing import Optional
import numpy as np
import torch
import torch.optim as optim
import torch.nn.utils as utils
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_runner,
                         get_dist_info)
from mmdet.core import DistEvalHook, EvalHook, build_optimizer
from mmdet.utils import (build_ddp, build_dp, compat_cfg,
                         find_latest_checkpoint, get_root_logger)
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
import normflows as nf
from flowDet_mmdet.utils.helper_utils import get_logger

class flowDetOptimizerHook(OptimizerHook):
    def __init__(self, flow_params, detect_anomalous_params: bool = False):
        super().__init__(grad_clip=None, detect_anomalous_params=detect_anomalous_params)
        """
        hook for the following functions:
        1. implement grad norm for normalizing flow parameters for stable training;
        2. updata nfr_weight after each epoch;
        """
        self.flow_params = flow_params

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        if self.detect_anomalous_params:
            self.detect_anomalous_parameters(runner.outputs['loss'], runner)
        runner.outputs['loss'].backward()

        # cur_nfr_w = runner.model.module.roi_head.bbox_head.loss_cls
        # cur_ep = runner.epoch
        # max_ep = runner._max_epochs
        # cur_itr = runner.iter
        # max_itr = runner._max_iters
        # w_ep_start = self.flow_params['nfr_weight_start']
        # w_ep_end = self.flow_params['nfr_weight_end']
        # w_start = w_ep_start  + cur_ep * (w_ep_start - w_ep_end)/max_ep 
        # w_end = w_ep_start  + (cur_ep+1) * (w_ep_start - w_ep_end)/max_ep 
        # cur_w = w_start + cur_itr * (w_end - w_start)/max_itr
        # runner.model.module.roi_head.nfr_weight = torch.tensor(w_start).to('cuda')

        if self.flow_params['max_norm'] > 0:
            # flow_params = runner.model.module.roi_head.flow_module.parameters()
            flow_params = runner.model.module.roi_head.bbox_head.loss_cls.flow_module.parameters()
            utils.clip_grad.clip_grad_norm_(flow_params, self.flow_params['max_norm'])
        runner.optimizer.step()
    
    def after_train_epoch(self, runner):
        # print("after_train_epoch in flowDetOptimizerHook")
        cur_ep = runner.epoch
        w_start = self.flow_params['nfr_weight_start_ep']
        if cur_ep > w_start or cur_ep == w_start:
            runner.model.module.roi_head.nfr_weight = torch.tensor(self.flow_params['nfr_weight']).to('cuda')
        # print("exit after_train_epoch")

def setup_flow_opt(params_dict, flow_module):
    params = []
    for module in flow_module.modules():
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            params.append(p)

    param_group = [{"params": params, 'weight_decay': params_dict['weight_decay']}]
    if params_dict['optm'] == "SGD":
        optimizer = optim.SGD(param_group, lr=params_dict['lr']) # SGD, Adam
    elif params_dict['optm'] == "Adam":
        optimizer = optim.Adam(param_group, lr=params_dict['lr']) # SGD, Adam
    elif params_dict['optm'] == "Adamax":
        optimizer = optim.Adamax (param_group, lr=params_dict['lr']) # SGD, Adam
    else:
         raise NotImplementedError(f"this optimizer({params_dict['optm']}) has not been implemented! ")

    return optimizer

def train_flow_epoch_v2(model, 
                        params_dict,
                        train_dataloader, 
                        optimizer, 
                        max_norm, 
                        log_file=None,
                        device="cuda"):
    model.train()
    mean_nll_epoch = []
    # prior_epoch = []
    # logdet_epoch = []
    logger = get_logger("flowDet_mmdet.utils.training_utils", log_file)
    for x, y in train_dataloader:
        x = x.to(device)

        if "cls" in params_dict['prior_type'] and "cls_ib" not in params_dict['prior_type']:
            y = y.to(device)
            y = torch.nn.functional.one_hot(y.to(torch.int64), model.q0.n_cls)
            loss = model.forward_kld(x, y)
            loss /= model.q0.dim
        elif "cls_ib" in params_dict['prior_type']:
            beta = params_dict['ib_beta']
            sigma = params_dict['ib_sigma']
            y = y.to(device)
            beta = torch.tensor(beta).to(device)
            sigma = torch.tensor(sigma).to(device)
            y = torch.nn.functional.one_hot(y.to(torch.int64), model.q0.n_cls)
            log_q, log_pxy = model.IB_loss(x, y, sigma)
            # log_q /= self.q0.dim
            beta_nll = 1. / (1 + beta)
            beta_cls = 1. * beta / (1 + beta)
            loss = -beta_nll*torch.mean(log_q) - beta_cls*torch.mean(log_pxy)
        else:
            loss = model.forward_kld(x)
            loss /= model.q0.dim

        if not torch.isnan(loss) and not torch.isinf(loss):
            optimizer.zero_grad()
            loss.backward() 
            if max_norm is not None:
                for group in optimizer.param_groups:
                    utils.clip_grad_norm_(group['params'], max_norm, norm_type=2)
            optimizer.step()
        
            mean_nll_epoch.append(loss.data.cpu().numpy())
            # logdet_epoch.append(logdet.data.cpu().numpy())
            # prior_epoch.append(prior.data.cpu().numpy())

        # Update Lipschitz constant if flows are residual
        if params_dict['flow_type'] == 'residual':
            nf.utils.update_lipschitz(model, 5)

    logger.info(f'mean nll is {np.mean(mean_nll_epoch):.3f}.')
    # print(f'mean negative prior is {np.mean(prior_epoch):.3f}.')
    # print(f'mean negative logdet is {np.mean(logdet_epoch):.3f}.')

def train_flowDet(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   log_file=None,
                   meta=None):

    cfg = compat_cfg(cfg)
    logger = get_root_logger(log_file=log_file, log_level=logging.DEBUG)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner[
        'type']

    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # build optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # init customized optim hook
    # optimizer_config = cfg.optimizer_config 
    if "flow_params" not in cfg:
        optimizer_config = cfg.optimizer_config 
    else:
        optimizer_config = flowDetOptimizerHook(cfg.flow_params, detect_anomalous_params=True)

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataloader_default_args = dict(
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=distributed,
            shuffle=False,
            persistent_workers=False)

        val_dataloader_args = {
            **val_dataloader_default_args,
            **cfg.data.get('val_dataloader', {})
        }
        # Support batch_size > 1 in validation

        if val_dataloader_args['samples_per_gpu'] > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
        # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
        runner.register_hook(
            eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)
    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)

# def train_flow_epoch(model, 
# 					train_dataloader, 
# 					optimizer, 
# 					prior_type, 
# 					max_norm, 
# 					device="cuda"):
# 	model.train()
# 	mean_nll_epoch = []
# 	prior_epoch = []
# 	logdet_epoch = []
# 	for x, y in train_dataloader:
# 		x = x.to(device)
# 		y = y.to(device)

# 		if prior_type == "gmm":
# 			# nll, prior, logdet = model.nll(x, y=y, return_more=True)
# 			nll, prior, logdet = model.nll(x, return_more=True, sum_class_dim=True)
# 		else:
# 			nll, prior, logdet = model.nll(x, return_more=True)
# 		# nll_bg = nll[y==num_classes]
# 		# nll_fg = nll[y!=num_classes]
# 		loss = nll.mean() # - nll_bg.mean() + nll_fg.mean() # nll.mean()
# 		prior = -prior.mean()
# 		logdet = -logdet.mean()

# 		mean_nll_epoch.append(loss.data.cpu().numpy())
# 		logdet_epoch.append(logdet.data.cpu().numpy())
# 		prior_epoch.append(prior.data.cpu().numpy())

# 		optimizer.zero_grad()
# 		loss.backward() 
# 		for group in optimizer.param_groups:
# 			utils.clip_grad_norm_(group['params'], max_norm, norm_type=2)
# 		optimizer.step()

# 	print(f'mean nll is {np.mean(mean_nll_epoch):.3f}.')
# 	print(f'mean negative prior is {np.mean(prior_epoch):.3f}.')
# 	print(f'mean negative logdet is {np.mean(logdet_epoch):.3f}.')
