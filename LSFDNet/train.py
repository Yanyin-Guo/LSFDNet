import os
import datetime
import logging
import math
import time
import torch
import numpy as np
import random
import archs,data,models,losses
from copy import deepcopy
from os import path as osp
from torch.utils.data import SequentialSampler 
from functools import partial

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher, InfiniteDataLoader
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

def init_tb_loggers(opt):
    # initialize wandb logger before tensorboard logger to allow proper sync
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
        tb_logger = init_tb_logger(log_dir=osp.join(*osp.split(root_path)[:-1], 'tb_logger', opt['name']))
    return tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loaders = None, []
    for phase, dataset_opt in opt['datasets'].items():
        if opt.get('hyp') is not None:
            new_opt = deepcopy(opt)
            new_opt.update(dataset_opt)
            dataset_opt = new_opt
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = build_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio)
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio / (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
            total_iters = int(opt['train']['total_iter'])
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        elif phase.split('_')[0] == 'val':
            val_set = build_dataset(dataset_opt)
            val_sampler = SequentialSampler(val_set)
            val_loader = build_dataloader(
                val_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=val_sampler, seed=opt['manual_seed'])
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader)
        else:
            raise ValueError(f'Dataset phase {phase} is not recognized.')
        
        opt['total_epochs'] = total_epochs

    return train_loader, train_sampler, val_loaders, val_sampler, total_epochs, total_iters

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """
    Override the build_dataloader of the base_model.
    """
    phase = dataset_opt['phase']
    rank, _ = get_dist_info()
    if phase == 'train':
        if dist:  # distributed training
            batch_size = dataset_opt['batch_size_per_gpu']
            num_workers = dataset_opt['num_worker_per_gpu']
        else:  # non-distributed training
            multiplier = 1 if num_gpu == 0 else num_gpu
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier
        dataloader_args = dict(
            dataset=dataset,
            collate_fn=getattr(dataset, "collate_fn", None),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sampler=sampler,
            drop_last=True)
        if sampler is None:
            dataloader_args['shuffle'] = True
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed) if seed is not None else None
    elif phase in ['val', 'test']:  # validation
        multiplier = 1 if num_gpu == 0 else num_gpu
        dataloader_args = dict(dataset=dataset, collate_fn=getattr(dataset, "collate_fn", None), batch_size=dataset_opt['batch_size_per_gpu'] * multiplier, shuffle=False, num_workers=dataset_opt['num_worker_per_gpu'] * multiplier)
    else:
        raise ValueError(f'Wrong dataset phase: {phase}. ' "Supported ones are 'train', 'val' and 'test'.")

    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False)
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False)

    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPUPrefetcher
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1)
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args)
    elif prefetch_mode == 'Infinite': 
        logger = get_root_logger()
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        return InfiniteDataLoader(**dataloader_args)
    else:
        # prefetch_mode=None: Normal dataloader
        # prefetch_mode='cuda': dataloader for CUDAPrefetcher
        return torch.utils.data.DataLoader(**dataloader_args)

def check_resume(opt, resume_iter):
    """
    Override the check_resume of the base_model.
    """
    if opt['path']['resume_state']:
        # get all the networks
        networks = [key for key in opt.keys() if key.startswith('network_')]
        pre_networks = [key for key in opt['path'].keys() if key.startswith('pretrain_network_')]
        flag_pretrain = False
        for network in networks:
            if opt['path'].get(f'pretrain_{network}') is not None:
                flag_pretrain = True
        if flag_pretrain:
            print('pretrain_network path will be ignored during resuming.')
            # set pretrained model paths
            for network in networks:
                print('pretrain_network path will be divided by network.')
                name = f'pretrain_{network}'
                basename = network.replace('network_', '')
                if opt['path'].get('ignore_resume_networks') is None or (network
                                                                        not in opt['path']['ignore_resume_networks']):
                    opt['path'][name] = osp.join(opt['path']['models'], f'net_{basename}_{resume_iter}.pth')
                    print(f"Set {name} to {opt['path'][name]}")
        else:
            print('Tyring to scaning the whole networks.')
            name = pre_networks[0]
            for filename in os.listdir(opt['path']['models']):
                if filename.endswith('_'+str(resume_iter)+'.pth'):
                    opt['path'][name] = os.path.join(opt['path']['models'], filename)
                    print(f"Set {name} to {opt['path'][name]}")
        # change param_key to params in resume
        param_keys = [key for key in opt['path'].keys() if key.startswith('param_key')]
        for param_key in param_keys:
            if opt['path'][param_key] == 'params_ema':
                opt['path'][param_key] = 'params'
                print(f'Set {param_key} to params')

def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    else:
        if opt['path'].get('resume_state'):
            resume_state_path = opt['path']['resume_state']

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
        check_resume(opt, resume_state['iter'])
    return resume_state


def train_pipeline(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, args = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # load resume states if necessary
    resume_state = load_resume_state(opt)
    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            # mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
            mkdir_and_rename(osp.join(*osp.split(root_path)[:-1], 'tb_logger', opt['name'])) #modify

    # copy the yml file to the experiment root
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    # WARNING: should not use get_root_logger in the above codes, including the called functions
    # Otherwise the logger will not be properly initialized
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    # initialize wandb and tb loggers
    tb_logger = init_tb_loggers(opt)

    # create train and validation dataloaders
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, val_sampler, total_epochs, total_iters = result

    # create model
    model = build_model(opt)
    if resume_state:  # resume training
        model.resume_training(resume_state)  # handle optimizers and schedulers
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else:
        start_epoch = 0
        current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    best_score = 0
    best_iter = 0
    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(opt['logger']['print_freq']), AvgTimer(opt['logger']['print_freq'])
    start_time = time.time()

    for epoch in range(start_epoch, total_epochs + 1):
        if opt["hyp"].get("close_mosaic", False) and epoch == opt["hyp"]["close_mosaic"]:
            train_loader.dataset.hyp.mosaic = False
            logger.info("Closing dataloader mosaic")
            train_loader.dataset.close_mosaic(hyp=train_loader.dataset.hyp)
            train_loader.reset()
        train_sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        for i, train_data in pbar:
        # while train_data is not None:
            data_timer.record()
            current_iter += 1
            if current_iter > total_iters:
                break

            # for val_loader in val_loaders:
            #     model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

            # update learning rate
            model.update_learning_rate(current_iter, epoch, warmup_iter=opt['train'].get('warmup_iter', -1))
            # training
            if not opt['val'].get('cal_score', False):
                model.feed_data(train_data)
                model.optimize_parameters(current_iter)
                iter_timer.record()
                if current_iter == 1:
                    # reset start time in msg_logger for more accurate eta_time
                    # not work in resume mode
                    msg_logger.reset_start_time()
                # log
                if current_iter % opt['logger']['print_freq'] == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                    log_vars.update(model.get_current_log())
                    if not opt['logger'].get('only_save_log', False):
                        model.save_current_log_img()
                    msg_logger(log_vars)
            # validation
            if current_iter >= opt['val']['start_val'] and (opt['val'].get('cal_score', False) or (opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0))):
                if len(val_loaders) > 1:
                    logger.warning('Val wrong. Multiple validation datasets are *only* supported by SRModel.')
                for val_loader in val_loaders:
                    model.is_ema_val = False  # 标记当前为常规验证
                    model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                    if hasattr(model, 'ema') and model.ema:
                        logger.info('--- Starting EMA Validation ---')
                        model.is_ema_val = True  # 标记当前为 EMA 验证
                        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
                        model.is_ema_val = False

            # save models and training states
            if current_iter >= opt['logger']['start_save'] and current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter) 
                if opt['logger'].get('only_save_last_best', False):
                    model.remove(int(current_iter - opt['logger']['save_checkpoint_freq']))
                    current_score = model.get_current_model_score()
                    # logger.info(current_score)
                    if current_score >= best_score:
                        model.save_best(current_iter)
                        model.remove_best(best_iter)
                        best_score = current_score
                        best_iter = current_iter
            data_timer.start()
            iter_timer.start()
        model.ema_update()
        # end of iter

    # end of epoch

    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # -1 stands for the latest
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])
    if tb_logger:
        tb_logger.close()

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir))
    train_pipeline(root_path)