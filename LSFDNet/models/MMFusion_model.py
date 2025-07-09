import logging
import functools
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
from os import path as osp
import numpy as np
from basicsr.utils import get_root_logger,tensor2img,imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
from core.Metric_fusion.eval_one_method import evaluation_one_method_fast, evaluation_one_method
import core.weights_init
from tqdm import tqdm
#from ultralytics import YOLO

@MODEL_REGISTRY.register()
class MMFusion(BaseModel):
    def __init__(self, opt):
        super(MMFusion, self).__init__(opt)
        logger = get_root_logger()
        # define network and load pretrained models
        self.netfusion =  build_network(opt['network_fusion'])
        self.netfusion = self.model_to_device(self.netfusion)
        self.print_network(self.netfusion)

        if isinstance(self.netfusion, (DataParallel, DistributedDataParallel)):
            self.netfusion = self.netfusion.module
        else:
            self.netfusion = self.netfusion

        if self.is_train:
            self.init_training_settings()
        else:
            self.netfusion.eval()

        load_path = self.opt['path'].get('pretrain_network_MMFusion', None)
        if load_path is not None:
            self.load_network(self.netfusion, load_path, self.opt['path'].get('strict_load_g', True), 'params_MMFusion')
            logger.info(f"Pretrained model is successfully loaded from {opt['path']['pretrain_network_MMFusion']}")

        self.current_iter = 0
        self.gama = opt['train']["a"]
        self.b=opt['train']["b"]
        self.c=opt['train']["c"]

    def _initialize_weights(self):  
        logger = get_root_logger()
        weights_init = functools.partial(core.weights_init.weights_init_normal)
        self.netfusion.apply(weights_init)
        logger.info(f"Initialize weights of model")
    
    def init_training_settings(self):
        self.netfusion.train()
        self.loss_dict_all = OrderedDict() 
        self.loss_dict_all['loss_all'] = []
        self.loss_dict_all['loss_back'] = []
        self.loss_dict_all['loss_label'] = []
        self.loss_dict_all['loss_in'] = []
        self.loss_dict_all['loss_grad'] = []
        self.loss_dict_all['ls'] = []
        self.loss_dict_all['lin'] = []
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.loss_LS = build_loss(self.opt['train']['Loss_OE']).to(self.device)

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_netfusion_params = list(self.netfusion.parameters())
        optim_params_g_netfusion = [{  # add normal params first
            'params': optim_netfusion_params,
            'lr': train_opt['optimizer']['lr']
        }]
        optim_type = train_opt['optimizer'].pop('type')
        lr = train_opt['optimizer']['lr']
        self.optimizer_g_netfusion = self.get_optimizer(optim_type, optim_params_g_netfusion, lr)
        self.optimizers.append(self.optimizer_g_netfusion)


    # Feeding all data to the DF model
    def feed_data(self, train_data):
        self.data = self.set_device(train_data[0])

    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        self.current_iter = current_iter
        self.optimizer_g_netfusion.zero_grad()
        loss_dict = OrderedDict() 
        self.pred_img = self.netfusion(self.data)
        loss_ss, loss_back, loss_label, loss_in,loss_grad, ls,lin = self.loss_LS(self.b,self.c,image_vis=self.data["SW"], image_ir=self.data["LW"],  generate_img=self.pred_img, label=self.data["label"], LW_th=self.data["LW_th"])
        for i in range(len(loss_label)):
            if loss_label[i] !=0 :
                loss_ss[i] = self.gama * loss_back[i] + (1-self.gama) * loss_label[i]
            else:
                loss_ss[i] = loss_back[i]
        loss_fs=loss_ss.mean()
        loss_fs.backward()
        self.optimizer_g_netfusion.step()

        loss_dict['loss_all'] = loss_ss
        loss_dict['loss_back'] = loss_back
        loss_dict['loss_label'] = loss_label
        loss_dict['loss_in'] = loss_in
        loss_dict['loss_grad'] = loss_grad
        loss_dict['ls'] = ls
        loss_dict['lin'] = lin
        loss_dict = self.set_device(loss_dict)
        self.log_dict = self.reduce_loss_dict(loss_dict)
        for name, value in self.log_dict.items():
            self.loss_dict_all[name].append(value)


    # Testing on given data
    def test(self):
        self.netfusion.eval()
        with torch.no_grad():
            self.pred_img = self.netfusion(self.data)
        self.netfusion.train()

        return self.pred_img
    
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    if key == 'label':
                        x[key] = item.to(self.device, dtype=torch.int)
                    else:
                        x[key] = item.to(self.device, dtype=torch.float)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device, dtype=torch.float)
        else:
            x = x.to(self.device, dtype=torch.float)
        return x
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = self.opt['datasets']['val'].get('type')
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # for idx, val_data in enumerate(dataloader):
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(val_data[1][0])[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()

            sr_img = tensor2img(visuals['pred_img'].detach(), min_max=(-1, 1))
            metric_data['img'] = sr_img

            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                             f'{img_name}.jpg')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, current_iter,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, current_iter,
                                                 f'{img_name}.jpg')
                imwrite(sr_img, save_img_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            r_dir, f_name = os.path.split(save_img_path) 
            save_dir =  osp.join(self.opt['path']['visualization'], 'metric')
            os.makedirs(save_dir, exist_ok=True)
            metric_r = evaluation_one_method(dataset_name='OSLSP_320', data_dir='/data', result_dir=r_dir, save_dir= save_dir+ f'/{current_iter}' + '_metric_C2FM.xlsx', Method='C2FM' , with_mean=True)
            metric_f=['EN', 'SF', 'AG', 'SD', 'CC', 'SCD', 'MSE', 'PSNR', 'Qabf', 'Nabf']
            for index, metric in enumerate(metric_f):
                self.metric_results[metric] = metric_r[index][0]
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
            log_str = f'Validation {dataset_name}\n'
            for metric, value in self.metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}'
                if hasattr(self, 'best_metric_results'):
                    log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                                f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
                log_str += '\n'

            logger = get_root_logger()
            logger.info(log_str)
            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    # Get current log
    def get_current_iter_log(self):
        #self.update_loss()
        return self.log_dict
    
    def get_current_log(self):
        for name, value in self.log_dict.items():
            self.log_dict[name] = np.average(self.loss_dict_all[name])
        visuals = self.get_current_visuals()
        grid_img = torch.cat((visuals['pred_img'].detach(),
                                    visuals['gt_SW'],
                                    visuals['gt_LW']), dim=0)
        grid_img = tensor2img(grid_img, min_max=(-1, 1))
        save_img_path = os.path.join(self.opt['path']['visualization'],'img_fused_iter_{}.jpg'.format(self.current_iter))
        imwrite(grid_img, save_img_path)
        return self.log_dict


    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_img'] = self.pred_img
        out_dict['gt_SW'] = self.data["SW"]
        out_dict['gt_LW'] = self.data["LW"]
        return out_dict
    
    def save(self, epoch, current_iter):
        self.save_network([self.netfusion], 'net_fe_g', current_iter, param_key=['params_MMFusion'])
        self.save_training_state(epoch, current_iter)