import functools
from collections import OrderedDict
import torch
from torch import nn, optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
from os import path as osp
import numpy as np
from basicsr.utils import get_root_logger,tensor2img,imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
from core.Metric_fusion.eval_one_method import evaluation_one_method_fast, evaluation_one_method_test, evaluation_pic_fast
import core.weights_init
from tqdm import tqdm
from scripts.util import RGB2YCrCb, YCrCb2RGB
from ultralytics.utils.torch_utils import ModelEMA, autocast

@MODEL_REGISTRY.register()
class MMFusion(BaseModel):
    def __init__(self, opt):
        super(MMFusion, self).__init__(opt)
        logger = get_root_logger()
        self.batch_size = opt['datasets']['train']['batch_size_per_gpu']
        self.target_size = self.opt['datasets']['train']['target_size']
        self.is_compile = self.opt['train'].get('compile', False)
        self.last_iter = 0
        self.accumulate = 1
        self.nw = max(round(opt['train']['warmup_iter']), 100) if self.opt['train'].get('warmup_iter', -1) > 0 else -1 
        self.expect_batch_size = 64
        # define network and load pretrained models
        self.netfusion =  build_network(opt['network_fusion'])
        self.netfusion = torch.compile(self.netfusion) if self.is_compile else self.netfusion
        self.netfusion = self.model_to_device(self.netfusion)
        self._initialize_weights() 
        if opt['logger'].get('print_net', False):
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
        self.alpha=opt['train']["alpha"]
        self.beta=opt['train']["beta"]
        self.sigma=opt['train']["sigma"]
        self.gamma=opt['train']["gamma"]

    def transfer_weights(self, small_model, large_model):  
        small_model_state_dict = small_model.state_dict()  
        large_model_state_dict = large_model.state_dict()   
         
        for name, param in small_model_state_dict.items():  
            if name in large_model_state_dict:  
                large_model_state_dict[name].data.copy_(param.data)  
            else:
                print(name)

    def _initialize_weights(self):  
        logger = get_root_logger()
        weights_init = functools.partial(core.weights_init.weights_init_normal)
        self.netfusion.apply(weights_init)
        logger.info(f"Initialize weights of model")
    
    def init_training_settings(self):
        self.netfusion.train()
        self.loss_dict_all = OrderedDict() 
        self.loss_dict_all['loss_all'] = []
        self.loss_dict_all['loss_global'] = []
        self.loss_dict_all['loss_object'] = []
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.loss_LS = build_loss(self.opt['train']['Loss_Fusion']).to(self.device)
        # Check AMP
        self.amp = torch.tensor(self.opt["train"].get('amp', False)).to(self.device)  # True or False
        self.amp = bool(self.amp)  # as boolean
        self.scaler = (torch.amp.GradScaler("cuda", enabled=self.amp))
        self.ema = self.opt['train'].get('ema', False)
        if self.ema:
            self.ema1 = ModelEMA(self.netfusion)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        self.optimizer_g_netfusion = self.build_optimizer(self.netfusion, 
                                                           name=train_opt['optimizer_fusion']['type'],
                                                           lr=train_opt['optimizer_fusion']['lr'],
                                                           momentum=train_opt['optimizer_fusion'].get('momentum', 0.9),
                                                           decay=train_opt['optimizer_fusion'].get('weight_decay', 1e-5),
                                                           iterations=train_opt['total_iter'])
        self.optimizers.append(self.optimizer_g_netfusion)
        self.optimizer_g_netfusion.zero_grad()

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        logger = get_root_logger()
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            logger.info(
                f"{'optimizer:'} 'optimizer=auto' found, "
                f"ignoring 'lr0' and 'momentum' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.nc  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn):  # weight (no decay)
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        logger.info(
            f"{'optimizer:'} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer
     
    # Feeding all data to the DF model
    def feed_data(self, train_data):
        self.batch = train_data
        self.data = {}
        self.data['SW'] = self.set_device(self.batch['img'][:,:1,:,:])
        self.data['LW'] = self.set_device(self.batch['an_img'][:,:1,:,:])
   
    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        self.current_iter = current_iter
        loss_dict = OrderedDict() 
        # with autocast(self.amp):
        with torch.cuda.amp.autocast(enabled = self.amp):
            self.pred_img = self.netfusion(self.data)
            loss_fusion, loss_global, loss_object = self.loss_LS(self.alpha, self.beta, self.sigma, self.gamma, image_SW=self.data["SW"], image_LW=self.data["LW"],  generate_img=self.pred_img, batch =self.batch)
            loss_all = loss_fusion
        self.accumulate = max(1, int(np.interp(self.current_iter, [0, self.nw], [1, self.expect_batch_size / self.batch_size]).round()))
        self.scaler.scale(loss_all/self.accumulate).backward()
        if self.current_iter - self.last_iter >= self.accumulate:
            self.last_iter = current_iter
            self.scaler.unscale_(self.optimizer_g_netfusion)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(self.netfusion.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer_g_netfusion)
            self.scaler.update()
            self.optimizer_g_netfusion.zero_grad()
            if self.ema:
                self.ema1.update(self.netfusion)

        loss_dict['loss_all'] = loss_fusion
        loss_dict['loss_global'] = loss_global
        loss_dict['loss_object'] = loss_object
        loss_dict = self.set_device(loss_dict)
        self.log_dict = self.reduce_loss_dict(loss_dict)
        for name, value in self.log_dict.items():
            self.loss_dict_all[name].append(value)
    
    def ema_update(self):
        if self.ema:
            self.ema1.update_attr(self.netfusion)

    # Testing on given data
    def test(self):
        if getattr(self, 'is_ema_val', False) and hasattr(self, 'ema') and self.ema:
            self.ema1.ema.eval()
            with torch.no_grad():
                self.pred_img = self.ema1.ema(self.data)
        else:
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
                        if isinstance(x[key], list):
                            for item_1 in x[key]:
                                if item_1 is not None and not isinstance(item_1, str) :
                                    item_1 = item_1.to(self.device, dtype=torch.float)
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
        self.dataset_name = self.opt['val'].get('name')
        ema_suffix = '_EMA' if getattr(self, 'is_ema_val', False) else ''
        if ema_suffix:
            dataset_name = f"{dataset_name}{ema_suffix}"
            self.dataset_name = f"{self.dataset_name}{ema_suffix}"

        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        self.data_dir = self.opt['val'].get('data_dir')
        self.save_img_y = self.opt['val'].get('save_img_y', False)
        metric_r = {}
        logger = get_root_logger()

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # for idx, val_data in enumerate(dataloader):
        for idx, val_data in enumerate(dataloader):
            im_name = val_data['im_name']
            self.feed_data(val_data)
            self.pred_img = self.test()
            visuals = self.get_current_visuals()
            torch.cuda.empty_cache()

            n=len(visuals['pred_img'])
            for i in range(n):
                sr_img = tensor2img(visuals['pred_img'][i].detach(), min_max=(0, 1))
                img_name = osp.splitext(im_name[i])[0]
                
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                f'{img_name}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                    f'{img_name}.png')
                    imwrite(sr_img, save_img_path)
                    
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            # self.print_results()
            r_dir, f_name = os.path.split(save_img_path) 
            save_dir =  osp.join(self.opt['path']['visualization'], 'metric')
            os.makedirs(save_dir, exist_ok=True)
            excel_filename = f'/{current_iter}_metric_LS{ema_suffix}.xlsx'
            metric_r = evaluation_one_method_fast(dataset_name=self.dataset_name, data_dir=self.data_dir, result_dir=r_dir, 
                                                  save_dir= save_dir + excel_filename, Method='LS' , 
                                                  with_mean=True, crop_size=(int(self.target_size[0]), int(self.target_size[1])))
                                                  
            metric_f=['EN', 'SF', 'AG', 'SD', 'CC', 'SCD', 'MSE', 'PSNR', 'Qabf', 'Nabf']
            for index, metric in enumerate(metric_f):
                self.metric_results[metric] = metric_r[index][0]
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self.current_fusion_metric = self.metric_results['Qabf']
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
        return self.log_dict
    
    def save_current_log_img(self):
        visuals = self.get_current_visuals()
        grid_img = torch.cat((visuals['pred_img'].detach(),
                                    visuals['gt_SW'],
                                    visuals['gt_LW']), dim=0)
        grid_img = tensor2img(grid_img, min_max=(0, 1))
        save_img_path = os.path.join(self.opt['path']['visualization'],'img_fused_iter_{}.png'.format(self.current_iter))
        imwrite(grid_img, save_img_path)

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_img'] = self.pred_img
        out_dict['gt_SW'] = self.data["SW"]
        out_dict['gt_LW'] = self.data["LW"]
        return out_dict
    
    def get_current_model_score(self):
        return self.current_fusion_metric if hasattr(self, 'current_fusion_metric') else 0
    
    def save(self, epoch, current_iter):
        self.save_network([self.netfusion], 'net_fe_g', current_iter, param_key=['params_MMFusion'])

        if hasattr(self, 'ema') and self.ema:
            self.save_network([self.ema1.ema], 'net_fe_g_ema', current_iter, param_key=['params_MMFusion'])
            
        self.save_training_state(epoch, current_iter)
    
    def save_best(self, current_iter):
        logger = get_root_logger()
        self.save_network([self.netfusion], 'net_best', current_iter, param_key=['params_MMFusion'])
        
        if hasattr(self, 'ema') and self.ema:
            self.save_network([self.ema1.ema], 'net_best_ema', current_iter, param_key=['params_MMFusion'])
            
        logger.info(f"Saving new best-model")

    def remove(self, pre_iter):
        logger = get_root_logger()
        if pre_iter > 0:
            model_path = os.path.join(self.opt['path']['models'], f"net_fe_g_{pre_iter}.pth")
            state_path = os.path.join(self.opt['path']['training_states'], f'{pre_iter}.state')
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted old model file: {model_path}")
            else:
                logger.info(f"Old model file not found: {model_path}")
            
            if hasattr(self, 'ema') and self.ema:
                ema_model_path = os.path.join(self.opt['path']['models'], f"net_fe_g_ema_{pre_iter}.pth")
                if os.path.exists(ema_model_path):
                    os.remove(ema_model_path)
                    logger.info(f"Deleted old EMA model file: {ema_model_path}")
            
            if os.path.exists(state_path):
                os.remove(state_path)
                logger.info(f"Deleted old state file: {state_path}")
            else:
                logger.info(f"Old state file not found: {state_path}")

    def remove_best(self, best_iter):
        logger = get_root_logger()
        if best_iter > 0:
            model_path = os.path.join(self.opt['path']['models'], f"net_best_{best_iter}.pth")
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted old best-model file: {model_path}")
            else:
                logger.info(f"Old best-model file not found: {model_path}")

            if hasattr(self, 'ema') and self.ema:
                ema_model_path = os.path.join(self.opt['path']['models'], f"net_best_ema_{best_iter}.pth")
                if os.path.exists(ema_model_path):
                    os.remove(ema_model_path)
                    logger.info(f"Deleted old best EMA model file: {ema_model_path}")