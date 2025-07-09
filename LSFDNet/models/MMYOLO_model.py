import time
import cv2
from pathlib import Path
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
from core.Metric_fusion.eval_one_method import evaluation_one_method_fast
import core.weights_init
from tqdm import tqdm
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils import LOGGER
from losses.LS_loss import Pre_batch
from losses.LS_loss import fuse_v8DetectionLoss
#from ultralytics import YOLO

@MODEL_REGISTRY.register()
class MMYOLO(BaseModel):
    def __init__(self, opt):
        super(MMYOLO, self).__init__(opt)
        logger = get_root_logger()
        # define network and load pretrained models
        self.netfusion =  build_network(opt['network_detfusion'])
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

        load_path = self.opt['path'].get('pretrain_network_MMYOLO', None)
        if load_path is not None:
            self.load_network(self.netfusion, load_path, self.opt['path'].get('strict_load_g', True), 'params_MMYOLO')
            logger.info(f"Pretrained model is successfully loaded from {opt['path']['pretrain_network_MMYOLO']}")

        self.current_iter = 0
        self.gama = opt['train']["a"]
        self.b=opt['train']["b"]
        self.c=opt['train']["c"]
        # self._initialize_weights()  

        self.seen = 0
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.plots = False
        self.single_cls = True
        self.conf = opt['Det_labels']['conf']
        self.nc = opt['Det_labels']['nc']
        self.iou = opt['Det_labels']['iou']
        self.iou_thres = opt['Det_labels']['iou_thres']
        self.task = "detect"
        self.matrix = np.zeros((self.nc + 1, self.nc + 1)) if self.task == "detect" else np.zeros((self.nc, self.nc))
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        self.save_json = True
        self.save_txt = True
        self.jdict = []
        self.class_map = list(range(1, self.nc + 1))
        self.names = {i: name for i, name in enumerate(opt['Det_labels']['names'])}
        self.save_conf = True
        self.save_dir = Path(opt['path']['experiments_root'] if 'experiments_root' in opt['path'] else opt['path']['results_root'] )
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.metrics.names = self.names
        self.metrics.plot = self.plots
 
    def transfer_weights(self, small_model, large_model):  
        small_model_state_dict = small_model.state_dict()  
        large_model_state_dict = large_model.state_dict()   
         
        for name, param in small_model_state_dict.items():  
            if name in large_model_state_dict:  
                large_model_state_dict[name].data.copy_(param.data)  

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
        self.loss_dict_all['loss_det'] = []
        self.loss_dict_all['loss_fusion'] = []
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
        self.set_device(train_data[1]['img'])
        self.batch = train_data[1]

    def cal_lam(self, citer):
        if citer < self.opt['train']['loss_dict'][0] or citer > self.opt['train']['loss_dict'][1]:
            lam = 0.9
        else:
            if citer < self.opt['train']['loss_dict'][0] + 1000:
                lam = 0.9 - ((citer-self.opt['train']['loss_dict'][0])//100) * 0.2
                lam = lam if lam > 0.1 else 0.1
            else:
                lam = 0.1 + ((citer-1000-self.opt['train']['loss_dict'][0])//375) * 0.1
                lam = lam if lam < 0.9 else 0.9
        return lam
    
    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        self.current_iter = current_iter
        self.optimizer_g_netfusion.zero_grad()
        loss_dict = OrderedDict() 
        self.pred_img, self.preds = self.netfusion(self.data)
        loss_det, loss_ss, loss_back, loss_label, loss_in,loss_grad, ls, lin = self.loss_LS(self.b,self.c,image_vis=self.data["SW"], image_ir=self.data["LW"],  generate_img=self.pred_img, preds=self.preds, batch =self.batch, label=self.data["label"], LW_th=self.data["LW_th"])
        for i in range(len(loss_label)):
            if loss_label[i] !=0 :
                loss_ss[i] = self.gama * loss_back[i] + (1-self.gama) * loss_label[i]
            else:
                loss_ss[i] = loss_back[i]
        loss_det = loss_det.repeat(ls.size(0))/ls.size(0)
        self.lam = self.cal_lam(current_iter) 
        loss_all = (1-self.lam)*loss_ss + self.lam * loss_det * 0.01
        loss_fs=loss_all.mean()
        loss_fs.backward()
        self.optimizer_g_netfusion.step()

        loss_dict['loss_all'] = loss_all
        loss_dict['loss_fusion'] = loss_ss
        loss_dict['loss_det'] = loss_det
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
            self.pred_img, self.preds= self.netfusion(self.data)
        self.netfusion.train()

        return self.pred_img, self.preds
    
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
        save_plot_det = self.opt['val']['plot_det']
        self.names = self.opt['Det_labels']['names']
        self.conf_test = self.opt['Det_labels']['conf_save']
        self.pre_batch = Pre_batch()
        self.init_metrics()
        self.det = fuse_v8DetectionLoss(self.device)
        logger = get_root_logger()

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
            img_name = osp.splitext(val_data[1]['im_name'][0])[0]
            self.feed_data(val_data)
            self.pred_img, self.preds = self.test()
            self.preds = self.postprocess(self.preds)
            self.batch = self.pre_pre_batch(self.batch)
            self.batch = self.pre_batch.pre_process(self.batch)
            self.update_metrics(self.preds, self.batch)
            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals['pred_img'].detach(), min_max=(0, 1))
            metric_data['img'] = sr_img
            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                             f'{img_name}.jpg')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                 f'{img_name}.jpg')
                imwrite(sr_img, save_img_path)
            if save_plot_det:
                self.label_img = self.plot_det(self.preds, sr_img)
                if self.opt['is_train']:
                    save_img_label_path = osp.join(self.opt['path']['visualization'], dataset_name + '_label', str(current_iter),
                                             f'{img_name}.jpg')
                else:
                    if self.opt['val']['suffix']:
                        save_img_label_path = osp.join(self.opt['path']['visualization'], dataset_name + '_label', str(current_iter),
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.jpg')
                    else:
                        save_img_label_path = osp.join(self.opt['path']['visualization'], dataset_name + '_label', str(current_iter),
                                                 f'{img_name}.jpg')
                imwrite(self.label_img, save_img_label_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            stats = self.get_stats()
            self.speed = None
            self.finalize_metrics()
            log_str = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
            pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
            log_str1 = pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
            logger.info(log_str)
            logger.info(log_str1)

            # self.print_results()
            r_dir, f_name = os.path.split(save_img_path) 
            save_dir =  osp.join(self.opt['path']['visualization'], 'metric')
            os.makedirs(save_dir, exist_ok=True)
            metric_r = evaluation_one_method_fast(dataset_name='NSLSR_test', data_dir='/data', result_dir=r_dir, save_dir= save_dir+ f'/{current_iter}' + '_metric_C2FM.xlsx', Method='C2FM' , with_mean=True)
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
        grid_img = tensor2img(grid_img, min_max=(0, 1))
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
        self.save_network([self.netfusion], 'net_fe_g', current_iter, param_key=['params_MMYOLO'])
        self.save_training_state(epoch, current_iter)

    def init_metrics(self):
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def plot_det(self, preds, img):
        pred = preds[0]
        img = np.stack((img,) * 3, axis=-1)  
        for (x1, y1, x2, y2, conf, cls) in pred.cpu().detach().numpy():
            if conf >= self.conf_test:   
                color = (0, 0, 255) 
                color1 = (255, 255, 255) 
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  

                label = f"{self.names[int(cls)]}: {conf:.2f}"  
                background_tl = (x1, y1-18)   
                background_br = (x1+93, y1)
 
                cv2.rectangle(img, background_tl, background_br, color, thickness=-1) 
                cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 1, lineType=cv2.LINE_AA)
                
        return img

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            cls = cls.to(self.device)
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.save_txt:
                self.save_one_txt(
                    predn,
                    self.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(batch['im_file'][si]).stem}.txt",
                )
    
    def pre_pre_batch(self, batch):
        batch['ori_shape'] = [torch.cat(batch['ori_shape'])]
        batch['resized_shape'] = [torch.cat(batch['resized_shape'])]
        batch['ratio_pad'] = batch['ratio_pad'].squeeze(0)

        return batch
    
    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor]): Processed predictions after NMS.
        """
        return ops.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            labels=[],
            nc=self.nc,
            multi_label=True,
            agnostic=False,
            max_det=300,
            end2end=False,
            rotated=False
        )

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict): Batch data containing images and annotations.

        Returns:
            (dict): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = bbox.to(self.device)
            cls = cls.to(self.device)
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}
    def _prepare_pred(self, pred, pbatch):
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            pbatch (dict): Prepared batch information.

        Returns:
            (torch.Tensor): Prepared predictions in native space.
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    
    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    
    def save_one_txt(self, predn, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (tuple): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            filename (str): Image filename.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )
    def finalize_metrics(self, *args, **kwargs):
        """
        Set final values for metrics speed and confusion matrix.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        
    def get_stats(self):
        """
        Calculate and return metrics statistics.

        Returns:
            (dict): Dictionary containing metrics results.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}
    
    def print_results(self):
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if not self.is_train and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )