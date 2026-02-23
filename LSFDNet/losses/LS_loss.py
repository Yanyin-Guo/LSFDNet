import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from ultralytics.utils.loss import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from ultralytics.utils.loss import *
from ultralytics.utils.ops import xywhn2xyxy
from scripts.util import RGB2YCrCb, YCrCb2RGB

import cv2
import os
from basicsr.utils import get_root_logger,tensor2img,imwrite,img2tensor
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

class fuse_v8DetectionLoss(nn.Module):
    def __init__(self, device, tal_topk=10, nc=6):  # model must be de-paralleled
        super(fuse_v8DetectionLoss, self).__init__()
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp_box = 7.5
        self.hyp_cls = 0.5
        self.hyp_dfl = 1.5
        self.stride = torch.tensor([8.0, 16.0, 32.0], device= self.device)  # model strides
        self.nc = nc  # number of classes
        self.no = self.nc + 16 * 4
        self.reg_max = 16
        self.use_dfl = self.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=tal_topk, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(self.reg_max).to(self.device)
        self.proj = torch.arange(self.reg_max, dtype=torch.float, device=self.device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                if n := matches.sum():
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, xywh=False)
        
    def forward(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            # pred_scores.detach().sigmoid() * 0.8 + dfl_conf.unsqueeze(-1) * 0.2,
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp_box  # box gain
        loss[1] *= self.hyp_cls  # cls gain
        loss[2] *= self.hyp_dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

@LOSS_REGISTRY.register()
class Fusionloss_OE(nn.Module):
    def __init__(self, device='cuda', nc=1):
        super(Fusionloss_OE, self).__init__()
        self.device = torch.device('cuda' if device == 'cuda' else 'cpu')
        self.sobelconv = Sobelxy()
        self.mse_criterion = nn.MSELoss(reduction='mean')

    def forward(self, alpha, beta, sigma, gamma, image_SW, image_LW, generate_img, batch):
        B, C, H, W = image_SW.shape
        
        image_y = image_SW[:, :1, :, :]
        image_th = torch.clamp(torch.pow(image_LW[:, :1, :, :], gamma), 0.0, 1.0)
        x_in_mean = torch.add(image_y * 0.5, image_th, alpha=0.5)
        x_in_mean_7 = torch.add(image_y * 0.3, image_th, alpha=0.7) # 用于 Object

        y_grad = self.sobelconv(image_y)
        x_in_mean_grad = self.sobelconv(x_in_mean)
        generate_img_grad = self.sobelconv(generate_img)
        x_in_mean_7_grad = self.sobelconv(x_in_mean_7) 
        
        x_in_mean_max = torch.max(image_y, x_in_mean)
        x_grad_joint = torch.max(y_grad, x_in_mean_grad)

        loss_in_global = 0.5 * F.l1_loss(generate_img, x_in_mean_max)
        loss_in_mse_global = 0.25 * self.mse_criterion(generate_img, image_y) + \
                             0.25 * self.mse_criterion(generate_img, image_LW)
        loss_grad_global = F.l1_loss(generate_img_grad, x_grad_joint)
        
        loss_fusion_global = alpha * (loss_in_global + loss_in_mse_global) + (1 - alpha) * loss_grad_global
        mask = torch.zeros((B, 1, H, W), device=self.device)
        if batch['bboxes'] is not None and len(batch['bboxes']) > 0:
            bboxes = xywhn2xyxy(batch['bboxes'], W, H) # 确保这个函数返回的是 Tensor 且在 GPU 上
            batch_idx = batch['batch_idx'] # [N]
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i].int() # 转整数坐标
                if (x2 - x1) * (y2 - y1) < 20:
                    continue
                b_i = int(batch_idx[i])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                mask[b_i, :, y1:y2, x1:x2] = 1.0
        object_in_max = torch.max(image_y, x_in_mean_7) # 全图 max
        loss_map_intensity = F.l1_loss(generate_img, object_in_max, reduction='none')
        ob_grad_max = torch.max(y_grad, x_in_mean_7_grad) # 全图梯度 max (注意: 这里用 y_grad 代替了 ob_SW_grad，效果更好且快)
        loss_map_grad = F.l1_loss(generate_img_grad, ob_grad_max, reduction='none')

        num_object_pixels = mask.sum()
        
        if num_object_pixels > 0:
            loss_in_object = (loss_map_intensity * mask).sum() / num_object_pixels
            loss_grad_object = (loss_map_grad * mask).sum() / num_object_pixels
        else:
            loss_in_object = torch.tensor(0.0, device=self.device)
            loss_grad_object = torch.tensor(0.0, device=self.device)

        loss_fusion_object = beta * loss_in_object + (1 - beta) * loss_grad_object
        loss_fusion = sigma * loss_fusion_global + (1 - sigma) * loss_fusion_object

        return loss_fusion, loss_fusion_global, loss_fusion_object

@LOSS_REGISTRY.register()
class FusionDetloss_OE(nn.Module):
    def __init__(self, device='cuda', nc=1):
        super(FusionDetloss_OE, self).__init__()
        self.device = torch.device('cuda' if device == 'cuda' else 'cpu')
        self.sobelconv = Sobelxy()
        self.FusionLOSS = Fusionloss_OE(device=device, nc=nc)
        self.det_v8DetectionLoss = fuse_v8DetectionLoss(self.device, nc=nc)

    def forward(self, alpha, beta, sigma, gamma, image_SW, image_LW, generate_img, preds, batch):
        self.loss_fusion, self.loss_fusion_global, self.loss_fusion_object = self.FusionLOSS(alpha, beta, sigma, gamma, image_SW, image_LW, generate_img, batch)
        self.loss_det, self.loss_items = self.det_v8DetectionLoss(preds, batch)

        return self.loss_det, self.loss_fusion