import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import LOSS_REGISTRY
from ultralytics.utils.loss import *
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        b, c, w, h = x.shape
        batch_list = []
        for i in range(b):
            tensor_list = []
            for j in range(c):
                sobelx_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weightx, padding=1)
                sobely_0 = F.conv2d(torch.unsqueeze(torch.unsqueeze(x[i, j, :, :], 0), 0), self.weighty, padding=1)
                add_0 = torch.abs(sobelx_0) + torch.abs(sobely_0)
                tensor_list.append(add_0)

            batch_list.append(torch.stack(tensor_list, dim=1))

        return torch.cat(batch_list, dim=0)

@LOSS_REGISTRY.register()
class Fusionloss_OE(nn.Module):
    def __init__(self):
        super(Fusionloss_OE, self).__init__()
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, b,c ,image_vis, image_ir, generate_img, label, LW_th):
        alpha = b
        beta = c
        image_y = image_vis
        B, C, H, W = image_vis.shape
        image_ir = image_ir.expand(B, C, H, W)
        image_th = LW_th.expand(B, C, H, W)

        Lssim, L1loss = torch.zeros(B,10),torch.zeros(B,10)
        loss_MSE, loss_in, loss_grad, loss_label, loss_ss = torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B)
        ls,lin=torch.zeros(B),torch.zeros(B)

        x_in_mean = torch.add(image_y*0.5, image_th, alpha=0.5)
        x_in_mean_7 = torch.add(image_y*0.3, image_th, alpha=0.7)
        x_in_mean_max = torch.max(image_y, x_in_mean)
        # Gradient
        y_grad = self.sobelconv(image_y)
        x_in_mean_grad = self.sobelconv(x_in_mean)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, x_in_mean_grad)

        for b in range(B):
            loss_in[b] = 0.5 * F.l1_loss(generate_img[b], x_in_mean_max[b])
            loss_grad[b]  = F.l1_loss(generate_img_grad[b], x_grad_joint[b])
            loss_MSE[b] =  0.25 * self.mse_criterion(generate_img[b],image_y[b]) + 0.25 * self.mse_criterion(generate_img[b],image_ir[b]) 

        loss_global = alpha*(loss_in+loss_MSE) + (1-alpha)*loss_grad
        #label
        for b,batch_label in enumerate(label):
            for i,single_label in enumerate(batch_label): 
                if(single_label[1]==0 and single_label[2]==0 and single_label[3]==0 and single_label[4]== 0):
                    continue
                else:
                    object_vis = image_y[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_ir =  image_th[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_mean =  x_in_mean_7[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_generate = generate_img[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]

                    object_vis = torch.unsqueeze(object_vis,0)
                    object_ir =  torch.unsqueeze(object_ir,0)
                    object_mean = torch.unsqueeze(object_mean,0)
                    object_generate = torch.unsqueeze(object_generate,0)

                    object_in_max = torch.maximum(object_vis, object_mean)

                    ob_vis_grad = self.sobelconv(object_vis)
                    ob_generate_grad = self.sobelconv(object_generate)
                    ob_mean_grad = self.sobelconv(object_mean)
                    ob_grad_max = torch.maximum(ob_vis_grad, ob_mean_grad)

                    Lssim[b][i] = F.l1_loss(ob_generate_grad, ob_grad_max)
                    L1loss[b][i] = F.l1_loss(object_generate, object_in_max)

                
            exist = (Lssim[b] != 0) | (L1loss[b] != 0)
            if exist.sum()==0:
                loss_label[b] = 0
                ls[b]=0
                lin[b]=0
            else:
                ls[b]=Lssim[b].sum()/exist.sum()
                lin[b]=L1loss[b].sum()/exist.sum()
                loss_label[b] = (1-beta) * ls[b] + beta * lin[b]


        return loss_ss, loss_global, loss_label, loss_in, loss_grad,ls,lin

class Pre_batch(nn.Module):
    def __init__(self):  # model must be de-paralleled
        super(Pre_batch, self).__init__()
    
    def pre_process(self, batch):
        batch['cls'] = self.process_cls(batch['cls']) 
        batch['bboxes'] = self.remove_empty(batch['bboxes']) 
        batch['batch_idx'] = self.process_batch_idx(batch['batch_idx']) 
        batch['ori_shape'] = [tensor.tolist() for tensor in batch['ori_shape']]
        batch['resized_shape'] = [tensor.tolist() for tensor in batch['resized_shape']]
        if 'ratio_pad' in batch:
            if isinstance(batch['ratio_pad'], torch.Tensor):  
                batch['ratio_pad'] = batch['ratio_pad'].cpu().numpy().astype(int).tolist()  
        return batch
    
    def process_cls(self, tensor_list, value_to_remove=999):  
        flattened_tensor = tensor_list.squeeze(2)     
        result_tensors = []   
        for row in flattened_tensor:  
            modified_tensor = self.remove_after_value(row, value_to_remove) 
            if modified_tensor.numel() > 0:  
                modified_tensor = modified_tensor.view(-1, 1)  
                result_tensors.append(modified_tensor) 
        if len(result_tensors) > 0: 
            result_tensors = torch.cat(result_tensors, dim=0) 
        else:
            result_tensors = torch.tensor([[]])  
        return result_tensors 
    
    def process_batch_idx(self, tensor_list, value_to_remove=999):  
        result_tensors = []  
        num = 0
        for i in range(tensor_list.size(0)):   
            row = tensor_list[i]  
            modified_row = self.remove_after_value(row, value_to_remove)  
            modified_row += num
            num += 1  
            result_tensors.append(modified_row)  
        if len(result_tensors) > 0:     
            result_tensors = torch.cat(result_tensors, dim=0) 
        else:
            result_tensors = torch.tensor([])
        return result_tensors 

    def remove_after_value(self, tensor, value):   
        index = (tensor == value).nonzero(as_tuple=True)  
        if index[0].numel() > 0:  
            last_index = index[0][0]   
            return tensor[:last_index] 
        
    def remove_empty(self, tensor):  
        valid_rows = []  
        for i in range(tensor.size(0)):  
            row = tensor[i, :, :]  
            for row_z in row:
                if not torch.all(row_z == 0):  
                    valid_rows.append(row_z)  
        if valid_rows:  
            result_tensor = torch.stack(valid_rows, dim=0)  
            return result_tensor.reshape(-1, 4)   
        else:  
            return torch.empty((0, 4))   

class fuse_v8DetectionLoss(nn.Module):
    def __init__(self, device, tal_topk=10):  # model must be de-paralleled
        super(fuse_v8DetectionLoss, self).__init__()
        self.pre_process = Pre_batch()
        self.device = device
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp_box = 7.5
        self.hyp_cls = 0.5
        self.hyp_dfl = 1.5
        self.stride = torch.tensor([8.0, 16.0, 32.0], device= self.device)  # model strides
        self.nc = 1  # number of classes
        self.no = 1 + 16 * 4
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

        batch = self.pre_process.pre_process(batch)
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
class FusionDetloss_OE(nn.Module):
    def __init__(self, device='cuda'):
        super(FusionDetloss_OE, self).__init__()
        self.device = torch.device('cuda' if device == 'cuda' else 'cpu')
        self.sobelconv = Sobelxy()
        self.mse_criterion = torch.nn.MSELoss()
        self.det_v8DetectionLoss = fuse_v8DetectionLoss(self.device)

    def forward(self, b,c ,image_vis, image_ir, generate_img, preds, batch, label, LW_th):
        alpha = b
        beta = c
        image_y = image_vis
        B, C, H, W = image_vis.shape
        image_ir = image_ir.expand(B, C, H, W)
        image_th = LW_th.expand(B, C, H, W)

        Lssim, L1loss = torch.zeros(B,10),torch.zeros(B,10)
        loss_MSE, loss_in, loss_grad, loss_label, loss_ss = torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B),torch.zeros(B)
        ls,lin=torch.zeros(B),torch.zeros(B)

        x_in_mean = torch.add(image_y*0.5, image_th, alpha=0.5)
        x_in_mean_7 = torch.add(image_y*0.3, image_th, alpha=0.7)
        x_in_mean_max = torch.max(image_y, x_in_mean)
        # Gradient
        y_grad = self.sobelconv(image_y)
        x_in_mean_grad = self.sobelconv(x_in_mean)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.maximum(y_grad, x_in_mean_grad)

        for b in range(B):
            loss_in[b] = 0.5 * F.l1_loss(generate_img[b], x_in_mean_max[b])
            loss_grad[b]  = F.l1_loss(generate_img_grad[b], x_grad_joint[b])
            loss_MSE[b] =  0.25 * self.mse_criterion(generate_img[b],image_y[b]) + 0.25 * self.mse_criterion(generate_img[b],image_ir[b]) 

        loss_global = alpha*(loss_in+loss_MSE) + (1-alpha)*loss_grad
        #label
        for b,batch_label in enumerate(label):
            for i,single_label in enumerate(batch_label): 
                if(single_label[1]==0 and single_label[2]==0 and single_label[3]==0 and single_label[4]== 0):
                    continue
                else:
                    object_vis = image_y[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_ir =  image_th[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_mean =  x_in_mean_7[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]
                    object_generate = generate_img[b][:,single_label[2]:single_label[4],single_label[1]:single_label[3]]

                    object_vis = torch.unsqueeze(object_vis,0)
                    object_ir =  torch.unsqueeze(object_ir,0)
                    object_mean = torch.unsqueeze(object_mean,0)
                    object_generate = torch.unsqueeze(object_generate,0)

                    object_in_max = torch.maximum(object_vis, object_mean)

                    ob_vis_grad = self.sobelconv(object_vis)
                    ob_generate_grad = self.sobelconv(object_generate)
                    ob_mean_grad = self.sobelconv(object_mean)
                    ob_grad_max = torch.maximum(ob_vis_grad, ob_mean_grad)

                    Lssim[b][i] = F.l1_loss(ob_generate_grad, ob_grad_max)
                    L1loss[b][i] = F.l1_loss(object_generate, object_in_max)

                
            exist = (Lssim[b] != 0) | (L1loss[b] != 0)
            if exist.sum()==0:
                loss_label[b] = 0
                ls[b]=0
                lin[b]=0
            else:
                ls[b]=Lssim[b].sum()/exist.sum()
                lin[b]=L1loss[b].sum()/exist.sum()
                loss_label[b] = (1-beta) * ls[b] + beta * lin[b]

        self.loss_det, self.loss_items = self.det_v8DetectionLoss(preds, batch)
        self.loss_det = self.loss_det.to('cpu', dtype=torch.float)


        return self.loss_det, loss_ss, loss_global, loss_label, loss_in, loss_grad,ls, lin
