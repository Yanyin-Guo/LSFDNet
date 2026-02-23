import torch
import torch.nn as nn 
import torch.nn.functional as F
import math
from ultralytics.nn.modules import Concat, Conv, A2C2f

def get_activation(activation_type):
    """Return the activation function given its name."""
    activation_type = activation_type.lower()
    return getattr(nn, activation_type, nn.ReLU)()

def _make_nConv(in_channels, out_channels, nb_conv, activation='ReLU'):
    """Stack nb_conv convolution blocks."""
    layers = [CBR(in_channels, out_channels, activation)]
    layers += [CBR(out_channels, out_channels, activation) for _ in range(nb_conv - 1)]
    return nn.Sequential(*layers)

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(activation)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class enFusion_block(nn.Module):
    def __init__(self, in_channels, Attn=False):
        super(enFusion_block, self).__init__()
        block = ResBlock
        if Attn:
            self.enhance = A2C2f(in_channels*2, in_channels, 2, True, 4)
        else:
            self.enhance = block(in_channels*2, in_channels)
        self.con = Concat(1)

    def forward(self, x1, x2):
        x_con = self.con([x1, x2])
        out = self.enhance(x_con)
        
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_conv, activation='ReLU'):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.body = _make_nConv(in_channels, out_channels, nb_conv, activation)
    def forward(self, x):
        return self.body(self.maxpool(x))

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(F_g, F_x))
        self.concat = Concat(1)
        self.conv = nn.Sequential(
            nn.Conv2d(F_x * 2, F_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(F_x),
            nn.ReLU(inplace=True)
        )
    def forward(self, g, x):
        avg_pool = lambda z: F.avg_pool2d(z, z.shape[2:]) # pool to [B,C,1,1]
        scale = torch.sigmoid((self.mlp_x(avg_pool(x)) + self.mlp_g(avg_pool(g))) / 2).unsqueeze(2).unsqueeze(3)
        feat = self.conv(self.concat([x, g])) * scale
        return feat

class SCA(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        def spa_mlp(f): # 3x3->1x1空域注意力
            return nn.Sequential(
                nn.Conv2d(f, f, kernel_size=3, padding=1),  
                nn.BatchNorm2d(f),
                nn.ReLU(inplace=True),
                nn.Conv2d(f, 1, kernel_size=1),  # 输出一个通道的空间注意力图  
            )
        self.mlp_x = spa_mlp(F_x)
        self.mlp_g = spa_mlp(F_g)
        self.sigmoid = nn.Sigmoid()
        self.concat = Concat(1)
        self.conv = nn.Sequential(
            nn.Conv2d(F_x * 2, F_x, kernel_size=3, padding=1),
            nn.BatchNorm2d(F_x),
            nn.ReLU(inplace=True)
        )
    def forward(self, g, x):
        scale = self.sigmoid((self.mlp_x(x) + self.mlp_g(g)) / 2).expand_as(x)
        return self.conv(self.concat([x, g])) * scale   

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nb_conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.body = _make_nConv(in_channels, out_channels, nb_conv, activation)

    def forward(self, x, skip_x):
        return self.body(torch.cat([skip_x, self.up(x)], dim=1)) 

class UpBlock_SCA(nn.Module):
    def __init__(self, in_channels, out_channels, nb_conv, activation='ReLU'):
        super().__init__()
        ch = in_channels // 2
        self.up = nn.Upsample(scale_factor=2)
        self.SCA = SCA(ch, ch)
        self.body = _make_nConv(ch, out_channels, nb_conv, activation)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x, skip_x):
        skip_x = self.SCA(self.up(x), skip_x)
        return self.body(skip_x)

class UpBlock_CCA(nn.Module):
    def __init__(self, in_channels, out_channels, nb_conv, activation='ReLU'):
        super().__init__()
        ch = in_channels // 2
        self.up = nn.Upsample(scale_factor=2)
        self.CCA = CCA(ch, ch)
        self.body = _make_nConv(in_channels, out_channels, nb_conv, activation)
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x, skip_x):
        skip_x = self.CCA(self.up(x), skip_x)
        return self.body(skip_x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
            if stride != 1 or out_channels != in_channels
            else None
        )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + residual)

class ResBlock_con(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.con_up = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
        self.bn_up = nn.BatchNorm2d(in_channels)
        self.relu_up = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride), nn.BatchNorm2d(out_channels))
            if stride != 1 or out_channels != in_channels
            else None
        )

    def forward(self, x):
        x = self.relu_up(self.bn_up(self.con_up(x)))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + residual)

class SA(nn.Module):  
    def __init__(self, F_x):  
        super().__init__()  
        self.mlp_x = nn.Sequential(  
            nn.Conv2d(F_x, F_x, kernel_size=3, padding=1),  
            nn.BatchNorm2d(F_x),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(F_x, 1, kernel_size=1),  # 输出一个通道的空间注意力图  
            nn.BatchNorm2d(1) 
        )  
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):  
        spatial_att_x = self.mlp_x(x)  
        scale = self.sigmoid(spatial_att_x)  
        x_after_spatial = x * scale  
        return x_after_spatial 

class ResBlock_SA(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.SA = SA(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride), nn.BatchNorm2d(out_channels))
            if stride != 1 or out_channels != in_channels
            else None
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x):
        out = self.SA(x)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.relu(out + residual)

class AdaptivePool2d(nn.Module):
    def __init__(self, out_h, out_w, pool_type='avg'):
        super().__init__()
        self.out_h, self.out_w, self.pool_type = out_h, out_w, pool_type
    def forward(self, x):
        if x.size(2) > self.out_h or x.size(3) > self.out_w:
            stride_h, stride_w = x.size(2) // self.out_h, x.size(3) // self.out_w
            k_h = x.size(2) - (self.out_h - 1) * stride_h
            k_w = x.size(3) - (self.out_w - 1) * stride_w
            op = nn.AvgPool2d if self.pool_type == 'avg' else nn.MaxPool2d
            return op((k_h, k_w), stride=(stride_h, stride_w))(x)
        return x

class LearnableWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    def forward(self, x1, x2):
        return x1 * self.w1 + x2 * self.w2

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return x * self.bias   

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert d_k % h == 0
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.qkv_proj = nn.ModuleList([nn.Linear(d_model, h * self.d_k) for _ in range(4)])
        self.v_proj = nn.ModuleList([nn.Linear(d_model, h * self.d_v) for _ in range(2)])
        self.out_proj = nn.ModuleList([nn.Linear(h * self.d_v, d_model) for _ in range(2)])
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        rgb, ir = self.ln[0](x[0]), self.ln[1](x[1])
        B, N = rgb.shape[:2]
        q_vis, k_vis, q_ir, k_ir = [proj(rgb) if i < 2 else proj(ir)
                                    for i, proj in enumerate(self.qkv_proj)]
        v_vis, v_ir = self.v_proj[0](rgb), self.v_proj[1](ir)
        def attn(q, k, v):
            q = q.view(B, N, self.h, self.d_k).permute(0, 2, 1, 3)
            k = k.view(B, N, self.h, self.d_k).permute(0, 2, 3, 1)
            v = v.view(B, N, self.h, self.d_v).permute(0, 2, 1, 3)
            att = torch.matmul(q, k) / math.sqrt(self.d_k)
            att = self.attn_drop(torch.softmax(att, -1))
            out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(B, N, self.h * self.d_v)
            return self.resid_drop(out)
        out_vis = self.out_proj[0](attn(q_ir,k_vis,v_vis))
        out_ir = self.out_proj[1](attn(q_vis,k_ir,v_ir))
        return [out_vis, out_ir]

class CrossTransformer(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        super().__init__()
        self.loops = loops_num
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_vis = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                     nn.GELU(),
                                     nn.Linear(block_exp * d_model, d_model),
                                     nn.Dropout(resid_pdrop))
        self.mlp_ir = nn.Sequential(nn.Linear(d_model, block_exp * d_model),
                                    nn.GELU(),
                                    nn.Linear(block_exp * d_model, d_model),
                                    nn.Dropout(resid_pdrop))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.coefficients = nn.ModuleList([LearnableCoefficient() for _ in range(8)])

    def forward(self, x):
        rgb, ir = x[0], x[1]
        for _ in range(self.loops):
            rgb_out, ir_out = self.crossatt([rgb, ir])
            rgb_att = self.coefficients[0](rgb) + self.coefficients[1](rgb_out)
            ir_att = self.coefficients[2](ir) + self.coefficients[3](ir_out)
            rgb = self.coefficients[4](rgb_att) + self.coefficients[5](self.mlp_vis(self.ln1(rgb_att)))
            ir = self.coefficients[6](ir_att) + self.coefficients[7](self.mlp_ir(self.ln2(ir_att)))
        return [rgb, ir]   

class CrossFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=16, horz_anchors=16, h=8, block_exp=4, n_layer=1, embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        self.pos_emb_vis = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, d_model))
        self.pos_emb_ir = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, d_model))
        self.avgpool = AdaptivePool2d(vert_anchors, horz_anchors, 'avg')
        self.maxpool = AdaptivePool2d(vert_anchors, horz_anchors, 'max')
        self.vis_coefficient = LearnableWeights()
        self.ir_coefficient = LearnableWeights()
        self.crosstransformer = nn.Sequential(
            *[CrossTransformer(d_model, d_model, d_model, h, block_exp, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)]
        )
        self.concat = Concat(dimension=1)
        self.conv1x1_out = Conv(c1=d_model * 2, c2=d_model, k=1, s=1, p=0, g=1, act=True)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, x_vi, x_ir):
        bs, c, h, w = x_vi.shape
        pooled_rgb = self.vis_coefficient(self.avgpool(x_vi), self.maxpool(x_vi))
        pooled_ir = self.ir_coefficient(self.avgpool(x_ir), self.maxpool(x_ir))
        new_c, new_h, new_w = pooled_rgb.shape[1:]
        rgb_fea_flat = pooled_rgb.view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_vis
        ir_fea_flat = pooled_ir.view(bs, new_c, -1).permute(0, 2, 1) + self.pos_emb_ir
        rgb_fea_flat, ir_fea_flat = self.crosstransformer([rgb_fea_flat, ir_fea_flat])
        rgb_fea_CFE = rgb_fea_flat.view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        ir_fea_CFE = ir_fea_flat.view(bs, new_h, new_w, new_c).permute(0, 3, 1, 2)
        interp_mode = 'nearest' if self.training else 'bilinear'
        rgb_fea_CFE = F.interpolate(rgb_fea_CFE, size=[h, w], mode=interp_mode) + x_vi
        ir_fea_CFE = F.interpolate(ir_fea_CFE, size=[h, w], mode=interp_mode) + x_ir
        fuse = self.concat([rgb_fea_CFE, ir_fea_CFE])
        new_fuse = self.conv1x1_out(fuse)
        return new_fuse