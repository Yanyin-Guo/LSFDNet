import torch
from torch import nn
import math
from basicsr.utils.registry import ARCH_REGISTRY


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, act_fn='relu', use_pad=True):  
    layers = []  
    if use_pad:  
        layers.append(nn.ReflectionPad2d((kernel_size // 2, kernel_size // 2,  
                                          kernel_size // 2, kernel_size // 2)))  
    layers.extend([  
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),  
        nn.BatchNorm2d(out_channels),  
    ])  
    if act_fn.lower() == 'relu':  
        layers.append(nn.ReLU())  
    elif act_fn.lower() == 'prelu':  
        layers.append(nn.PReLU())  
    return nn.Sequential(*layers)  

def mlp_block(dim, hidden_mult=2):  
    return nn.Sequential(  
        nn.Linear(dim, dim * hidden_mult),  
        nn.ReLU(),  
        nn.Linear(dim * hidden_mult, dim)  
    ) 

class FeatureExtractorBase(nn.Module):  
    def __init__(self):  
        super(FeatureExtractorBase, self).__init__()  
        self.Base = nn.Sequential(  
            conv_block(1, 4, 3),  
            conv_block(4, 8, 3),  
            conv_block(8, 8, 3),  
        )  

    def forward(self, x):  
        return self.Base(x)  

class FeatureExtractorMulNet(nn.Module):  
    def __init__(self):  
        super(FeatureExtractorMulNet, self).__init__()  
        self.Mul_1 = nn.Sequential(  
            conv_block(8, 16, 3),  
            conv_block(16, 16, 3),  
            conv_block(16, 32, 3),  
            conv_block(32, 32, 3),  
            conv_block(32, 16, 3),  
            conv_block(16, 16, 3),  
            conv_block(16, 8, 3),  
        )  

    def forward(self, x):  
        return self.Mul_1(x)  

class PositionalEncoding(nn.Module):  
    def __init__(self, d_model, dropout, max_len=5000):  
        super(PositionalEncoding, self).__init__()  
        self.dropout = nn.Dropout(p=dropout)  
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  
        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  
        pe = pe.unsqueeze(0)  
        self.register_buffer('pe', pe)  

    def forward(self, x):  
        x = x + self.pe[:, :x.size(1)].to(x.device)  
        return self.dropout(x)  

def apply_attention(input_s, MHA, is_cross=False, query=None, key_value=None):  
    embeding_dim = input_s.size(2) if not is_cross else query.size(2)  
    if MHA.training:
        PE = PositionalEncoding(embeding_dim, 0.1).train()
    else:
        PE = PositionalEncoding(embeding_dim, 0.1).eval() 
    if is_cross:  
        q = PE(query)  
        k = PE(key_value)  
        v = key_value  
    else:  
        q = PE(input_s)  
        k = PE(input_s)  
        v = input_s 
    attention_output = MHA(q, k, v)[0]  
    return attention_output + (query if is_cross else input_s)  

class CrossFusion(nn.Module):  
    def __init__(self, size=8, dropout=0.1, channel=4):  
        super(CrossFusion, self).__init__()  
        self.size = size  
        self.dim = size * size * channel  
        self.SWIREncoder = nn.Sequential(  
            conv_block(8, 4, 3, act_fn='prelu'),  
            conv_block(4, 4, 3, act_fn='prelu'),  
        )  
        self.LWIREncoder = nn.Sequential(  
            conv_block(8, 4, 3, act_fn='prelu'),  
            conv_block(4, 4, 3, act_fn='prelu'),  
        )  
        
        self.mlp_SA_SWIR = mlp_block(self.dim)  
        self.mlp_SA_LWIR = mlp_block(self.dim)  
        self.mlp_CA_SWIR = mlp_block(self.dim)  
        self.mlp_CA_LWIR = mlp_block(self.dim)  

        self.SA_SWIR, self.SA_LWIR, self.CA_SWIR, self.CA_LWIR = [  
            nn.MultiheadAttention(self.dim, 1, dropout) for _ in range(4)  
        ]  
        self.Decoder_mul = nn.Sequential(  
            conv_block(8, 8, 3, act_fn='prelu'),  
            conv_block(8, 16, 3, act_fn='prelu'),  
            conv_block(16, 16, 3, act_fn='prelu'),  
            conv_block(16, 8, 3, act_fn='prelu')
        )  

    def forward(self, feats_SW, feats_LW):  
        HW_size = feats_SW.size(3)  
        unflod_win = nn.Unfold(kernel_size=(self.size, self.size), stride=self.size)  
        flod_win = nn.Fold(output_size=(HW_size, HW_size), kernel_size=(self.size, self.size), stride=self.size)  

        SWIR_scale_f = self.SWIREncoder(feats_SW)  
        LWIR_scale_f = self.LWIREncoder(feats_LW)  

        SWIR_unfolded = unflod_win(SWIR_scale_f).permute(2, 0, 1)  
        LWIR_unfolded = unflod_win(LWIR_scale_f).permute(2, 0, 1)  

        SWIR_att = self.mlp_SA_SWIR(apply_attention(SWIR_unfolded, self.SA_SWIR))  
        LWIR_att = self.mlp_SA_LWIR(apply_attention(LWIR_unfolded, self.SA_LWIR))  

        SWIR_cross = self.mlp_CA_SWIR(apply_attention(SWIR_att, self.CA_SWIR, is_cross=True,  
                                                      query=SWIR_att, key_value=LWIR_att))  
        LWIR_cross = self.mlp_CA_LWIR(apply_attention(LWIR_att, self.CA_LWIR, is_cross=True,  
                                                      query=LWIR_att, key_value=SWIR_att))  

        SWIR_reconstructed = flod_win(SWIR_cross.permute(1, 2, 0))  
        LWIR_reconstructed = flod_win(LWIR_cross.permute(1, 2, 0))  

        f_cat = torch.cat([SWIR_reconstructed, LWIR_reconstructed], dim=1)  
        enhance_f = self.Decoder_mul(f_cat)
        
        return enhance_f

class MulLayerFusion(nn.Module):  
    def __init__(self, size=8, channel=8, act_fn='prelu'):  
        super(MulLayerFusion, self).__init__()  
        self.size = size  

        self.downsample_layer_SW = self._make_downsample_layer(channel, act_fn)  
        self.downsample_layer_LW = self._make_downsample_layer(channel, act_fn)  
        self.upsample_layer = self._make_upsample_layer(channel, act_fn)  
        self.Decoder = self._make_decoder_layer(act_fn)  

        self.CMAB_H = CrossFusion(size=size, channel=channel//2)    
        self.CMAB_L = CrossFusion(size=size, channel=channel//2)   
        self.CMAB_F = CrossFusion(size=size, channel=channel//2)    

    def _make_downsample_layer(self, channel, act_fn):  
        return nn.Sequential(  
            conv_block(channel, channel, 3, stride=2, act_fn=act_fn),  
            conv_block(channel, channel, 3, act_fn=act_fn),  
            conv_block(channel, channel * 2, 3, act_fn=act_fn),  
            conv_block(channel * 2, channel, 3, act_fn=act_fn),  
            conv_block(channel, channel, 3, act_fn=act_fn),  
        )  

    def _make_upsample_layer(self, channel, act_fn):   
        return nn.Sequential(  
            nn.ReflectionPad2d((1, 1, 1, 1)),  
            nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2),  
            nn.BatchNorm2d(channel),  
            nn.PReLU(),  
            conv_block(channel, channel, 3, act_fn=act_fn),  
            conv_block(channel, channel * 2, 3, act_fn=act_fn),  
            conv_block(channel * 2, channel, 3, act_fn=act_fn),  
            conv_block(channel, channel, 3, act_fn=act_fn), 
        )  

    def _make_decoder_layer(self, act_fn):   
        return nn.Sequential(  
            conv_block(8, 8, 3, act_fn=act_fn),  
            conv_block(8, 16, 3, act_fn=act_fn),  
            conv_block(16, 16, 3, act_fn=act_fn),  
            conv_block(16, 32, 3, act_fn=act_fn),  
            conv_block(32, 32, 3, act_fn=act_fn),  
            conv_block(32, 16, 3, act_fn=act_fn),
            conv_block(16, 16, 3, act_fn=act_fn),  
            conv_block(16, 4, 3, act_fn=act_fn),  
            nn.ReflectionPad2d((1, 1, 1, 1)),  
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=1)  
        )  

    def forward(self, feats_SW, feats_LW):    
        fuse_h = self.CMAB_H(feats_SW, feats_LW)   
        down_fS = self.downsample_layer_SW(feats_SW)  
        down_fL = self.downsample_layer_LW(feats_LW)  
        fuse_l = self.CMAB_L(down_fS, down_fL)  
        fuse_l_up = self.upsample_layer(fuse_l)  
        feats_mul = self.CMAB_F(fuse_h, fuse_l_up)  
        out = self.Decoder(feats_mul)  

        return out  


@ARCH_REGISTRY.register()  
class MMFusion(nn.Module):  
    def __init__(self):  
        super(MMFusion, self).__init__()  
        self.netfe_base = FeatureExtractorBase()  
        self.netfe_SW = FeatureExtractorMulNet()  
        self.netfe_LW = FeatureExtractorMulNet()  
        self.netMF_mulLayer = MulLayerFusion()  

    def forward(self, data):    
        feats_SW_base = self.netfe_base(data['SW'])  
        feats_LW_base = self.netfe_base(data['LW'])   
        feats_SW = self.netfe_SW(feats_SW_base)  
        feats_LW = self.netfe_LW(feats_LW_base)  
        pred_img = self.netMF_mulLayer(feats_SW, feats_LW)  

        return pred_img 