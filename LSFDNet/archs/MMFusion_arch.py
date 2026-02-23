import torch
from torch import nn
import torch.nn.functional as F
import math
from basicsr.utils.registry import ARCH_REGISTRY


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, act_fn='relu', use_pad=True):  
    layers = []  
    if use_pad:  
        # layers.append(nn.ReflectionPad2d((kernel_size // 2, kernel_size // 2,  
        #                                   kernel_size // 2, kernel_size // 2)))  
        layers.append(nn.ReplicationPad2d(kernel_size // 2))
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
    def __init__(self, d_model, dropout=0.1, max_len=2048):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class PatchEmbed(nn.Module):
    """ 
    将 2D 图像切分成 Patch Embedding
    (B, C, H, W) -> (B, N, Embed_dim) 
    """
    def __init__(self, img_size=None, patch_size=8, in_chans=4, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)            # -> (B, Embed_dim, H/P, W/P)
        x = x.flatten(2)            # -> (B, Embed_dim, N)
        x = x.transpose(1, 2)       # -> (B, N, Embed_dim) 符合 Transformer 输入格式
        x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    """
    将 Patch Embedding 恢复成 2D 图像
    (B, N, Embed_dim) -> (B, C, H, W)
    """
    def __init__(self, patch_size=8, out_chans=4, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.ConvTranspose2d(embed_dim, out_chans, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, H, W):
        B, N, C = x.shape
        h_new = H // self.patch_size
        w_new = W // self.patch_size
        
        x = x.transpose(1, 2).reshape(B, C, h_new, w_new)
        x = self.proj(x) # -> (B, out_chans, H, W)
        return x
class DINOv3SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, pos_encoding_obj=None):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if pos_encoding_obj is not None:
            pe = pos_encoding_obj.pe[:, :N, :].reshape(1, N, self.num_heads, self.head_dim).transpose(1, 2)
            q = q + pe
            k = k + pe
        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop.p if self.training else 0.,
            scale=self.scale
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class DINOv3CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)      # Query 来源
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias) # Key/Value 来源 (合并以提速)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context, pos_encoding_obj=None):
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).reshape(B, N, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if pos_encoding_obj is not None:
            pe = pos_encoding_obj.pe[:, :N, :].reshape(1, N, self.num_heads, self.head_dim).transpose(1, 2)
            q = q + pe
            k = k + pe

        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop.p if self.training else 0.,
            scale=self.scale
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class CrossFusion(nn.Module):  
    def __init__(self, size=8, dropout=0.1, channel=4, num_heads=8):  
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
        self.pos_encoder = PositionalEncoding(d_model=self.dim, dropout=dropout, max_len=10000)
        self.patch_embed_swir = PatchEmbed(patch_size=size, in_chans=4, embed_dim=self.dim)
        self.patch_embed_lwir = PatchEmbed(patch_size=size, in_chans=4, embed_dim=self.dim)
        self.patch_unembed_swir = PatchUnEmbed(patch_size=size, out_chans=4, embed_dim=self.dim)
        self.patch_unembed_lwir = PatchUnEmbed(patch_size=size, out_chans=4, embed_dim=self.dim)

        self.mlp_SA_SWIR = mlp_block(self.dim)
        self.mlp_SA_LWIR = mlp_block(self.dim)
        self.mlp_CA_SWIR = mlp_block(self.dim)
        self.mlp_CA_LWIR = mlp_block(self.dim)
        
        self.norm_sa_swir = nn.LayerNorm(self.dim)
        self.norm_sa_lwir = nn.LayerNorm(self.dim)
        self.norm_ca_swir = nn.LayerNorm(self.dim)
        self.norm_ca_lwir = nn.LayerNorm(self.dim)
        assert self.dim % num_heads == 0, "Dim must be divisible by num_heads"
        
        self.SA_SWIR = DINOv3SelfAttention(self.dim, num_heads=num_heads, attn_drop=dropout)
        self.SA_LWIR = DINOv3SelfAttention(self.dim, num_heads=num_heads, attn_drop=dropout)
        
        self.CA_SWIR = DINOv3CrossAttention(self.dim, num_heads=num_heads, attn_drop=dropout)
        self.CA_LWIR = DINOv3CrossAttention(self.dim, num_heads=num_heads, attn_drop=dropout)
        self.Decoder_mul = nn.Sequential(  
            conv_block(8, 8, 3, act_fn='prelu'),  
            conv_block(8, 16, 3, act_fn='prelu'),  
            conv_block(16, 16, 3, act_fn='prelu'),  
            conv_block(16, 8, 3, act_fn='prelu')
        )  
        self.initialize_weights()  # 调用初始化方法

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, feats_SW, feats_LW):
        SWIR_scale_f = self.SWIREncoder(feats_SW)
        LWIR_scale_f = self.LWIREncoder(feats_LW)
        B, C, H, W = SWIR_scale_f.shape
        swir_tokens = self.patch_embed_swir(SWIR_scale_f)
        lwir_tokens = self.patch_embed_lwir(LWIR_scale_f)

        swir_norm = self.norm_sa_swir(swir_tokens)
        swir_att = self.SA_SWIR(swir_norm, pos_encoding_obj=self.pos_encoder)
        swir_tokens = swir_tokens + self.mlp_SA_SWIR(swir_att)
        lwir_norm = self.norm_sa_lwir(lwir_tokens)
        lwir_att = self.SA_LWIR(lwir_norm, pos_encoding_obj=self.pos_encoder)
        lwir_tokens = lwir_tokens + self.mlp_SA_LWIR(lwir_att)


        swir_ctx = swir_tokens
        lwir_ctx = lwir_tokens

        swir_norm = self.norm_ca_swir(swir_tokens)
        swir_cross = self.CA_SWIR(swir_norm, context=lwir_ctx, pos_encoding_obj=self.pos_encoder)
        swir_tokens = swir_tokens + self.mlp_CA_SWIR(swir_cross)
        lwir_norm = self.norm_ca_lwir(lwir_tokens)
        lwir_cross = self.CA_LWIR(lwir_norm, context=swir_ctx, pos_encoding_obj=self.pos_encoder)
        lwir_tokens = lwir_tokens + self.mlp_CA_LWIR(lwir_cross)
        SWIR_reconstructed = self.patch_unembed_swir(swir_tokens, H, W)
        LWIR_reconstructed = self.patch_unembed_lwir(lwir_tokens, H, W)

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
            # nn.ReflectionPad2d((1, 1, 1, 1)),  
            # nn.ConvTranspose2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=2),  
            # nn.BatchNorm2d(channel),  
            # nn.PReLU(),  
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d((1, 1, 1, 1)), 
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.PReLU(),
            conv_block(channel, channel, 3, act_fn=act_fn),  
            conv_block(channel, channel * 2, 3, act_fn=act_fn),  
            conv_block(channel * 2, channel, 3, act_fn=act_fn),  
            conv_block(channel, channel, 3, act_fn=act_fn), 
        )  

    def forward(self, feats_SW, feats_LW):    
        fuse_h = self.CMAB_H(feats_SW, feats_LW)   
        down_fS = self.downsample_layer_SW(feats_SW)  
        down_fL = self.downsample_layer_LW(feats_LW)  
        fuse_l = self.CMAB_L(down_fS, down_fL)  
        fuse_l_up = self.upsample_layer(fuse_l)  
        feats_mul = self.CMAB_F(fuse_h, fuse_l_up)  

        return feats_mul
    
class Decoder(nn.Module):  
    def __init__(self, layers_config):  
        super(Decoder, self).__init__()  
        layers = []  
        for config in layers_config:  
            layers.append(conv_block(*config))
        layers.append(nn.ReflectionPad2d((1, 1, 1, 1)))
        layers.append(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=1))  
        self.module = nn.Sequential(*layers)  

    def forward(self, x):  
        return self.module(x) 
    
@ARCH_REGISTRY.register()  
class MMFusion(nn.Module):  
    def __init__(self):  
        super(MMFusion, self).__init__()  
        self.fusion_decoder_config = [  
            (8, 8, 3, 1, "prelu"),  
            (8, 16, 3, 1, "prelu"),  
            (16, 16, 3, 1, "prelu"),  
            (16, 32, 3, 1, "prelu"),  
            (32, 32, 3, 1, "prelu"),  
            (32, 16, 3, 1, "prelu"),  
            (16, 16, 3, 1, "prelu"),  
            (16, 4, 3, 1, "prelu"),  
        ]
        self.netfe_base = FeatureExtractorBase()  
        self.netfe_SW = FeatureExtractorMulNet()  
        self.netfe_LW = FeatureExtractorMulNet()  
        self.netMF_mulLayer = MulLayerFusion()  
        self.netDe_fusion = Decoder(self.fusion_decoder_config)

    def forward(self, data):    
        feats_SW_base = self.netfe_base(data['SW'])  
        feats_LW_base = self.netfe_base(data['LW'])   
        feats_SW = self.netfe_SW(feats_SW_base)  
        feats_LW = self.netfe_LW(feats_LW_base)  
        feats_mul = self.netMF_mulLayer(feats_SW, feats_LW) 
        pred_img = self.netDe_fusion(feats_mul) 

        return pred_img 
