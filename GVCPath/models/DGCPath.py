# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
from layers.weight_init import trunc_normal_
from functools import partial
from layers.patch_embed import PatchEmbed, PositionEmbed
from layers.dcl import DCL, DCLW
import torch
import math
import random
import torch.nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block
from utils.pos_embed import get_2d_sincos_pos_embed

class VariationalEncoder(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        

        self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)
        

        nn.init.normal_(self.mu_head.weight, 0, 0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.normal_(self.logvar_head.weight, 0, 0.01)
        nn.init.zeros_(self.logvar_head.bias)
        
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)

        logvar = torch.clamp(logvar, -10, 5)
        return mu, logvar
class KLDivergenceContrastiveLearning(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, latent_dim=128, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.latent_dim = latent_dim
        

        self.encoder_view1 = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        self.encoder_view2 = VariationalEncoder(input_dim, hidden_dim, latent_dim)
        
    def forward(self, z1, z2):

        mu1, logvar1 = self.encoder_view1(z1)
        mu2, logvar2 = self.encoder_view2(z2)
        
        return {
            'mu1': mu1, 'logvar1': logvar1,
            'mu2': mu2, 'logvar2': logvar2
        }
def kl_divergence_gaussian(mu1, logvar1, mu2, logvar2):
    """

    KL(P||Q) = KL(N(μ1,σ1²) || N(μ2,σ2²))
    = 0.5 * [log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1]
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    

    eps = 1e-8
    var2 = var2 + eps
    
    kl = 0.5 * (
        logvar2 - logvar1 +  # log(σ2²/σ1²)
        (var1 + (mu1 - mu2).pow(2)) / var2 -  # (σ1² + (μ1-μ2)²)/σ2²
        1  # -1
    )
    
    return torch.sum(kl, dim=1)  # 

def symmetric_kl_divergence(mu1, logvar1, mu2, logvar2):
    """

    """
    kl_pq = kl_divergence_gaussian(mu1, logvar1, mu2, logvar2)
    kl_qp = kl_divergence_gaussian(mu2, logvar2, mu1, logvar1)
    return (kl_pq + kl_qp) / 2

def js_divergence(mu1, logvar1, mu2, logvar2):

    mu_m = (mu1 + mu2) / 2
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    var_m = (var1 + var2) / 2
    logvar_m = torch.log(var_m + 1e-8)
    
    kl_pm = kl_divergence_gaussian(mu1, logvar1, mu_m, logvar_m)
    kl_qm = kl_divergence_gaussian(mu2, logvar2, mu_m, logvar_m)
    
    return 0.5 * (kl_pm + kl_qm)

def kl_contrastive_loss(mu1, logvar1, mu2, logvar2, temperature=0.1, 
                       divergence_type='symmetric_kl', margin=0.0):
  
    batch_size = mu1.shape[0]
    device = mu1.device

    if divergence_type == 'kl':
        divergence_fn = kl_divergence_gaussian
    elif divergence_type == 'symmetric_kl':
        divergence_fn = symmetric_kl_divergence
    elif divergence_type == 'js':
        divergence_fn = js_divergence
    else:
        raise ValueError(f"Unknown divergence type: {divergence_type}")
    

    all_divergences = torch.zeros(batch_size, batch_size, device=device)
    
    for i in range(batch_size):
        for j in range(batch_size):
            div = divergence_fn(
                mu1[i:i+1], logvar1[i:i+1],
                mu2[j:j+1], logvar2[j:j+1]
            )
            all_divergences[i, j] = div.item()
    

    pos_divergences = torch.diag(all_divergences)
    

    mask = torch.eye(batch_size, device=device).bool()
    neg_divergences = all_divergences.masked_select(~mask).view(batch_size, -1)
    
    if margin > 0:
     
        pos_expanded = pos_divergences.unsqueeze(1).expand(-1, batch_size - 1)
        margin_losses = torch.clamp(margin + pos_expanded - neg_divergences, min=0)
        return margin_losses.mean()
    else:
     
        pos_sim = -pos_divergences / temperature
        neg_sim = -neg_divergences / temperature
        
        # 构建logits矩阵
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        return F.cross_entropy(logits, labels)
def kl_regularization_loss(mu1, logvar1, mu2, logvar2):
  
  
    kl1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp(), dim=1)
    kl2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp(), dim=1)
    
    return (kl1.mean() + kl2.mean()) / 2

def distribution_alignment_loss(mu1, logvar1, mu2, logvar2):
  
    mu_align = F.mse_loss(mu1, mu2)
    

    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    var_align = F.mse_loss(var1, var2)
    
    return mu_align + var_align

    
def analyze_learned_distributions(mu1, logvar1, mu2, logvar2):

    with torch.no_grad():
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
   
        stats = {
            'mu1_mean': mu1.mean().item(),
            'mu1_std': mu1.std().item(),
            'mu2_mean': mu2.mean().item(), 
            'mu2_std': mu2.std().item(),
            'var1_mean': var1.mean().item(),
            'var1_std': var1.std().item(),
            'var2_mean': var2.mean().item(),
            'var2_std': var2.std().item()
        }
        
    
        kl_div = kl_divergence_gaussian(mu1, logvar1, mu2, logvar2)
        sym_kl_div = symmetric_kl_divergence(mu1, logvar1, mu2, logvar2)
        js_div = js_divergence(mu1, logvar1, mu2, logvar2)
        
        stats.update({
            'kl_divergence_mean': kl_div.mean().item(),
            'kl_divergence_std': kl_div.std().item(),
            'symmetric_kl_mean': sym_kl_div.mean().item(),
            'js_divergence_mean': js_div.mean().item()
        })
        
        return stats

class DiffusionViewGenerator(nn.Module):
    def __init__(self, embed_dim, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.denoise_fn = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward_diffusion(self, z_0, t):
        """
        Add noise to the original embedding z_0 at step t
        """
        noise = torch.randn_like(z_0)
        alpha_t = torch.cos(t / self.num_steps * math.pi / 2)
        z_t = alpha_t * z_0 + (1 - alpha_t) * noise
        return z_t, noise

    def reverse_diffusion(self, z_T, steps=20):
        """
        Reverse diffusion: start from noise z_T, denoise step by step
        """
        z = z_T
        for _ in range(steps):
            eps = self.denoise_fn(z)
            z = z - eps * 0.1  # simple Euler update
        return z

    def training_loss(self, z_0):
        """
        Compute diffusion loss by predicting the noise
        """
        B = z_0.size(0)
        t = torch.randint(1, self.num_steps, (B, 1, 1), device=z_0.device).float()
        z_t, noise = self.forward_diffusion(z_0, t)
        noise_pred = self.denoise_fn(z_t)
        loss = F.mse_loss(noise_pred, noise)
        return loss


class MultiTaskLossWrapper(nn.Module):
    def __init__(self):
        super(MultiTaskLossWrapper, self).__init__()
       
        self.log_vars = nn.Parameter(torch.zeros(4))  

    def forward(self, loss1, loss2, loss3, loss4):
        loss = (
            0.5 * torch.exp(-self.log_vars[0]) * loss1 + self.log_vars[0] +
            0.5 * torch.exp(-self.log_vars[1]) * loss2 + self.log_vars[1] +
            0.5 * torch.exp(-self.log_vars[2]) * loss3 + self.log_vars[2] +
            0.5 * torch.exp(-self.log_vars[3]) * loss4 + self.log_vars[3]
        )
        return loss


class DGCPath(nn.Module):
    # """ Path Autoencoder with VisionTransformer backbone
    # """
    def __init__(self, 
                 node2vec,
                 input_dim=128,
                 hidden_dim=512,
                 latent_dim=128,
                 output_dim=128,
                 in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16, Temp=0.05,
                 mlp_ratio=4., embed_layer=PatchEmbed, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # self.temp =Temp
        self.diffusion = DiffusionViewGenerator(embed_dim=embed_dim)
        # spaial embedding
        self.patch_embed = embed_layer(node2vec=node2vec, embed_dim=128, dropout=0.)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------



        self.temperature = Temp
        self.kl_reg_weight = 0.01
        self.alignment_weight = 0.1
        self.contrastive_weight = 1.0
        self.divergence_type = 'js'
        self.margin = 0.0
        ############
        self.ve = VariationalEncoder()


        self.multi_loss_optimize = MultiTaskLossWrapper()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_1d_sincos_pos_embed(self, embed_dim, num_patches):
        # num_patches: sequence length
        position = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(num_patches, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0) 

    def path_encoder(self, x):
     
        x= self.patch_embed(x)
        B, L, D = x.shape

        pos_embed = self.get_1d_sincos_pos_embed(D, L).to(x.device)
        x = x + pos_embed

        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, L+1, D]

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    def forward(self, path_batch):

        z = self.path_encoder(path_batch)    

        z_wo_cls = z[:, 1:, :]            # [B, L, D]
        loss_diffusion = self.diffusion.training_loss(z_wo_cls)

        # view generation
        z_T = torch.randn_like(z_wo_cls)
        z_view1 = self.diffusion.reverse_diffusion(z_T)
        z_view2 = self.diffusion.reverse_diffusion(torch.randn_like(z_wo_cls))

        z_view1 = F.normalize(z_view1, dim=-1)
        z_view2 = F.normalize(z_view2, dim=-1)

        # Compute global contrastive loss (using mean pooling)
        z1 = z_view1.mean(dim=1)
        z2 = z_view2.mean(dim=1)
        ###Gaussian Distribution
        mu1, logvar1 = self.ve(z1)
        mu2, logvar2 = self.ve(z2)

        contrastive_loss = kl_contrastive_loss(
            mu1, logvar1, mu2, logvar2,
            temperature=self.temperature,
            divergence_type=self.divergence_type,
            margin=self.margin
        )
        

        kl_reg_loss = kl_regularization_loss(mu1, logvar1, mu2, logvar2)

        alignment_loss = distribution_alignment_loss(mu1, logvar1, mu2, logvar2)
  
        total_loss = self.multi_loss_optimize(contrastive_loss, kl_reg_loss,alignment_loss,loss_diffusion)
    
    

        return total_loss
    
    def embed(self, intputs):
        intputs = torch.squeeze(intputs,dim=1)
        latent1= self.path_encoder(intputs)
        x =torch.squeeze(torch.mean(latent1, dim=1),dim=1)
        return x