import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
DAFormer with Task-Specific Attention Modules
"""

# ==================== LIGHTWEIGHT BUILDING BLOCKS ====================

class ChannelAttention(nn.Module):
    """Lightweight Channel Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# ==================== TASK-SPECIFIC ATTENTION ====================

class MultiTaskAwareAttention(nn.Module):
    """
    TASK-SPECIFIC: Each task gets own degradation detector
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        # ==================== TASK-SPECIFIC DETECTORS ====================
        self.task_detectors = nn.ModuleDict({
            'shadow': self._make_detector(channels),
            'rain': self._make_detector(channels),
            'denoise': self._make_detector(channels),
            'lol': self._make_detector(channels)
        })
        # ==================== END TASK-SPECIFIC ====================
        
        # Depth-wise separable convs
        self.feature_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Conv2d(channels * 2, channels, 1, bias=False)
    
    def _make_detector(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, task_name='shadow'):
        # ==================== TASK-SPECIFIC: detector ====================
        if task_name in self.task_detectors:
            degradation_mask = self.task_detectors[task_name](x)
        else:
            degradation_mask = self.task_detectors['shadow'](x)
        # ==================== END TASK-SPECIFIC ====================
        
        # Separate degraded and clean features
        degraded_feat = self.feature_enhance(x) * degradation_mask
        clean_feat = self.feature_enhance(x) * (1 - degradation_mask)
        
        # Fuse both pathways
        out = self.fusion(torch.cat([degraded_feat, clean_feat], dim=1))
        return out + x, degradation_mask


# ==================== EFFICIENT TRANSFORMER BLOCK ====================

class EfficientAttention(nn.Module):
    """
    Efficient Transformer Attention
    """
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        # Lightweight QKV projection
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.project_out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w)
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w)
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)
        
        return out


class LightTransformerBlock(nn.Module):
    """Lightweight Transformer Block"""
    def __init__(self, dim, num_heads=2):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, dim)
        self.attn = EfficientAttention(dim, num_heads)
        
        self.norm2 = nn.GroupNorm(8, dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, int(dim * 2), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * 2), dim, 1)
        )
        
        # MultiTaskAwareAttention (task-specific)
        self.task_aware = MultiTaskAwareAttention(dim)

    def forward(self, x, task_name='shadow'):
        # ==================== TASK-SPECIFIC: Pass task_name ====================
        x_ta, degradation_mask = self.task_aware(x, task_name)
        # ==================== END TASK-SPECIFIC ====================
        
        # Self-attention
        x = x + self.attn(self.norm1(x_ta))
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x, degradation_mask


# ==================== LIGHTWEIGHT MULTI-SCALE FUSION ====================

class LightweightMultiScaleFusion(nn.Module):
    """
    Multi-Scale Fusion (Optimized)
    """
    def __init__(self, channels):
        super().__init__()
        
        self.scale1 = nn.Conv2d(channels, channels // 2, 3, padding=1, groups=channels//2, bias=False)
        self.scale2 = nn.Conv2d(channels, channels // 2, 3, padding=2, dilation=2, groups=channels//2, bias=False)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        self.channel_att = ChannelAttention(channels, reduction=16)

    def forward(self, x):
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        
        out = torch.cat([s1, s2], dim=1)
        out = self.fusion(out)
        out = self.channel_att(out)
        
        return out + x


# ==================== MAIN DAFORMER MODEL ====================

class DAF_IRNet(nn.Module):
    """
    DAFormer: Degrade-Aware Transformer for Image Restoration
    """
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        embed_dim=48,
        num_blocks=[2, 3, 3, 2],
        num_heads=[2, 4, 8, 8]
    ):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        
        # Encoder stages
        self.encoder_stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8] # [48, 96, 192, 384]
        
        for i in range(4): # i = 0, 1, 2, 3 (4 encoder stages)
            if i > 0:
                self.downsample_layers.append(
                    nn.Sequential(
                        nn.Conv2d(dims[i-1], dims[i], 2, stride=2, bias=False),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.downsample_layers.append(nn.Identity())
            
            blocks = nn.ModuleList([
                LightTransformerBlock(dims[i], num_heads[i])
                for _ in range(num_blocks[i])  # num_blocks = [2, 3, 3, 2]
            ])

            # Encoder 1: LT Block × 2
            # Encoder 2: LT Block × 3
            # Encoder 3: LT Block × 3
            # Encoder 4: LT Block × 2
            self.encoder_stages.append(blocks)
        
        # Bottleneck
        # MSF #1
        # Bottleneck: LT Block × 1
        # MSF #2
        self.bottleneck = nn.Sequential(
            LightweightMultiScaleFusion(dims[-1]),
            LightTransformerBlock(dims[-1], num_heads[-1]),
            LightweightMultiScaleFusion(dims[-1])
        )
        
        # Decoder
        self.decoder_stages = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(3, -1, -1):  # i = 3, 2, 1, 0 (4 decoder stages)
            if i < 3:
                self.skip_connections.append(
                    nn.Conv2d(dims[i] * 2, dims[i], 1, bias=False)
                )
            else:
                self.skip_connections.append(nn.Identity())
            
            if i > 0:
                self.upsample_layers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(dims[i], dims[i-1], 2, stride=2, bias=False),
                        nn.ReLU(inplace=True)
                    )
                )
            else:
                self.upsample_layers.append(nn.Identity())
            
            blocks = nn.ModuleList([
                LightTransformerBlock(dims[i], num_heads[i])
                for _ in range(num_blocks[i])  # num_blocks = [2, 3, 3, 2]
            ])
            
            # Decoder 4: LT Block × 2
            # Decoder 3: LT Block × 3
            # Decoder 2: LT Block × 3
            # Decoder 1: LT Block × 2
            self.decoder_stages.append(blocks)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(dims[0], dims[0]//2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims[0]//2, output_channels, 3, padding=1),
            nn.Tanh()
        )
        
        self.shadow_masks = []

    def forward(self, x, task_names=None):
        b, c, h, w = x.shape
        
        self.shadow_masks = []
        
        # ==================== TASK-SPECIFIC: Store task_names ====================
        if task_names is None:
            task_names = ['shadow'] * b
        # ==================== END TASK-SPECIFIC ====================
        
        # Initial features
        x = self.input_proj(x)
        
        # ENCODER
        encoder_features = []
        
        for i, (downsample, blocks) in enumerate(zip(self.downsample_layers, self.encoder_stages)):
            x = downsample(x)
            
            for block in blocks:
                # ==================== TASK-SPECIFIC: Process each sample with its task ====================
                if b == 1:
                    x, shadow_mask = block(x, task_names[0])
                    self.shadow_masks.append(shadow_mask)
                else:
                    x_list, mask_list = [], []
                    for sample_idx in range(b):
                        x_sample, mask_sample = block(x[sample_idx:sample_idx+1], task_names[sample_idx])
                        x_list.append(x_sample)
                        mask_list.append(mask_sample)
                    x = torch.cat(x_list, dim=0)
                    self.shadow_masks.append(torch.cat(mask_list, dim=0))
                # ==================== END TASK-SPECIFIC ====================
            
            encoder_features.append(x)
        
        # BOTTLENECK
        for layer in self.bottleneck:
            if isinstance(layer, LightTransformerBlock):
                # ==================== TASK-SPECIFIC ====================
                if b == 1:
                    x, _ = layer(x, task_names[0])
                else:
                    x_list = []
                    for sample_idx in range(b):
                        x_sample, _ = layer(x[sample_idx:sample_idx+1], task_names[sample_idx])
                        x_list.append(x_sample)
                    x = torch.cat(x_list, dim=0)
                # ==================== END TASK-SPECIFIC ====================
            else:
                x = layer(x)
        
        # DECODER
        for block in self.decoder_stages[0]:
            # ==================== TASK-SPECIFIC ====================
            if b == 1:
                x, shadow_mask = block(x, task_names[0])
                self.shadow_masks.append(shadow_mask)
            else:
                x_list, mask_list = [], []
                for sample_idx in range(b):
                    x_sample, mask_sample = block(x[sample_idx:sample_idx+1], task_names[sample_idx])
                    x_list.append(x_sample)
                    mask_list.append(mask_sample)
                x = torch.cat(x_list, dim=0)
                self.shadow_masks.append(torch.cat(mask_list, dim=0))
            # ==================== END TASK-SPECIFIC ====================
        x = self.upsample_layers[0](x)
        
        x = self.skip_connections[1](torch.cat([x, encoder_features[2]], dim=1))
        for block in self.decoder_stages[1]:
            # ==================== TASK-SPECIFIC ====================
            if b == 1:
                x, shadow_mask = block(x, task_names[0])
                self.shadow_masks.append(shadow_mask)
            else:
                x_list, mask_list = [], []
                for sample_idx in range(b):
                    x_sample, mask_sample = block(x[sample_idx:sample_idx+1], task_names[sample_idx])
                    x_list.append(x_sample)
                    mask_list.append(mask_sample)
                x = torch.cat(x_list, dim=0)
                self.shadow_masks.append(torch.cat(mask_list, dim=0))
            # ==================== END TASK-SPECIFIC ====================
        x = self.upsample_layers[1](x)
        
        x = self.skip_connections[2](torch.cat([x, encoder_features[1]], dim=1))
        for block in self.decoder_stages[2]:
            # ==================== TASK-SPECIFIC ====================
            if b == 1:
                x, shadow_mask = block(x, task_names[0])
                self.shadow_masks.append(shadow_mask)
            else:
                x_list, mask_list = [], []
                for sample_idx in range(b):
                    x_sample, mask_sample = block(x[sample_idx:sample_idx+1], task_names[sample_idx])
                    x_list.append(x_sample)
                    mask_list.append(mask_sample)
                x = torch.cat(x_list, dim=0)
                self.shadow_masks.append(torch.cat(mask_list, dim=0))
            # ==================== END TASK-SPECIFIC ====================
        x = self.upsample_layers[2](x)
        
        x = self.skip_connections[3](torch.cat([x, encoder_features[0]], dim=1))
        for block in self.decoder_stages[3]:
            # ==================== TASK-SPECIFIC ====================
            if b == 1:
                x, shadow_mask = block(x, task_names[0])
                self.shadow_masks.append(shadow_mask)
            else:
                x_list, mask_list = [], []
                for sample_idx in range(b):
                    x_sample, mask_sample = block(x[sample_idx:sample_idx+1], task_names[sample_idx])
                    x_list.append(x_sample)
                    mask_list.append(mask_sample)
                x = torch.cat(x_list, dim=0)
                self.shadow_masks.append(torch.cat(mask_list, dim=0))
            # ==================== END TASK-SPECIFIC ====================
        x = self.upsample_layers[3](x)
        
        # OUTPUT
        final_out = self.output_proj(x)
        
        return {
            'output': final_out,
            'refined1': final_out,
            'refined2': final_out,
            'shadow_masks': self.shadow_masks
        }
    # Degrade aware loss
    def compute_shadow_aware_loss(self):
        if len(self.shadow_masks) == 0:
            return torch.tensor(0.0)
        
        loss = 0.0
        for i in range(len(self.shadow_masks) - 1):
            mask1 = F.interpolate(self.shadow_masks[i], 
                                 size=self.shadow_masks[-1].shape[2:], 
                                 mode='bilinear', align_corners=False)
            mask2 = F.interpolate(self.shadow_masks[i+1], 
                                 size=self.shadow_masks[-1].shape[2:], 
                                 mode='bilinear', align_corners=False)
            loss += F.mse_loss(mask1, mask2)
        
        return loss / max(1, len(self.shadow_masks) - 1)
    
    # ==================== DEGRADATION SUPERVISION ====================
    def compute_degradation_supervision_loss(self, img_input, gt_input):
        if len(self.shadow_masks) == 0:
            return torch.tensor(0.0, device=img_input.device)
        
        # FIXED: Resize all masks to same size before stacking
        target_size = self.shadow_masks[-1].shape[-2:]  # Use last mask size
        
        resized_masks = []
        for mask in self.shadow_masks:
            if mask.shape[-2:] != target_size:
                mask_resized = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
            else:
                mask_resized = mask
            resized_masks.append(mask_resized)
        
        avg_attention = torch.stack(resized_masks).mean(dim=0)
        
        # ALWAYS use difference map (no gt_shadow)
        difference = torch.abs(img_input - gt_input).mean(dim=1, keepdim=True)
        target = F.interpolate(difference, size=target_size, mode='bilinear', align_corners=False)
        target = target / (target.max() + 1e-8)
        
        return F.mse_loss(avg_attention, target)
    # ==================== END DEGRADATION SUPERVISION ====================


if __name__ == "__main__":
    model = DAF_IRNet()
    x = torch.randn(2, 3, 256, 256)
    task_names = ['shadow', 'rain']
    out = model(x, task_names=task_names)
    print(f"Output shape: {out['output'].shape}")
    print(f"Shadow masks: {len(out['shadow_masks'])}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")