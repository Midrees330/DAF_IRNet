import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import os
import time
import math
import platform
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid, save_image
from collections import OrderedDict
from utils.data_loader import create_multitask_dataloaders
from models.DAF_IRNet import DAF_IRNet  # TASK-SPECIFIC VERSION

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class SSIMLoss(nn.Module):
    """SSIM Loss for structural similarity"""
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2):
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        window = self.window
        channel = img1.size(1)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return 1 - ssim_map.mean()
        else:
            return 1 - ssim_map.mean(1).mean(1).mean(1)


class MultiScalePerceptualLoss(nn.Module):
    """Multi-scale perceptual loss with VGG"""
    def __init__(self, device='cuda'):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        
        self.slice1 = nn.Sequential(*[vgg[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[vgg[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[vgg[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[vgg[x] for x in range(12, 21)])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # Proper normalization to [0, 1] then to ImageNet stats
        pred = (pred * 0.5) + 0.5
        target = (target * 0.5) + 0.5
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std  
        
        # Clamp to prevent extreme values in FP16
        pred = torch.clamp(pred, -10, 10)
        target = torch.clamp(target, -10, 10)
        
        with torch.no_grad():
            target_f1 = self.slice1(target)
            target_f2 = self.slice2(target_f1)
            target_f3 = self.slice3(target_f2)
            target_f4 = self.slice4(target_f3)
        
        pred_f1 = self.slice1(pred)
        pred_f2 = self.slice2(pred_f1)
        pred_f3 = self.slice3(pred_f2)
        pred_f4 = self.slice4(pred_f3)
        
        # Safe loss computation with clamping
        loss = (torch.clamp(F.l1_loss(pred_f1, target_f1), 0, 100) + 
                torch.clamp(F.l1_loss(pred_f2, target_f2), 0, 100) + 
                torch.clamp(F.l1_loss(pred_f3, target_f3), 0, 100) + 
                torch.clamp(F.l1_loss(pred_f4, target_f4), 0, 100)) / 4.0
        
        return loss


class EdgeAwareLoss(nn.Module):
    """Edge-aware loss to preserve boundaries"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor(
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]], dtype=torch.float32)

        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1))

    def get_edges(self, x):
        sobel_x = self.sobel_x.to(x.device)
        sobel_y = self.sobel_y.to(x.device)
        edge_x = F.conv2d(x, sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(x, sobel_y, padding=1, groups=3)
        
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-4)  
        return edges

    def forward(self, pred, target):
        pred_edges = self.get_edges(pred)
        target_edges = self.get_edges(target)
        # Safe loss with clamping
        loss = torch.clamp(F.l1_loss(pred_edges, target_edges), 0, 100)
        return loss


class ColorConstancyLoss(nn.Module):
    """Color constancy loss"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, input_img):
        pred_diff = pred - input_img
        target_diff = target - input_img
        
        pred_mean = torch.mean(pred, dim=[2, 3], keepdim=True)
        target_mean = torch.mean(target, dim=[2, 3], keepdim=True)
        
        diff_loss = F.l1_loss(pred_diff, target_diff)
        mean_loss = F.l1_loss(pred_mean, target_mean)
        
        return diff_loss + mean_loss


class FrequencyLoss(nn.Module):
    """Frequency domain loss for texture preservation"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Input clamping for stability
        pred = torch.clamp(pred, -1, 1)
        target = torch.clamp(target, -1, 1)
        
        # Force FP32 for stability
        try:
            with torch.cuda.amp.autocast(enabled=False):
                pred_fp32 = pred.float()
                target_fp32 = target.float()
                
                pred_fft = torch.fft.fft2(pred_fp32)
                target_fft = torch.fft.fft2(target_fp32)
                
                # Magnitude spectrum
                pred_mag = torch.abs(pred_fft)
                target_mag = torch.abs(target_fft)
                
                # Clamp magnitudes to prevent extreme values
                pred_mag = torch.clamp(pred_mag, 0, 1000)
                target_mag = torch.clamp(target_mag, 0, 1000)
                
                loss = torch.clamp(F.l1_loss(pred_mag, target_mag), 0, 100)
                return loss
        except:
            # Fallback
            return torch.tensor(0.0, device=pred.device)


class EnhancedLoss(nn.Module):
    """ loss function for all tasks"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.perceptual_loss = MultiScalePerceptualLoss(device=device).to(device)
        self.edge_loss = EdgeAwareLoss()
        self.color_loss = ColorConstancyLoss()
        self.freq_loss = FrequencyLoss()

    def safe_loss(self, loss_fn, name='Loss'):
        """Compute loss with NaN detection"""
        try:
            loss = loss_fn()
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: {name} is NaN/Inf, returning 0")
                return torch.tensor(0.0, device=loss.device)
            return torch.clamp(loss, 0, 1000)  # Prevent extreme values
        except Exception as e:
            print(f"WARNING: {name} computation failed: {e}")
            return torch.tensor(0.0, device='cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, outputs, target, input_img, epoch=0):
        pred = outputs['output']
        
        # Input validation
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("WARNING: NaN/Inf detected in model output!")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # CRITICAL: Compute ALL losses in FP32 for stability
        with torch.cuda.amp.autocast(enabled=False):
            pred_fp32 = pred.float()
            target_fp32 = target.float()
            input_fp32 = input_img.float()
            
            # Core losses with NaN checking
            l1 = self.safe_loss(lambda: self.l1_loss(pred_fp32, target_fp32), 'L1')
            ssim = self.safe_loss(lambda: self.ssim_loss(pred_fp32, target_fp32), 'SSIM')
            perceptual = self.safe_loss(lambda: self.perceptual_loss(pred_fp32, target_fp32), 'Perceptual')
            edge = self.safe_loss(lambda: self.edge_loss(pred_fp32, target_fp32), 'Edge')
            color = self.safe_loss(lambda: self.color_loss(pred_fp32, target_fp32, input_fp32), 'Color')
            freq = self.safe_loss(lambda: self.freq_loss(pred_fp32, target_fp32), 'Frequency')
        
        # Degrade mask consistency
        shadow_consistency = torch.tensor(0.0, device=target.device)
        if 'shadow_masks' in outputs and len(outputs['shadow_masks']) > 1:
            try:
                for i in range(len(outputs['shadow_masks']) - 1):
                    mask1 = F.interpolate(outputs['shadow_masks'][i], 
                                         size=outputs['shadow_masks'][-1].shape[2:],
                                         mode='bilinear', align_corners=False)
                    mask2 = F.interpolate(outputs['shadow_masks'][i+1],
                                         size=outputs['shadow_masks'][-1].shape[2:],
                                         mode='bilinear', align_corners=False)
                    shadow_consistency += F.mse_loss(mask1, mask2)
                shadow_consistency /= (len(outputs['shadow_masks']) - 1)
                shadow_consistency = torch.clamp(shadow_consistency, 0, 10)
            except:
                shadow_consistency = torch.tensor(0.0, device=target.device)
        
        # Weighted combination with REDUCED weights for multi-task stability
        total_loss = (
            torch.clamp(1.0 * l1, 0, 100) +
            torch.clamp(0.5 * ssim, 0, 50) +
            torch.clamp(0.4 * perceptual, 0, 40) +
            torch.clamp(0.3 * edge, 0, 30) +
            torch.clamp(0.2 * color, 0, 20) +      
            torch.clamp(0.1 * freq, 0, 10) +        
            torch.clamp(0.1 * shadow_consistency, 0, 10)
        )
        
        # Clamp total loss to prevent extreme values
        total_loss = torch.clamp(total_loss, 0, 300) 
        
        # Final NaN/Inf check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("CRITICAL: Total loss is NaN/Inf! Using fallback L1 loss.")
            total_loss = l1 if not (torch.isnan(l1) or torch.isinf(l1)) else torch.tensor(1.0, device=target.device)
        
        return {
            'total': total_loss,
            'l1': l1,
            'ssim': ssim,
            'perceptual': perceptual,
            'edge': edge,
            'color': color,
            'freq': freq,
            'shadow_consistency': shadow_consistency # Degrade consistency
        }


def get_parser():
    parser = argparse.ArgumentParser(description='DAFormer Multi-Task Training')
    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default=None, help='Resume checkpoint')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.95, help='Train-val split ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286, help='Image size')
    parser.add_argument('-cs', '--crop_size', type=int, default=256, help='Crop size')
    parser.add_argument('-lr', '--lr', type=float, default=1e-4, help='Learning rate')
    #parser.add_argument('--num_workers', type=int, default=0, help='Number of workers (0 for Windows)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    return parser


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    return new_state_dict


def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x


def check_dir():
    for d in ['./logs', './checkpoints', './result', './result/val']:
        os.makedirs(d, exist_ok=True)
    
    # Create degradation visualization folders for all tasks
    for task in ['shadow', 'rain', 'denoise', 'lol']:
        os.makedirs(f'./result/{task}_degradation', exist_ok=True)


def compute_psnr(pred, target):
    pred_u = (pred + 1.0) * 0.5
    target_u = (target + 1.0) * 0.5
    mse = F.mse_loss(pred_u, target_u)
    if mse.item() < 1e-10:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def compute_ssim_metric(pred, target):
    pred_u = (pred + 1.0) * 0.5
    target_u = (target + 1.0) * 0.5
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu1 = F.avg_pool2d(pred_u, 3, 1, 1)
    mu2 = F.avg_pool2d(target_u, 3, 1, 1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(pred_u * pred_u, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(target_u * target_u, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(pred_u * target_u, 3, 1, 1) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def visualize_degradation_detection(model, dataset, device, task_name, epoch):
    """
    Visualize degradation detection for any task
    Works for: shadow, rain, denoise, lol
    """
    model.eval()
    num_vis = min(3, len(dataset))
    items = [dataset[i] for i in range(num_vis)]
    imgs = torch.stack([it[0] for it in items]).to(device)
    gts = torch.stack([it[2] for it in items]).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # TASK-SPECIFIC: Pass task_names for correct MoE routing
            task_names_list = [task_name] * num_vis
            outputs = model(imgs, task_names=task_names_list)
            pred = outputs['output']
            
            shadow_masks = outputs.get('shadow_masks', [])
            if len(shadow_masks) > 0:
                target_size = pred.shape[2:]
                aggregated_mask = torch.zeros(pred.shape[0], 1, *target_size, device=device)
                for mask in shadow_masks:
                    resized_mask = F.interpolate(mask, size=target_size, mode='bilinear', align_corners=False)
                    aggregated_mask += resized_mask
                aggregated_mask /= len(shadow_masks)
            else:
                aggregated_mask = torch.zeros_like(pred[:, :1, :, :])
    
    rows = []
    for i in range(num_vis):
        row = torch.cat([
            unnormalize(imgs[i:i+1].cpu()),
            aggregated_mask[i:i+1].repeat(1, 3, 1, 1).cpu(),
            unnormalize(gts[i:i+1].cpu()),
            unnormalize(pred[i:i+1].cpu()),
        ], dim=0)
        rows.append(row)
    
    all_imgs = torch.cat(rows, dim=0)
    grid = make_grid(all_imgs, nrow=4, padding=10, pad_value=1.0)
    
    # Save to task-specific folder
    output_dir = f'./result/{task_name}_degradation'
    os.makedirs(output_dir, exist_ok=True)
    save_image(grid, f'{output_dir}/val_{task_name}_epoch{epoch}.jpg')
    
    model.train()


def evaluate(model, dataset, device, filename, epoch, task_name='all'):
    model.eval()
    num_eval = min(6, len(dataset))
    items = [dataset[i] for i in range(num_eval)]
    imgs = torch.stack([it[0] for it in items]).to(device)
    gts = torch.stack([it[2] for it in items]).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # TASK-SPECIFIC: Pass task_names for correct MoE routing
            task_names_list = [task_name] * num_eval if task_name != 'all' else ['shadow'] * num_eval
            outputs = model(imgs, task_names=task_names_list)
            pred = outputs['output']
        
        grid = make_grid(
            torch.cat([unnormalize(imgs.cpu()), unnormalize(gts.cpu()), unnormalize(pred.cpu())], dim=0),
            nrow=num_eval
        )
        save_image(grid, f'{filename}_{task_name}_epoch{epoch}.jpg')
    model.train()


def plot_loss_curves(loss_history, save_name='DAFormer'):
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    
    ax = axes[0, 0]
    if len(loss_history['total']) > 0:
        epochs = range(1, len(loss_history['total']) + 1)
        ax.plot(epochs, loss_history['total'], linewidth=2, color='#e74c3c')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Total Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    if len(loss_history['l1']) > 0:
        epochs = range(1, len(loss_history['l1']) + 1)
        ax.plot(epochs, loss_history['l1'], linewidth=2, color='#3498db', label='L1')
        ax.plot(epochs, loss_history['ssim'], linewidth=2, color='#2ecc71', label='SSIM')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('L1 + SSIM Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    ax = axes[0, 2]
    if len(loss_history['perceptual']) > 0:
        epochs = range(1, len(loss_history['perceptual']) + 1)
        ax.plot(epochs, loss_history['perceptual'], linewidth=2, color='#9b59b6', label='Perceptual')
        ax.plot(epochs, loss_history['edge'], linewidth=2, color='#e67e22', label='Edge')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Perceptual + Edge Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    ax = axes[0, 3]
    if 'degradation' in loss_history and len(loss_history['degradation']) > 0:
        epochs = range(1, len(loss_history['degradation']) + 1)
        ax.plot(epochs, loss_history['degradation'], linewidth=2, color='#f39c12', label='Degradation')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Degradation Supervision Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    ax = axes[1, 0]
    if len(loss_history['lr']) > 0:
        epochs = range(1, len(loss_history['lr']) + 1)
        ax.plot(epochs, loss_history['lr'], linewidth=2, color='#1abc9c')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
    
    ax = axes[1, 1]
    if len(loss_history['psnr']) > 0:
        psnr_epochs = [5*i for i in range(1, len(loss_history['psnr'])+1)]
        ax.plot(psnr_epochs, loss_history['psnr'], linewidth=2, color='#e74c3c', marker='o', markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('PSNR (dB)', fontsize=11)
        ax.set_title('Validation PSNR', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if len(loss_history['psnr']) > 0:
            best_psnr = max(loss_history['psnr'])
            best_epoch = psnr_epochs[loss_history['psnr'].index(best_psnr)]
            ax.axhline(y=best_psnr, color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Best: {best_psnr:.2f} dB @ Epoch {best_epoch}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax = axes[1, 2]
    if len(loss_history['ssim_metric']) > 0:
        ssim_epochs = [5*i for i in range(1, len(loss_history['ssim_metric'])+1)]
        ax.plot(ssim_epochs, loss_history['ssim_metric'], linewidth=2, color='#3498db', marker='s', markersize=6)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('SSIM', fontsize=11)
        ax.set_title('Validation SSIM', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        if len(loss_history['ssim_metric']) > 0:
            best_ssim = max(loss_history['ssim_metric'])
            best_epoch = ssim_epochs[loss_history['ssim_metric'].index(best_ssim)]
            ax.axhline(y=best_ssim, color='r', linestyle='--', alpha=0.5)
            ax.text(0.02, 0.98, f'Best: {best_ssim:.4f} @ Epoch {best_epoch}',
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax = axes[1, 3]
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./logs/{save_name}_loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_model(model, dataloader, val_datasets, num_epochs, parser, save_name='DAF_IRNet_'):
    check_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine actual num_workers (Windows override)
    #actual_workers = parser.num_workers if platform.system() != 'Windows' else 0

    
    print(f"Device: {device}")
    print(f"Batch Size: {parser.batch_size}")
    print(f"Initial LR: {parser.lr}")
    #print(f"Workers: {actual_workers} {'(Windows override)' if platform.system() == 'Windows' and parser.num_workers != 0 else ''}")
    print(f"Workers: {parser.num_workers}")
    print(f"Total training samples: {len(dataloader.dataset)}")
    
    model = model.to(device)
    
    optimizer = AdamW(
        model.parameters(),
        lr=parser.lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        fused=False
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    criterion = EnhancedLoss(device=device)
    scaler = torch.cuda.amp.GradScaler()
    
    loss_history = {
        'total': [], 'l1': [], 'ssim': [], 'perceptual': [],
        'edge': [], 'color': [], 'lr': [], 'psnr': [], 'ssim_metric': [],
        'degradation': []  # MULTI-TASK FIX
    }
    
    best_psnr = 0.0
    nan_batch_count = 0  # Track total NaN batches across all epochs
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = {
            'total': 0.0, 'l1': 0.0, 'ssim': 0.0,
            'perceptual': 0.0, 'edge': 0.0, 'color': 0.0,
            'degradation': 0.0  # MULTI-TASK FIX
        }
        
        task_counts = {'shadow': 0, 'denoise': 0, 'lol': 0, 'rain': 0}
        
        t_start = time.time()
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        num_batches = 0
        epoch_nan_count = 0  # Track NaN batches in current epoch
        
        for batch_idx, (images, gt_shadow, gt, task_names) in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            # gt_shadow (MASK) not used in task-specific version
            
            # NaN HANDLING 1: Input validation
            if torch.isnan(images).any() or torch.isnan(gt).any():
                print(f"WARNING: Skipping batch {batch_idx}: Input contains NaN")
                epoch_nan_count += 1
                continue
            
            for task in task_names:
                task_counts[task] += 1
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                # ==================== TASK-SPECIFIC: PASS TASK_NAMES ====================
                outputs = model(images, task_names=task_names)
                # ==================== END TASK-SPECIFIC ====================
                
                # NaN HANDLING 2: Output validation
                if torch.isnan(outputs['output']).any():
                    print(f"WARNING: Skipping batch {batch_idx}: Model output contains NaN")
                    epoch_nan_count += 1
                    continue
                
                # ==================== TASK-SPECIFIC: DEGRADATION SUPERVISION (NO GT_SHADOW (MASK)) ====================
                degradation_loss = model.compute_degradation_supervision_loss(images, gt)
                # ==================== END TASK-SPECIFIC ====================
                
                loss_dict = criterion(outputs, gt, images, epoch=epoch)
                
                # ==================== MULTI-TASK FIX: ADD TO TOTAL LOSS ====================
                loss_dict['total'] = loss_dict['total'] + 0.1 * degradation_loss
                loss_dict['degradation'] = degradation_loss
                # ==================== END MULTI-TASK FIX ====================
                
                # NaN HANDLING 3: Loss validation
                if torch.isnan(loss_dict['total']) or torch.isinf(loss_dict['total']):
                    print(f"CRITICAL: NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}!")
                    print(f"   L1: {loss_dict['l1'].item():.4f}, Perceptual: {loss_dict['perceptual'].item():.4f}")
                    print(f"   Edge: {loss_dict['edge'].item():.4f}, Color: {loss_dict['color'].item():.4f}")
                    epoch_nan_count += 1
                    continue
            
            scaler.scale(loss_dict['total']).backward()
            
            # Removed overly aggressive batch skipping logic
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Only skip if gradients are truly invalid (NaN/Inf), not just large
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(f"WARNING: NaN/Inf gradient norm, skipping batch {batch_idx}")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                epoch_nan_count += 1
                continue
            
            scaler.step(optimizer)
            scaler.update()
            
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key].item()
            
            num_batches += 1
            
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({
                    'Loss': f"{loss_dict['total'].item():.4f}",
                    'PSNR_est': f"{-10*math.log10(loss_dict['l1'].item() + 1e-8):.1f}",
                    'Tasks': '-'.join(set(task_names))
                })
        
        scheduler.step()
        t_end = time.time()
        epoch_time = t_end - t_start
        
        # Report NaN batches if any were skipped
        if epoch_nan_count > 0:
            print(f"INFO: Skipped {epoch_nan_count} batches due to NaN/Inf in epoch {epoch+1}")
            nan_batch_count += epoch_nan_count
        
        if num_batches == 0:
            print(f"ERROR: All batches in epoch {epoch+1} were invalid! Stopping training.")
            break
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            loss_history[key].append(epoch_losses[key])
        
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            val_psnr = 0.0
            val_ssim = 0.0
            num_val_total = 0
            
            with torch.no_grad():
                for task_name, val_dataset in val_datasets.items():
                    num_val = min(10, len(val_dataset))
                    num_val_total += num_val
                    
                    for i in range(num_val):
                        img, _, gt_img, _ = val_dataset[i]
                        img = img.unsqueeze(0).to(device, non_blocking=True)
                        gt_img = gt_img.unsqueeze(0).to(device, non_blocking=True)
                        
                        with torch.cuda.amp.autocast():
                            # ==================== TASK-SPECIFIC: PASS TASK_NAME ====================
                            outputs = model(img, task_names=[task_name])
                            # ==================== END TASK-SPECIFIC ====================
                        
                        val_psnr += compute_psnr(outputs['output'], gt_img)
                        val_ssim += compute_ssim_metric(outputs['output'], gt_img)
            
            if num_val_total > 0:
                val_psnr /= num_val_total
                val_ssim /= num_val_total
                loss_history['psnr'].append(val_psnr)
                loss_history['ssim_metric'].append(val_ssim)
            
            model.train()
            
            print(f"\n[Epoch {epoch+1}] Time: {epoch_time:.1f}s ({epoch_time/60:.2f}min)")
            print(f"  Task counts: Shadow={task_counts['shadow']}, Denoise={task_counts['denoise']}, LOL={task_counts['lol']}, Rain={task_counts['rain']}")
            print(f"  Total: {epoch_losses['total']:.4f} | L1: {epoch_losses['l1']:.4f} | SSIM: {epoch_losses['ssim']:.4f}")
            print(f"  Perceptual: {epoch_losses['perceptual']:.4f} | Edge: {epoch_losses['edge']:.4f}")
            if num_val_total > 0:
                print(f"  Val PSNR: {val_psnr:.2f} dB | Val SSIM: {val_ssim:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.7f}")
            
            # Save best model
            if num_val_total > 0 and val_psnr > best_psnr:
                best_psnr = val_psnr
                # Save with metadata
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_psnr': best_psnr,
                    'val_ssim': val_ssim
                }
                torch.save(checkpoint, f'{parser.checkpoint_dir}/{save_name}best.pth')
                for task in val_datasets.keys():
                    torch.save(checkpoint, f'{parser.checkpoint_dir}/DAF_IRNet_{task}_best.pth')
                print(f"  Best model saved! PSNR: {best_psnr:.2f} dB")
        else:
            print(f"\n[Epoch {epoch+1}] Time: {epoch_time:.1f}s | Total: {epoch_losses['total']:.4f}, "
                  f"L1: {epoch_losses['l1']:.4f}, SSIM: {epoch_losses['ssim']:.4f}")
        
        # Checkpoint saving: Epoch 1, then every 10 epochs
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            # Save with metadata
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
                'loss_history': loss_history
            }
            torch.save(checkpoint, f'{parser.checkpoint_dir}/{save_name}{epoch+1}.pth')
            for task in val_datasets.keys():
                torch.save(checkpoint, f'{parser.checkpoint_dir}/DAF_IRNet_{task}_{epoch+1}.pth')
            print(f"  Checkpoint saved: epoch {epoch+1}")
        
        # Grid saving: Epoch 1, then every 10 epochs
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            for task_name, val_dataset in val_datasets.items():
                if len(val_dataset) > 0:
                    evaluate(model, val_dataset, device, './result/val', epoch+1, task_name)
            
            # Degradation visualization for ALL tasks
            for task_name in ['shadow', 'rain', 'denoise', 'lol']:
                if task_name in val_datasets and len(val_datasets[task_name]) > 0:
                    visualize_degradation_detection(model, val_datasets[task_name], device, 
                                                   task_name, epoch+1)
            print(f"  Visual grids saved: epoch {epoch+1}")
        
        # Loss plots: Save every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == 1:
            plot_loss_curves(loss_history, save_name.rstrip('_'))
            print(f"  Loss curves saved: logs/{save_name.rstrip('_')}_loss_curves.png")
    
    print(f"\nTraining Complete! Best PSNR: {best_psnr:.2f} dB")
    print(f"Total NaN batches skipped: {nan_batch_count}")
    return model, best_psnr


def main(parser):
    print("\n" + "="*80)
    print("DAFormer Multi-Task Training")
    print("="*80 + "\n")
    
    os.makedirs(parser.checkpoint_dir, exist_ok=True)
    
    model = DAF_IRNet(
        input_channels=3,
        output_channels=3,
        embed_dim=48,
        num_blocks=[2, 3, 3, 2],
        num_heads=[2, 4, 8, 8]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)\n")
    
    if parser.load:
        checkpoint_path = f'{parser.checkpoint_dir}/DAF_IRNet_{parser.load}.pth'
        print(f'Loading checkpoint: {checkpoint_path}')
        try:
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # New format with metadata
                model.load_state_dict(fix_model_state_dict(checkpoint['model_state_dict']))
                if 'epoch' in checkpoint:
                    print(f"  Loaded from epoch: {checkpoint['epoch']}")
                if 'best_psnr' in checkpoint:
                    print(f"  Best PSNR: {checkpoint['best_psnr']:.2f} dB")
            else:
                # (state_dict only)
                model.load_state_dict(fix_model_state_dict(checkpoint))
            print(" Loaded successfully!\n")
        except Exception as e:
            print(f" Error loading: {e}\n")
    
    print("Loading datasets...")
    train_loader, val_loader, test_loaders, train_datasets, val_datasets = create_multitask_dataloaders(
        batch_size=parser.batch_size,
        num_workers=parser.num_workers if platform.system() != 'Windows' else 0,
        size=parser.image_size,
        crop_size=parser.crop_size,
        rate=parser.hold_out_ratio
    )
    
    print(f"Total training samples: {len(train_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}\n")
    model, best_psnr = train_model(
        model=model,
        dataloader=train_loader,
        val_datasets=val_datasets,
        num_epochs=parser.epoch,
        parser=parser,
        save_name='DAF_IRNet_'
    )
    
    # Save final models
    print("\nSaving final models...")
    final_checkpoint = {
        'epoch': parser.epoch,
        'model_state_dict': model.state_dict(),
        'best_psnr': best_psnr,
        'message': 'Final model after complete training'
    }
    torch.save(final_checkpoint, f'{parser.checkpoint_dir}/DAF_IRNet_multitask_final.pth')
    for task in train_datasets.keys():
        torch.save(final_checkpoint, f'{parser.checkpoint_dir}/DAF_IRNet_{task}_final.pth')
    print(" Final models saved!")


if __name__ == "__main__":
    if platform.system() == 'Windows':
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    parser = get_parser().parse_args()
    main(parser)