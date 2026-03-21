"""
Loss Contribution Analysis for DAF-IRNet
=========================================
Loads checkpoint DAF_IRNet_690.pth and evaluates all seven
individual loss components on each test dataset independently.
Reports mean ± std per loss per task, and produces:
  1. Console table
  2. loss_contribution_results.txt
  3. Ready-to-paste LaTeX table (table7)

Usage (from project root):
    python loss_contribution_analysis.py --load 690

No retraining required — uses the saved best checkpoint only.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# ── project imports ─────────────────────────────────────────────────────────
from utils.data_loader import create_multitask_dataloaders
from models.DAF_IRNet import DAF_IRNet

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ════════════════════════════════════════════════════════════════════════════
#  Loss modules (copied verbatim from train.py for self-contained script)
# ════════════════════════════════════════════════════════════════════════════

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, 1)

    def _gaussian(self, window_size, sigma):
        import math
        gauss = torch.Tensor([
            math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def _create_window(self, window_size, channel):
        _1D = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        channel = img1.size(1)
        if channel != self.channel or self.window.data.type() != img1.data.type():
            self.window = self._create_window(self.window_size, channel)
            self.channel = channel
        window = self.window.to(img1.device).type_as(img1)
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12   = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()


class MultiScalePerceptualLoss(nn.Module):
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
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        pred   = (pred   * 0.5 + 0.5 - self.mean) / self.std
        target = (target * 0.5 + 0.5 - self.mean) / self.std
        pred   = torch.clamp(pred,   -10, 10)
        target = torch.clamp(target, -10, 10)
        with torch.no_grad():
            tf1 = self.slice1(target); tf2 = self.slice2(tf1)
            tf3 = self.slice3(tf2);   tf4 = self.slice4(tf3)
        pf1 = self.slice1(pred);  pf2 = self.slice2(pf1)
        pf3 = self.slice3(pf2);   pf4 = self.slice4(pf3)
        return (torch.clamp(F.l1_loss(pf1, tf1), 0, 100) +
                torch.clamp(F.l1_loss(pf2, tf2), 0, 100) +
                torch.clamp(F.l1_loss(pf3, tf3), 0, 100) +
                torch.clamp(F.l1_loss(pf4, tf4), 0, 100)) / 4.0


class EdgeAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=torch.float32)
        self.register_buffer('sx', sx.view(1,1,3,3).repeat(3,1,1,1))
        self.register_buffer('sy', sy.view(1,1,3,3).repeat(3,1,1,1))

    def forward(self, pred, target):
        sx = self.sx.to(pred.device); sy = self.sy.to(pred.device)
        pe = torch.sqrt(F.conv2d(pred,  sx, padding=1, groups=3)**2 +
                        F.conv2d(pred,  sy, padding=1, groups=3)**2 + 1e-4)
        te = torch.sqrt(F.conv2d(target,sx, padding=1, groups=3)**2 +
                        F.conv2d(target,sy, padding=1, groups=3)**2 + 1e-4)
        return torch.clamp(F.l1_loss(pe, te), 0, 100)


class ColorConstancyLoss(nn.Module):
    def forward(self, pred, target, inp):
        return (F.l1_loss(pred - inp, target - inp) +
                F.l1_loss(pred.mean([2,3], keepdim=True),
                          target.mean([2,3], keepdim=True)))


class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        try:
            with torch.cuda.amp.autocast(enabled=False):
                pf = torch.fft.fft2(torch.clamp(pred,  -1, 1).float())
                tf = torch.fft.fft2(torch.clamp(target,-1, 1).float())
                return torch.clamp(
                    F.l1_loss(torch.clamp(torch.abs(pf), 0, 1000),
                              torch.clamp(torch.abs(tf), 0, 1000)), 0, 100)
        except:
            return torch.tensor(0.0, device=pred.device)


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def fix_state_dict(sd):
    new = OrderedDict()
    for k, v in sd.items():
        new[k[7:] if k.startswith('module.') else k] = v
    return new


def compute_psnr(pred, target):
    p = (pred   + 1.0) * 0.5
    t = (target + 1.0) * 0.5
    mse = F.mse_loss(p, t)
    if mse.item() < 1e-10:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()


def compute_ssim_val(pred, target):
    p = (pred   + 1.0) * 0.5
    t = (target + 1.0) * 0.5
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(p, 3, 1, 1);  mu2 = F.avg_pool2d(t, 3, 1, 1)
    s1  = F.avg_pool2d(p*p, 3,1,1) - mu1**2
    s2  = F.avg_pool2d(t*t, 3,1,1) - mu2**2
    s12 = F.avg_pool2d(p*t, 3,1,1) - mu1*mu2
    ssim = ((2*mu1*mu2+C1)*(2*s12+C2)) / ((mu1**2+mu2**2+C1)*(s1+s2+C2))
    return ssim.mean().item()


# ════════════════════════════════════════════════════════════════════════════
#  Per-image loss evaluation
# ════════════════════════════════════════════════════════════════════════════

LOSS_NAMES = ['L1', 'SSIM', 'Perceptual', 'Edge', 'Color', 'Frequency', 'Deg.Mask']
WEIGHTS    = [1.0,   0.5,    0.4,          0.3,    0.2,     0.1,         0.1]

def evaluate_losses(model, loader, task_name, losses, device):
    """
    Returns dict of lists: one list per loss name containing
    per-image unweighted loss values.
    Also returns per-image PSNR and SSIM.
    """
    results = {n: [] for n in LOSS_NAMES}
    psnr_list, ssim_list = [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {task_name}", leave=False):
            img, gt_mask, gt, _ = batch
            img = img.to(device); gt = gt.to(device)

            with torch.cuda.amp.autocast():
                out = model(img, task_names=[task_name] * img.size(0))
            pred = out['output']

            # cast to FP32 for loss computation
            p = pred.float(); t = gt.float(); inp = img.float()

            l1_v   = F.l1_loss(p, t).item()
            ssim_v = losses['ssim'](p, t).item()
            perc_v = losses['perc'](p, t).item()
            edge_v = losses['edge'](p, t).item()
            col_v  = losses['color'](p, t, inp).item()
            freq_v = losses['freq'](p, t).item()

            # Degradation mask consistency
            masks = out['shadow_masks']
            if len(masks) > 1:
                deg_v = 0.0
                for i in range(len(masks) - 1):
                    m1 = F.interpolate(masks[i],   size=masks[-1].shape[2:],
                                       mode='bilinear', align_corners=False)
                    m2 = F.interpolate(masks[i+1], size=masks[-1].shape[2:],
                                       mode='bilinear', align_corners=False)
                    deg_v += F.mse_loss(m1, m2).item()
                deg_v /= max(1, len(masks) - 1)
            else:
                deg_v = 0.0

            results['L1'].append(l1_v)
            results['SSIM'].append(ssim_v)
            results['Perceptual'].append(perc_v)
            results['Edge'].append(edge_v)
            results['Color'].append(col_v)
            results['Frequency'].append(freq_v)
            results['Deg.Mask'].append(deg_v)

            psnr_list.append(compute_psnr(pred, gt))
            ssim_list.append(compute_ssim_val(pred, gt))

    return results, psnr_list, ssim_list


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load',           type=str,   default='690')
    parser.add_argument('--checkpoint_dir', type=str,   default='./checkpoints')
    parser.add_argument('--num_workers',    type=int,   default=4)
    parser.add_argument('--image_size',     type=int,   default=286)
    parser.add_argument('--crop_size',      type=int,   default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # ── load model ─────────────────────────────────────────────────────────
    model = DAF_IRNet().to(device)
    ckpt_path = os.path.join(args.checkpoint_dir, f'DAF_IRNet_{args.load}.pth')
    if not os.path.exists(ckpt_path):
        # try alternate naming
        ckpt_path = os.path.join(args.checkpoint_dir, f'DAF_IRNet_shadow_{args.load}.pth')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: checkpoint not found. Tried:\n  {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(fix_state_dict(state))
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}\n")

    # ── instantiate loss modules ───────────────────────────────────────────
    losses = {
        'ssim':  SSIMLoss().to(device),
        'perc':  MultiScalePerceptualLoss(device=str(device)).to(device),
        'edge':  EdgeAwareLoss().to(device),
        'color': ColorConstancyLoss(),
        'freq':  FrequencyLoss(),
    }

    # ── create test loaders ────────────────────────────────────────────────
    print("Loading test datasets...")
    _, _, test_loaders, _, _ = create_multitask_dataloaders(
        batch_size=1,
        num_workers=args.num_workers,
        size=args.image_size,
        crop_size=args.crop_size,
        rate=0.95
    )

    # ── evaluate ───────────────────────────────────────────────────────────
    TASKS = ['shadow', 'rain', 'lol']
    DENOISE_LEVELS = [15, 25, 50]

    all_task_results  = {}   # task_label -> {loss_name: [values]}
    all_task_psnr     = {}
    all_task_ssim_val = {}

    print("\n" + "="*70)
    print("  Evaluating individual loss components per task...")
    print("="*70)

    for task in TASKS:
        if task not in test_loaders:
            print(f"  Skipping {task}: loader not found")
            continue
        print(f"\n  Task: {task.upper()}")
        res, psnr_l, ssim_l = evaluate_losses(
            model, test_loaders[task], task, losses, device)
        all_task_results[task]  = res
        all_task_psnr[task]     = psnr_l
        all_task_ssim_val[task] = ssim_l

    # Denoising — average across all three noise levels
    if 'denoise' in test_loaders:
        print("\n  Task: DENOISE (averaging σ=15,25,50)")
        combined = {n: [] for n in LOSS_NAMES}
        combined_psnr, combined_ssim = [], []
        for nl in DENOISE_LEVELS:
            if nl not in test_loaders['denoise']:
                continue
            res, psnr_l, ssim_l = evaluate_losses(
                model, test_loaders['denoise'][nl],
                'denoise', losses, device)
            for n in LOSS_NAMES:
                combined[n].extend(res[n])
            combined_psnr.extend(psnr_l)
            combined_ssim.extend(ssim_l)
        all_task_results['denoise']  = combined
        all_task_psnr['denoise']     = combined_psnr
        all_task_ssim_val['denoise'] = combined_ssim

    # ── compute statistics ─────────────────────────────────────────────────
    TASK_LABELS = {
        'shadow':  'Shadow',
        'rain':    'Derain',
        'denoise': 'Denoise',
        'lol':     'LLIE',
    }

    print("\n\n" + "="*90)
    print(f"  {'Task':<10}  {'PSNR':>6}  {'SSIM':>6}  " +
          "  ".join([f"{n:>11}" for n in LOSS_NAMES]))
    print("="*90)

    rows = {}
    for task_key in ['shadow', 'rain', 'denoise', 'lol']:
        if task_key not in all_task_results:
            continue
        label = TASK_LABELS[task_key]
        res   = all_task_results[task_key]
        psnr_avg = np.mean(all_task_psnr[task_key])
        ssim_avg = np.mean(all_task_ssim_val[task_key])

        means = {n: np.mean(res[n]) for n in LOSS_NAMES}
        stds  = {n: np.std(res[n])  for n in LOSS_NAMES}

        row_str = f"  {label:<10}  {psnr_avg:>6.2f}  {ssim_avg:>6.4f}  "
        row_str += "  ".join([f"{means[n]:>5.4f}±{stds[n]:.4f}" for n in LOSS_NAMES])
        print(row_str)
        rows[label] = {'psnr': psnr_avg, 'ssim': ssim_avg,
                       'means': means, 'stds': stds}

    print("="*90)

    # ── weighted contribution (%) ──────────────────────────────────────────
    print("\n  Weighted loss contribution (%) — shows which loss dominates per task:")
    print("-"*70)
    header = f"  {'Task':<10}  " + "  ".join([f"{n:>11}" for n in LOSS_NAMES])
    print(header)
    print("-"*70)

    for label, data in rows.items():
        weighted = {n: data['means'][n] * w
                    for n, w in zip(LOSS_NAMES, WEIGHTS)}
        total_w = sum(weighted.values()) + 1e-8
        pct = {n: weighted[n] / total_w * 100 for n in LOSS_NAMES}
        row_str = f"  {label:<10}  " + "  ".join([f"{pct[n]:>11.1f}" for n in LOSS_NAMES])
        print(row_str)
    print("-"*70)

    # ── save results ───────────────────────────────────────────────────────
    out_dir = './loss_analysis_results'
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, 'loss_contribution_results.txt')

    with open(out_file, 'w') as f:
        f.write("DAF-IRNet  Loss Contribution Analysis\n")
        f.write(f"Checkpoint: DAF_IRNet_{args.load}.pth\n\n")
        f.write(f"{'Task':<10}  {'PSNR':>6}  {'SSIM':>6}  " +
                "  ".join([f"{n:>13}" for n in LOSS_NAMES]) + "\n")
        f.write("-"*100 + "\n")
        for label, data in rows.items():
            line = f"{label:<10}  {data['psnr']:>6.2f}  {data['ssim']:>6.4f}  "
            line += "  ".join([
                f"{data['means'][n]:>6.4f}±{data['stds'][n]:.4f}"
                for n in LOSS_NAMES])
            f.write(line + "\n")
        f.write("\nWeighted contribution (%):\n")
        f.write(f"{'Task':<10}  " +
                "  ".join([f"{n:>11}" for n in LOSS_NAMES]) + "\n")
        f.write("-"*80 + "\n")
        for label, data in rows.items():
            weighted = {n: data['means'][n] * w
                        for n, w in zip(LOSS_NAMES, WEIGHTS)}
            total_w = sum(weighted.values()) + 1e-8
            pct = {n: weighted[n] / total_w * 100 for n in LOSS_NAMES}
            line = f"{label:<10}  " + "  ".join([f"{pct[n]:>11.1f}" for n in LOSS_NAMES])
            f.write(line + "\n")

    print(f"\n  Full results saved to: {out_file}")

    # ── LaTeX table ────────────────────────────────────────────────────────
    latex_file = os.path.join(out_dir, 'table_loss_contribution.tex')
    with open(latex_file, 'w') as f:
        f.write("% ============================================================\n")
        f.write("% Table: Per-Task Loss Contribution Analysis\n")
        f.write("% Mean unweighted loss value per component per task\n")
        f.write("% ============================================================\n\n")
        f.write("\\begin{table}[!t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Per-task loss contribution analysis on the trained\n")
        f.write("DAF-IRNet model (checkpoint~690). Values report the mean\n")
        f.write("unweighted loss magnitude of each component evaluated on\n")
        f.write("the respective test set. The weighted contribution column\n")
        f.write("($\\lambda_k \\cdot \\bar{\\mathcal{L}}_k / \\mathcal{L}_{\\mathrm{total}}$,\n")
        f.write("\\%) quantifies each term's relative influence on the total\n")
        f.write("objective per task.}\n")
        f.write("\\label{table_loss}\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\scriptsize\n")

        # Header
        f.write("\\begin{tabular}{lcc" + "c"*len(LOSS_NAMES) + "}\n")
        f.write("\\toprule\n")
        f.write("Task & PSNR$\\uparrow$ & SSIM$\\uparrow$ & ")
        f.write(" & ".join([f"$\\mathcal{{L}}_{{{{{n}}}}}$"
                             for n in LOSS_NAMES]))
        f.write(" \\\\\n")
        f.write("& (dB) & & " + " & ".join(
            [f"($\\lambda$={w})" for w in WEIGHTS]) + " \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for label, data in rows.items():
            line = f"{label} & {data['psnr']:.2f} & {data['ssim']:.4f}"
            for n in LOSS_NAMES:
                line += f" & {data['means'][n]:.4f}"
            line += " \\\\\n"
            f.write(line)

        f.write("\\midrule\n")
        f.write("\\multicolumn{3}{l}{\\textit{Weighted contribution (\\%)}} \\\\\n")

        for label, data in rows.items():
            weighted = {n: data['means'][n] * w
                        for n, w in zip(LOSS_NAMES, WEIGHTS)}
            total_w = sum(weighted.values()) + 1e-8
            pct = {n: weighted[n] / total_w * 100 for n in LOSS_NAMES}
            line = f"{label} & — & —"
            for n in LOSS_NAMES:
                line += f" & {pct[n]:.1f}"
            line += " \\\\\n"
            f.write(line)

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  LaTeX table saved to: {latex_file}")
    print("\nDone.\n")


if __name__ == '__main__':
    main()