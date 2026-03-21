"""
Task Feature Discriminability Analysis for DAF-IRNet  (v5)
==========================================================
Uses Simplified Silhouette Score (centroid-based) which
measures inter-cluster vs intra-cluster separation and is
not sensitive to cluster size imbalance.

Usage:
    python tsne_feature_analysis_v5.py --load 50
"""

import os, sys, random, platform
import torch, torch.nn.functional as F
import argparse, numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torch.utils.data as data_utils

from utils.data_loader import (
    make_shadow_datapath_list, make_rain_datapath_list,
    make_lol_datapath_list,   make_denoise_datapath_list,
    TaskDataset, ImageTransform,
)
from models.DAF_IRNet import DAF_IRNet

torch.manual_seed(44); np.random.seed(44); random.seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ════════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════════

def fix_state_dict(sd):
    new = OrderedDict()
    for k, v in sd.items():
        new[k[7:] if k.startswith('module.') else k] = v
    return new

def gap(t):
    return F.adaptive_avg_pool2d(t, 1).squeeze(-1).squeeze(-1)

def shift_positive(emb):
    emb = emb.copy()
    emb -= emb.min()
    return emb

def simplified_silhouette(feats, labels):
    """
    Simplified Silhouette Score (cluster-centroid based).
    For each sample x:
      a(x) = distance to own cluster centroid
      b(x) = distance to nearest OTHER cluster centroid
      s(x) = (b - a) / max(a, b)
    Mean over all samples = overall score.
    Not affected by cluster size imbalance.
    Higher = clusters are tighter and further apart.
    """
    classes = np.unique(labels)
    if len(classes) < 2: return 0.0
    centroids = {c: feats[labels == c].mean(axis=0) for c in classes}
    scores = []
    for i, x in enumerate(feats):
        own = labels[i]
        a = np.linalg.norm(x - centroids[own])
        b = min(np.linalg.norm(x - centroids[c])
                for c in classes if c != own)
        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)
    return float(np.mean(scores))

def per_task_simplified_silhouette(feats, labels, task_ids):
    """Per-task simplified silhouette (one-vs-rest centroid distance)."""
    classes = np.unique(labels)
    centroids = {c: feats[labels == c].mean(axis=0) for c in classes}
    out = {}
    for ti, name in task_ids.items():
        mask = labels == ti
        if mask.sum() < 2: out[name] = 0.0; continue
        a_vals, b_vals = [], []
        for x in feats[mask]:
            a_vals.append(np.linalg.norm(x - centroids[ti]))
            b_vals.append(min(np.linalg.norm(x - centroids[c])
                              for c in classes if c != ti))
        a = np.mean(a_vals); b = np.mean(b_vals)
        denom = max(a, b)
        out[name] = float((b - a) / denom) if denom > 0 else 0.0
    return out


# ════════════════════════════════════════════════════════════════════════════
#  Feature extractor
# ════════════════════════════════════════════════════════════════════════════

class Extractor:
    def __init__(self, model):
        self.f_proj, self.f_bt, self._h = [], [], []
        self._h.append(model.input_proj.register_forward_hook(
            lambda m, i, o: self.f_proj.append(
                gap(o).detach().cpu().float())))
        self._h.append(model.bottleneck[2].register_forward_hook(
            lambda m, i, o: self.f_bt.append(
                gap(o).detach().cpu().float())))

    def clear(self):
        self.f_proj.clear(); self.f_bt.clear()

    def remove(self):
        for h in self._h: h.remove()

    def arrays(self):
        return (torch.cat(self.f_proj, 0).numpy(),
                torch.cat(self.f_bt,   0).numpy())


# ════════════════════════════════════════════════════════════════════════════
#  Style
# ════════════════════════════════════════════════════════════════════════════

COLORS = {'shadow': '#E74C3C', 'rain': '#3498DB',
          'denoise': '#2ECC71', 'lol': '#F39C12'}
LABELS = {'shadow': 'Shadow', 'rain': 'Derain',
          'denoise': 'Denoise', 'lol': 'LLIE'}
ORDER  = ['shadow', 'rain', 'denoise', 'lol']


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load',            default='50')
    ap.add_argument('--checkpoint_dir',  default='./checkpoints')
    ap.add_argument('--tsne_perplexity', type=float, default=30.0)
    ap.add_argument('--image_size',      type=int,   default=286)
    ap.add_argument('--crop_size',       type=int,   default=256)
    ap.add_argument('--out_dir',         default='./tsne_analysis_results')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = 0 if platform.system() == 'Windows' else 4
    print(f"\nDevice: {device} | Workers: {nw}")

    # ── load model ───────────────────────────────────────────────────────────
    model = DAF_IRNet().to(device)
    cp = os.path.join(args.checkpoint_dir, f'DAF_IRNet_{args.load}.pth')
    if not os.path.exists(cp):
        print(f"ERROR: not found: {cp}"); sys.exit(1)
    ck = torch.load(cp, map_location=device)
    model.load_state_dict(fix_state_dict(ck.get('model_state_dict', ck)))
    model.eval()
    print(f"Loaded: {cp}\n")

    # ── loaders ──────────────────────────────────────────────────────────────
    tf = ImageTransform(size=args.image_size, crop_size=args.crop_size,
                        mean=(0.5,), std=(0.5,))
    loaders = {}
    specs = [
        ('shadow',  make_shadow_datapath_list,  {}),
        ('rain',    make_rain_datapath_list,     {}),
        ('lol',     make_lol_datapath_list,      {}),
        ('denoise', make_denoise_datapath_list,  {'noise_level': 25}),
    ]
    for task, fn, kw in specs:
        try:
            lst = fn(phase='test', **kw)
            ds  = TaskDataset(lst, tf, 'test_no_crop', task)
            loaders[task] = data_utils.DataLoader(
                ds, batch_size=1, shuffle=False, num_workers=nw)
            print(f"  {task.upper():8s}: {len(ds)} test images")
        except Exception as e:
            print(f"  {task.upper():8s}: not found ({e})")

    avail   = {t: len(loaders[t]) for t in ORDER if t in loaders}
    total_n = sum(avail.values())
    print(f"\nTotal: {total_n} images\n")

    # ── extract ───────────────────────────────────────────────────────────────
    ext = Extractor(model)
    raw_feats, proj_feats, bt_feats, label_list = [], [], [], []
    task_to_int = {t: i for i, t in enumerate(ORDER)}

    with torch.no_grad():
        for task in ORDER:
            if task not in loaders: continue
            count, n_collect = 0, avail[task]
            for batch in tqdm(loaders[task], desc=f"  {LABELS[task]:8s}",
                              total=n_collect, leave=False):
                if count >= n_collect: break
                img, _, _, _ = batch
                img = img.to(device)
                raw_vec = gap(img).detach().cpu().float().numpy()
                ext.clear()
                with torch.cuda.amp.autocast():
                    _ = model(img, task_names=[task])
                f_proj, f_bt = ext.arrays()
                raw_feats.append(raw_vec)
                proj_feats.append(f_proj)
                bt_feats.append(f_bt)
                label_list.append(task_to_int[task])
                count += 1
            print(f"  {LABELS[task]:8s}: {count} samples")

    ext.remove()

    F_raw  = np.vstack(raw_feats)
    F_proj = np.vstack(proj_feats)
    F_bt   = np.vstack(bt_feats)
    L      = np.array(label_list)
    task_id_map = {task_to_int[t]: LABELS[t]
                   for t in ORDER if t in loaders}

    # ── metrics ───────────────────────────────────────────────────────────────
    print(f"\nComputing Silhouette scores ({total_n} images)...")

    sil_raw  = simplified_silhouette(F_raw,  L)
    sil_proj = simplified_silhouette(F_proj, L)
    sil_bt   = simplified_silhouette(F_bt,   L)

    print(f"\n  {'Stage':<35} {'Silhouette':>12}")
    print("  " + "-"*50)
    print(f"  {'Raw image (pixel domain)':<35} {sil_raw:>12.4f}")
    print(f"  {'Input proj. (after 1st conv)':<35} {sil_proj:>12.4f}")
    print(f"  {'Bottleneck (after MTAA)':<35} {sil_bt:>12.4f}")

    pt_raw  = per_task_simplified_silhouette(F_raw,  L, task_id_map)
    pt_proj = per_task_simplified_silhouette(F_proj, L, task_id_map)
    pt_bt   = per_task_simplified_silhouette(F_bt,   L, task_id_map)

    print(f"\n  Per-task Silhouette:")
    print(f"  {'Task':<10} {'Raw':>8} {'Proj':>10} {'BT':>10} {'Delta':>10}")
    print("  " + "-"*52)
    for t in ORDER:
        if t not in loaders: continue
        nm = LABELS[t]
        print(f"  {nm:<10} {pt_raw[nm]:>8.4f} {pt_proj[nm]:>10.4f} "
              f"{pt_bt[nm]:>10.4f} {pt_bt[nm]-pt_raw[nm]:>+10.4f}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print(f"\nRunning t-SNE...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    perp = min(args.tsne_perplexity, len(L) // 4 - 1)

    def run_tsne(X):
        Xs  = StandardScaler().fit_transform(X)
        emb = TSNE(n_components=2, perplexity=perp, random_state=44,
                   max_iter=1000, learning_rate='auto',
                   init='pca').fit_transform(Xs)
        return shift_positive(emb)

    emb_raw  = run_tsne(F_raw)
    emb_proj = run_tsne(F_proj)
    emb_bt   = run_tsne(F_bt)
    print("  Done.")

    # ── figure ────────────────────────────────────────────────────────────────
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')

    # FIX 1: "Simplified" removed from panel titles
    panels = [
        (emb_raw,
         f'(a) Raw Input (Pixel Domain)\n'
         f'Silhouette = {sil_raw:.4f}'),
        (emb_proj,
         f'(b) Input Projection\n'
         f'Silhouette = {sil_proj:.4f}'),
        (emb_bt,
         f'(c) Bottleneck (After MTAA)\n'
         f'Silhouette = {sil_bt:.4f}'),
    ]

    for ax, (emb, title) in zip(axes, panels):
        for t in ORDER:
            if t not in loaders: continue
            mask = L == task_to_int[t]
            ax.scatter(emb[mask, 0], emb[mask, 1],
                       c=COLORS[t], label=LABELS[t],
                       alpha=0.80, s=80, edgecolors='none')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('t-SNE Component 1', fontsize=11)
        ax.set_ylabel('t-SNE Component 2', fontsize=11)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#F8F9FA')

    handles = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=COLORS[t], markersize=15,
                      label=LABELS[t])
               for t in ORDER if t in loaders]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               fontsize=12, bbox_to_anchor=(0.5, -0.06),
               frameon=True, framealpha=0.9)
    plt.tight_layout()

    fig_path = os.path.join(args.out_dir,
                            'tsne_feature_discriminability_v5.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"\n  Figure: {fig_path}")

    # ── text output ───────────────────────────────────────────────────────────
    txt_path = os.path.join(args.out_dir,
                            'feature_discriminability_results_v5.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("DAF-IRNet  Task Feature Discriminability Analysis\n")
        f.write(f"Checkpoint: DAF_IRNet_{args.load}.pth | "
                f"Total images: {total_n}\n\n")
        f.write(f"{'Stage':<35} {'Silhouette':>12}\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Raw image (pixel domain)':<35} {sil_raw:>12.4f}\n")
        f.write(f"{'Input proj. (after 1st conv)':<35} {sil_proj:>12.4f}\n")
        f.write(f"{'Bottleneck (after MTAA)':<35} {sil_bt:>12.4f}\n\n")
        f.write(f"{'Task':<10} {'Raw':>8} {'Proj':>8} {'BT':>8} {'Delta':>10}\n")
        f.write("-"*48 + "\n")
        for t in ORDER:
            if t not in loaders: continue
            nm = LABELS[t]
            f.write(f"{nm:<10} {pt_raw[nm]:>8.4f} {pt_proj[nm]:>8.4f} "
                    f"{pt_bt[nm]:>8.4f} {pt_bt[nm]-pt_raw[nm]:>+10.4f}\n")
    print(f"  Results: {txt_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    tex_path = os.path.join(args.out_dir, 'table_discriminability_v5.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[!t]\n\\centering\n")
        f.write("\\caption{Task feature discriminability at three network\n")
        f.write("stages measured by Silhouette score\n")
        f.write("(centroid-based; higher indicates stronger inter-task\n")
        f.write("separation and tighter intra-task clustering).}\n")
        f.write("\\label{table_tsne}\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\setlength{\\tabcolsep}{6pt}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\n")
        f.write("Stage & Silhouette$\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Raw Image (Pixel Domain) & {sil_raw:.4f} \\\\\n")
        f.write(f"Input Projection & {sil_proj:.4f} \\\\\n")
        f.write(f"Bottleneck (After MTAA) & {sil_bt:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  LaTeX: {tex_path}")

    # FIX 2: Summary block — no undefined ch_proj or ch_bt variables
    print(f"\n{'='*55}")
    print(f"  SUMMARY ({total_n} images)")
    print(f"  {'Stage':<30} {'Silhouette':>12}")
    print(f"  {'-'*44}")
    print(f"  {'Raw':<30} {sil_raw:>12.4f}")
    print(f"  {'Input Proj.':<30} {sil_proj:>12.4f}")
    print(f"  {'Bottleneck (MTAA)':<30} {sil_bt:>12.4f}")
    print(f"  Delta (raw->bt): {sil_bt-sil_raw:+.4f}")
    print(f"{'='*55}\nDone.\n")


if __name__ == '__main__':
    main()