import os, sys, platform, random
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
    """Global average pool [B,C,H,W] -> [B,C]"""
    return F.adaptive_avg_pool2d(t, 1).squeeze(-1).squeeze(-1)

def shift_positive(emb):
    emb = emb.copy()
    emb -= emb.min()
    return emb

def compute_silhouette(feats, labels):
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) < 2: return 0.0
    if len(feats) > 4000:
        idx = np.random.choice(len(feats), 4000, replace=False)
        feats, labels = feats[idx], labels[idx]
    return float(silhouette_score(feats, labels, metric='euclidean'))


# ════════════════════════════════════════════════════════════════════════════
#  MTAA hook — captures f_deg and f_cln
# ════════════════════════════════════════════════════════════════════════════

class MTAAHook:
    """
    Monkey-patches MultiTaskAwareAttention.forward to store
    degraded_feat and clean_feat without changing output.
    """
    def __init__(self, mtaa_module):
        self.mtaa  = mtaa_module
        self.f_deg = None
        self.f_cln = None
        self._orig = mtaa_module.forward
        mtaa_module.forward = self._patched

    def _patched(self, x, task_name='shadow'):
        if task_name in self.mtaa.task_detectors:
            mask = self.mtaa.task_detectors[task_name](x)
        else:
            mask = self.mtaa.task_detectors['shadow'](x)
        xt = self.mtaa.feature_enhance(x)
        degraded_feat = xt * mask
        clean_feat    = xt * (1 - mask)
        # store GAP vectors
        self.f_deg = gap(degraded_feat).detach().cpu().float()
        self.f_cln = gap(clean_feat).detach().cpu().float()
        out = self.mtaa.fusion(
            torch.cat([degraded_feat, clean_feat], dim=1))
        return out + x, mask

    def restore(self):
        self.mtaa.forward = self._orig


# ════════════════════════════════════════════════════════════════════════════
#  Style
# ════════════════════════════════════════════════════════════════════════════

TASK_COLORS = {
    'shadow':  '#E74C3C',
    'rain':    '#3498DB',
    'denoise': '#2ECC71',
    'lol':     '#F39C12',
}
TASK_LABELS = {
    'shadow':  'Shadow',
    'rain':    'Derain',
    'denoise': 'Denoise',
    'lol':     'LLIE',
}
ORDER = ['shadow', 'rain', 'denoise', 'lol']

# Degraded vs Clean colours (consistent across all subplots)
DEG_COLOR = '#C0392B'   # dark red
CLN_COLOR = '#2980B9'   # dark blue


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load',            default='690')
    ap.add_argument('--checkpoint_dir',  default='./checkpoints')
    ap.add_argument('--tsne_perplexity', type=float, default=30.0)
    ap.add_argument('--image_size',      type=int,   default=286)
    ap.add_argument('--crop_size',       type=int,   default=256)
    ap.add_argument('--out_dir',         default='./mtaa_tsne_results')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nw = 0  # Windows safe
    print(f"\nDevice: {device}")

    # ── load model ───────────────────────────────────────────────────────────
    model = DAF_IRNet().to(device)
    cp = os.path.join(args.checkpoint_dir, f'DAF_IRNet_{args.load}.pth')
    if not os.path.exists(cp):
        print(f"ERROR: {cp} not found"); sys.exit(1)
    ck = torch.load(cp, map_location=device)
    model.load_state_dict(fix_state_dict(ck.get('model_state_dict', ck)))
    model.eval()
    print(f"Loaded: {cp}\n")

    # ── install MTAA hook on first encoder LT block ───────────────────────────
    first_mtaa = model.encoder_stages[0][0].task_aware
    hook = MTAAHook(first_mtaa)
    print("MTAA hook installed on encoder_stages[0][0].task_aware\n")

    # ── build test loaders ────────────────────────────────────────────────────
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

    # ── extract f_deg and f_cln for each task ─────────────────────────────────
    # For each image we get TWO vectors: f_deg and f_cln
    # Label: 0 = degraded pathway, 1 = clean pathway
    # We also track which task each sample came from

    all_feats  = []   # [N*2, C]
    all_labels = []   # [N*2]  0=degraded 1=clean
    all_tasks  = []   # [N*2]  task index
    task_to_int = {t: i for i, t in enumerate(ORDER)}

    per_task_feats  = {t: {'deg': [], 'cln': []} for t in ORDER}

    print("\nExtracting f_deg and f_cln features...")
    with torch.no_grad():
        for task in ORDER:
            if task not in loaders: continue
            n_use = len(loaders[task])
            count = 0
            for batch in tqdm(loaders[task],
                              desc=f"  {TASK_LABELS[task]:8s}",
                              total=n_use, leave=False):
                if count >= n_use: break
                img, _, _, _ = batch
                img = img.to(device)
                with torch.cuda.amp.autocast():
                    _ = model(img, task_names=[task])
                # hook has stored f_deg and f_cln
                deg_vec = hook.f_deg.numpy()   # [1, C]
                cln_vec = hook.f_cln.numpy()   # [1, C]

                all_feats.append(deg_vec)
                all_feats.append(cln_vec)
                all_labels.append(0)   # degraded
                all_labels.append(1)   # clean
                all_tasks.append(task_to_int[task])
                all_tasks.append(task_to_int[task])

                per_task_feats[task]['deg'].append(deg_vec)
                per_task_feats[task]['cln'].append(cln_vec)
                count += 1
            print(f"  {TASK_LABELS[task]:8s}: {count} images -> "
                  f"{count*2} feature vectors")

    hook.restore()

    F_all = np.vstack(all_feats)    # [N*2, C]
    L_all = np.array(all_labels)    # 0=deg, 1=cln
    T_all = np.array(all_tasks)     # task index

    print(f"\nTotal feature vectors: {len(F_all)}")
    print(f"  Degraded pathway: {(L_all==0).sum()}")
    print(f"  Clean pathway:    {(L_all==1).sum()}")

    # ── Silhouette score (degraded vs clean) ──────────────────────────────────
    print(f"\nComputing Silhouette score (degraded vs clean)...")
    sil_overall = compute_silhouette(F_all, L_all)
    print(f"  Overall Silhouette (deg vs cln): {sil_overall:.4f}")

    # per-task silhouette
    print(f"\n  Per-task Silhouette:")
    print(f"  {'Task':<12} {'Silhouette':>12}")
    print("  " + "-"*26)
    per_task_sil = {}
    for task in ORDER:
        if task not in loaders: continue
        d = np.vstack(per_task_feats[task]['deg'])
        c = np.vstack(per_task_feats[task]['cln'])
        F_t = np.vstack([d, c])
        L_t = np.array([0]*len(d) + [1]*len(c))
        s   = compute_silhouette(F_t, L_t)
        per_task_sil[task] = s
        print(f"  {TASK_LABELS[task]:<12} {s:>12.4f}")

    # ── t-SNE ─────────────────────────────────────────────────────────────────
    print(f"\nRunning t-SNE (overall, {len(F_all)} vectors)...")
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    def run_tsne(X):
        # perplexity must be strictly less than n_samples
        # recompute per call so small datasets (e.g. LOL 30 vectors) work
        n = len(X)
        perp = min(args.tsne_perplexity, n // 3 - 1)
        perp = max(5.0, float(perp))   # minimum sensible perplexity
        Xs  = StandardScaler().fit_transform(X)
        emb = TSNE(n_components=2, perplexity=perp, random_state=44,
                   max_iter=1000, learning_rate='auto',
                   init='pca').fit_transform(Xs)
        return shift_positive(emb)

    print("  Feature extraction done.")

    # ── Figure: Per-task — 4 subplots ──────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    print(f"  Running per-task t-SNE...")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.patch.set_facecolor('white')

    for col, task in enumerate(ORDER):
        ax = axes[col]
        if task not in loaders:
            ax.axis('off'); continue

        d = np.vstack(per_task_feats[task]['deg'])
        c = np.vstack(per_task_feats[task]['cln'])
        F_t = np.vstack([d, c])
        L_t = np.array([0]*len(d) + [1]*len(c))
        emb_t = run_tsne(F_t)

        ax.scatter(emb_t[L_t==0, 0], emb_t[L_t==0, 1],
                   c=DEG_COLOR, alpha=0.65, s=25, edgecolors='none',
                   label=f'$f_{{\\mathrm{{deg}}}}$')
        ax.scatter(emb_t[L_t==1, 0], emb_t[L_t==1, 1],
                   c=CLN_COLOR, alpha=0.65, s=25, edgecolors='none',
                   label=f'$f_{{\\mathrm{{cln}}}}$')

        s = per_task_sil.get(task, 0.0)
        ax.set_title(f'{TASK_LABELS[task]}\nSilhouette = {s:.4f}',
                     fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel('t-SNE Component 1', fontsize=10)
        ax.set_ylabel('t-SNE Component 2', fontsize=10)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#F8F9FA')
        ax.legend(fontsize=10, framealpha=0.9)

    plt.tight_layout()
    fig2_path = os.path.join(args.out_dir,
                             'tsne_disentanglement_per_task.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Figure saved: {fig2_path}")

    # ── text results ──────────────────────────────────────────────────────────
    txt_path = os.path.join(args.out_dir,
                            'disentanglement_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("DAF-IRNet MTAA Feature Disentanglement Analysis\n")
        f.write(f"Checkpoint: DAF_IRNet_{args.load}.pth\n")
        f.write(f"Feature vectors: {len(F_all)} "
                f"(all test images x 2 pathways)\n\n")
        f.write(f"Overall Silhouette (deg vs cln): {sil_overall:.4f}\n\n")
        f.write(f"{'Task':<12} {'Silhouette':>12}\n")
        f.write("-"*26 + "\n")
        for task in ORDER:
            if task in per_task_sil:
                f.write(f"{TASK_LABELS[task]:<12} "
                        f"{per_task_sil[task]:>12.4f}\n")
    print(f"  Results: {txt_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────────
    tex_path = os.path.join(args.out_dir,
                            'table_disentanglement.tex')
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[!t]\n\\centering\n")
        f.write("\\caption{MTAA feature disentanglement quantified by\n")
        f.write("Silhouette score between the degraded pathway\n")
        f.write("$f_{\\mathrm{deg}}$ and clean pathway\n")
        f.write("$f_{\\mathrm{cln}}$ across each task. Higher score\n")
        f.write("indicates stronger separation between the two pathways,\n")
        f.write("confirming that MTAA successfully decouples\n")
        f.write("degradation-specific from content-preserving features.}\n")
        f.write("\\label{table_disent}\n")
        f.write("\\renewcommand{\\arraystretch}{1.2}\n")
        f.write("\\setlength{\\tabcolsep}{8pt}\n")
        f.write("\\scriptsize\n")
        f.write("\\begin{tabular}{lc}\n\\toprule\n")
        f.write("Task & Silhouette ($f_{\\mathrm{deg}}$ vs "
                "$f_{\\mathrm{cln}}$)$\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        for task in ORDER:
            if task in per_task_sil:
                f.write(f"{TASK_LABELS[task]} & "
                        f"{per_task_sil[task]:.4f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Overall & {sil_overall:.4f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  LaTeX: {tex_path}")

    print(f"\n{'='*55}")
    print(f"  DISENTANGLEMENT SUMMARY")
    print(f"  Overall Silhouette: {sil_overall:.4f}")
    print(f"  (>0 = degraded and clean features are separated)")
    print(f"{'='*55}\nDone.\n")


if __name__ == '__main__':
    main()
