"""
Reload & re-plot script for the Spectral QEC [[5,1,3]] RyRz depolarizing experiment.
Loads the saved .npz files and reconstructs the editable figures.

Requires (from the original run):
  perfect_results_<tag>.npz
  perfect_fidelity_vs_p_<tag>.npz
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 0. USER-EDITABLE CONFIG  —  change plot text/style here
# ================================================================
FILE_TAG = "layerwise"          # "multistart" or "layerwise"

RESULTS_NPZ = f"data/perfect_results_{FILE_TAG}.npz"
P_NPZ       = f"data/perfect_fidelity_vs_p_{FILE_TAG}.npz"

P_NOISE = 0.05

# --- colors ---
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
LAYER_LS     = ['-', '--', '-', '--', '-']

# --- editable titles ---
TITLE_FIG_TRAIN   = "Training fidelity vs epoch (full curve, both phases, ±1σ band)"
TITLE_FIG_LOSS    = "Baseline loss vs epoch (full curve, ±1σ band)"
TITLE_FIG_TESTBAR = "Test fidelity vs layer count"
TITLE_FIG_LOGICAL = "Logical spectrum error vs layer count\n(lower = better logical info restoration)"

# Fig-2 title + annotation are auto-filled from data via {L} and {p} placeholders.
TITLE_FIG_P_TMPL = "Perfect [[5,1,3]] depolarizing — best model (L={L}, trained at p={p})"
ANNOT_FIG_P_TMPL = "trained @ p={p}"

# ================================================================
# 1. Load main results
# ================================================================
D = np.load(RESULTS_NPZ, allow_pickle=True)

LAYER_SWEEP   = list(np.array(D['meta_layers']).tolist())
USE_LAYERWISE = bool(int(D['meta_layerwise']))
INIT_EPOCHS   = int(D['meta_INIT_EPOCHS'])
MAIN_EPOCHS   = int(D['meta_MAIN_EPOCHS'])
best_nl_save  = int(D['best_n_layers'])
BASE_W        = np.array(D['meta_baseline_w']).tolist()   # [S, ZL, XL, YL]
W_STAB_BASE, W_ZL_BASE, W_XL_BASE, W_YL_BASE = BASE_W

# global test-set arrays (identical across layers)
f_std_test   = D[f"L{LAYER_SWEEP[0]}_f_std_test"]
f_noisy_test = D[f"L{LAYER_SWEEP[0]}_f_noisy_test"]
N_TEST_STATES = f_std_test.shape[0]
# train count: infer from probe meta if present, else fall back to label text
N_TRAIN_STATES = int(D['meta_probe_n']) if 'meta_probe_n' in D.files else 100

# Rebuild the per-layer results dict
sweep = {}
for nl in LAYER_SWEEP:
    sweep[nl] = {
        'n_layers':      nl,
        'params':        D[f'L{nl}_params'],
        'hist_epoch':    D[f'L{nl}_epoch'],
        'hist_phase':    D[f'L{nl}_phase'],
        'hist_loss':     D[f'L{nl}_loss_hist'],
        'hist_loss_std': D[f'L{nl}_loss_hist_std'],
        'hist_fid':      D[f'L{nl}_fid_hist'],
        'hist_fid_std':  D[f'L{nl}_fid_hist_std'],
        'f_qae_test':    D[f'L{nl}_f_qae_test'],
        'f_std_test':    f_std_test,
        'f_noisy_test':  f_noisy_test,
        'dzl_test':      D[f'L{nl}_dzl_test'],
        'dxl_test':      D[f'L{nl}_dxl_test'],
        'dyl_test':      D[f'L{nl}_dyl_test'],
    }
    sweep[nl]['n_params'] = int(sweep[nl]['params'].shape[0])

print(f"Loaded {RESULTS_NPZ}")
print(f"  p_noise={P_NOISE} (config), layers={LAYER_SWEEP}, best L={best_nl_save}, "
      f"train={N_TRAIN_STATES}, test={N_TEST_STATES}, layerwise={USE_LAYERWISE}")

# ================================================================
# 2. 4-panel comparison figure
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- (0,0) training fidelity (±1σ band) ---
ax = axes[0, 0]
for (nl, res), c, ls in zip(sweep.items(), LAYER_COLORS, LAYER_LS):
    ax.plot(res['hist_epoch'], res['hist_fid'], color=c, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(res['hist_epoch'], res['hist_fid']-res['hist_fid_std'],
                    res['hist_fid']+res['hist_fid_std'], color=c, alpha=0.12)
ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(INIT_EPOCHS+1, 0.625, "Phase 2 →", fontsize=8, color='gray')
ax.text(1, 0.625, "← Phase 1", fontsize=8, color='gray')
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)")
ax.set_ylabel(f"Avg fidelity (all {N_TRAIN_STATES} train states)")
ax.set_title(TITLE_FIG_TRAIN, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim([0.60, 1.02]); ax.grid(True, alpha=0.3)

# --- (0,1) baseline loss (±1σ band) ---
ax = axes[0, 1]
for (nl, res), c, ls in zip(sweep.items(), LAYER_COLORS, LAYER_LS):
    ax.plot(res['hist_epoch'], res['hist_loss'], color=c, ls=ls, lw=2, label=f"L={nl}")
    ax.fill_between(res['hist_epoch'], res['hist_loss']-res['hist_loss_std'],
                    res['hist_loss']+res['hist_loss_std'], color=c, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(INIT_EPOCHS+1, ax.get_ylim()[1]*0.9, "Phase 2 →", fontsize=8, color='gray')
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)")
ax.set_ylabel("Fixed-weight baseline spectral loss")
ax.set_title(f"{TITLE_FIG_LOSS}\nweights S={W_STAB_BASE}, ZL={W_ZL_BASE}, "
             f"XL={W_XL_BASE}, YL={W_YL_BASE}", fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

# --- (1,0) test fidelity bars ---
ax = axes[1, 0]; x = np.arange(len(LAYER_SWEEP)); bw = 0.22
fn_m = [sweep[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep[nl]['f_std_test'].mean()   for nl in LAYER_SWEEP]
fq_m = [sweep[nl]['f_qae_test'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep[nl]['f_qae_test'].std()    for nl in LAYER_SWEEP]
ax.bar(x-bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x,    fs_m, bw, label="Standard QEC", color='#3498db', alpha=0.8)
bars = ax.bar(x+bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
ax.errorbar(x+bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4, lw=1.5)
for b, v in zip(bars, fq_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test states)")
ax.set_title(TITLE_FIG_TESTBAR, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim([0.60, 1.02]); ax.grid(True, alpha=0.3, axis='y')

# --- (1,1) logical spectrum error bars ---
ax = axes[1, 1]; bw2 = 0.25
dz = [sweep[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dx = [sweep[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dy = [sweep[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
ax.bar(x-bw2, dz, bw2, label="ΔZL", color='#e74c3c', alpha=0.85)
ax.bar(x,     dx, bw2, label="ΔXL", color='#3498db', alpha=0.85)
ax.bar(x+bw2, dy, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
for xi, (a, b, c) in enumerate(zip(dz, dx, dy)):
    ax.text(xi-bw2, a+0.002, f"{a:.3f}", ha='center', va='bottom', fontsize=7)
    ax.text(xi,     b+0.002, f"{b:.3f}", ha='center', va='bottom', fontsize=7)
    ax.text(xi+bw2, c+0.002, f"{c:.3f}", ha='center', va='bottom', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean |target − recovered| ({N_TEST_STATES} test states)")
ax.set_title(TITLE_FIG_LOGICAL, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Perfect [[5,1,3]] depolarizing layer sweep  |  p={P_NOISE}  "
             f"|  layerwise warm-start={USE_LAYERWISE}",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"perfect_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : perfect_fidelity_{FILE_TAG}.png")
print(f"  Saved : RyRz_fidelity_{FILE_TAG}.png")

# ================================================================
# 3. Fidelity vs p figure  (title + annotation auto-filled from data)
# ================================================================
S = np.load(P_NPZ, allow_pickle=True)
P_EVAL   = S['p_eval']
fqae     = S['f_qae']
fqae_std = S['f_qae_std']
fnoi     = S['f_noisy']
fstd     = S['f_std']
best_nl  = int(S['best_n_layers'])

title_fig_p = TITLE_FIG_P_TMPL.format(L=best_nl, p=f"{P_NOISE:g}")
annot_fig_p = ANNOT_FIG_P_TMPL.format(L=best_nl, p=f"{P_NOISE:g}")

fig2, ax = plt.subplots(figsize=(9, 6))
#ax.plot(P_EVAL, fqae, 'o-', color='#2ecc71', lw=2, label=f"SRL (L={best_nl})")
ax.plot(P_EVAL, fqae, 'o-', color='#2ecc71', lw=2, label="SRL (L=3)")
#ax.fill_between(P_EVAL, fqae-fqae_std, fqae+fqae_std, color='#2ecc71', alpha=0.15)
#ax.plot(P_EVAL, fstd, 's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(P_EVAL, fstd, '--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(P_EVAL, fnoi, '^-', color='#e74c3c', lw=2, label="No correction")
ax.axvline(P_NOISE, color='black', ls=':', lw=1.2, alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(P_NOISE + 0.005, _ylo + 0.05*(_yhi-_ylo), annot_fig_p, fontsize=12, va='bottom')
ax.set_xlabel("p", fontsize=15); ax.set_ylabel(r"$\langle F \rangle$",fontsize=15)
ax.tick_params(axis='both', labelsize=12)
#ax.set_title(title_fig_p, fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(f"perfect_fidelity_vs_p_{FILE_TAG}.pdf", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved : perfect_fidelity_vs_p_{FILE_TAG}.pdf")

print("\nDone ✓")