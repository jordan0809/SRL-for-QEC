"""
Reload & re-plot script for the Spectral QEC [[3,1,3]] bit-flip experiment.
Loads the saved .npz files and reconstructs the editable figures.

Requires (from the original run):
  bitflip_results_<tag>.npz
  bitflip_fidelity_vs_p_<tag>.npz
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 0. USER-EDITABLE CONFIG  —  change plot text/style here
# ================================================================
FILE_TAG = "layerwise"          # "multistart" or "layerwise"

RESULTS_NPZ = f"data/bitflip_results_{FILE_TAG}.npz"
P_NPZ       = f"data/bitflip_fidelity_vs_p_{FILE_TAG}.npz"

# --- colors ---
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
LAYER_LS     = ['-', '--', '-', '--', '-']

# --- editable titles ---
TITLE_FIG_TRAIN   = "Training fidelity vs epoch (full curve, both phases)"
TITLE_FIG_LOSS    = "Training loss vs epoch"
TITLE_FIG_TESTBAR = "Test fidelity vs layer count (full bit-flip channel)"
TITLE_FIG_SINGLE  = "Single bit-flip fidelity vs layer count\n(correctable errors)"

# Fig-2 title + annotation are auto-filled from data via {L} and {p} placeholders.
TITLE_FIG_P_TMPL = "Repetition [[3,1,3]] bit-flip — best model (L={L}, trained at p={p})"
ANNOT_FIG_P_TMPL = "trained @ p={p}"

# loss-panel subtitle weights (only text; the run used these)
W_STAB = 1.0; W_LOGICAL = 15.0; W_YL = 15.0

# ================================================================
# 1. Load main results
# ================================================================
D = np.load(RESULTS_NPZ, allow_pickle=True)

P_NOISE       = float(D['meta_p_noise'])
LAYER_SWEEP   = list(np.array(D['meta_layers']).tolist())
USE_LAYERWISE = bool(int(D['meta_layerwise']))
INIT_EPOCHS   = int(D['meta_INIT_EPOCHS'])
MAIN_EPOCHS   = int(D['meta_MAIN_EPOCHS'])
best_nl_save  = int(D['best_n_layers'])
N_TRAIN_STATES = int(D['meta_n_train'])
N_TEST_STATES  = int(D['meta_n_test'])

# Rebuild the per-layer results dict
sweep = {}
for nl in LAYER_SWEEP:
    sweep[nl] = {
        'n_layers':       nl,
        'params':         D[f'L{nl}_params'],
        'hist_epoch':     D[f'L{nl}_epoch'],
        'hist_phase':     D[f'L{nl}_phase'],
        'loss_hist':      D[f'L{nl}_loss_hist'],
        'fid_hist':       D[f'L{nl}_fid_hist'],
        's1_hist':        D[f'L{nl}_s1_hist'],
        's2_hist':        D[f'L{nl}_s2_hist'],
        'f_noisy_full':   D[f'L{nl}_f_noisy_full'],
        'f_std_full':     D[f'L{nl}_f_std_full'],
        'f_qae_full':     D[f'L{nl}_f_qae_full'],
        'f_qae_single':   D[f'L{nl}_f_qae_single'],
    }
    sweep[nl]['n_params'] = int(sweep[nl]['params'].shape[0])

print(f"Loaded {RESULTS_NPZ}")
print(f"  p_noise={P_NOISE}, layers={LAYER_SWEEP}, best L={best_nl_save}, "
      f"train={N_TRAIN_STATES}, test={N_TEST_STATES}, layerwise={USE_LAYERWISE}")

# Empirical reference lines (fixed test set -> identical across layers)
f_std_ref   = sweep[LAYER_SWEEP[0]]['f_std_full'].mean()
f_noisy_ref = sweep[LAYER_SWEEP[0]]['f_noisy_full'].mean()

# ================================================================
# 2. 4-panel comparison figure
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- (0,0) training fidelity vs global epoch ---
ax = axes[0, 0]
for (nl, res), c, ls in zip(sweep.items(), LAYER_COLORS, LAYER_LS):
    ax.plot(res['hist_epoch'], res['fid_hist'], color=c, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_full'].mean():.4f}")
ax.axhline(f_std_ref, color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_ref:.4f}")
ax.axhline(f_noisy_ref, color='gray', ls=':', lw=1.5, label=f"No correction {f_noisy_ref:.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(INIT_EPOCHS+1, 0.53, "Phase 2 →", fontsize=8, color='gray')
ax.text(1, 0.53, "← Phase 1", fontsize=8, color='gray')
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)")
ax.set_ylabel(f"Avg fidelity (all {N_TRAIN_STATES} train states)")
ax.set_title(TITLE_FIG_TRAIN, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim([0.50, 1.02]); ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

# --- (0,1) training loss vs global epoch ---
ax = axes[0, 1]
for (nl, res), c, ls in zip(sweep.items(), LAYER_COLORS, LAYER_LS):
    ax.plot(res['hist_epoch'], res['loss_hist'], color=c, ls=ls, lw=2, label=f"L={nl}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)")
ax.set_ylabel("Training loss")
ax.set_title(f"{TITLE_FIG_LOSS}\nweights S={W_STAB}, log={W_LOGICAL}, YL={W_YL}",
             fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)

# --- (1,0) test fidelity bars (full channel) ---
ax = axes[1, 0]; x = np.arange(len(LAYER_SWEEP)); bw = 0.22
fn_m = [sweep[nl]['f_noisy_full'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep[nl]['f_std_full'].mean()   for nl in LAYER_SWEEP]
fq_m = [sweep[nl]['f_qae_full'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep[nl]['f_qae_full'].std()    for nl in LAYER_SWEEP]
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
ax.legend(fontsize=9); ax.set_ylim([0.50, 1.02]); ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='both', labelsize=12)

# --- (1,1) single bit-flip QAE fidelity vs layer count ---
ax = axes[1, 1]
fqs_m = [sweep[nl]['f_qae_single'].mean() for nl in LAYER_SWEEP]
fqs_s = [sweep[nl]['f_qae_single'].std()  for nl in LAYER_SWEEP]
bars = ax.bar(x, fqs_m, 0.5, color='#2ecc71', alpha=0.85, label="QAE (1-flip)")
ax.errorbar(x, fqs_m, yerr=fqs_s, fmt='none', color='black', capsize=4, lw=1.5)
for b, v in zip(bars, fqs_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.002, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.axhline(1.0, color='#3498db', ls='--', lw=1.5, label="Std QEC (1.0)")
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test states)")
ax.set_title(TITLE_FIG_SINGLE, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim([0.80, 1.02]); ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='both', labelsize=12)

plt.suptitle(f"Spectral QAE — [[3,1,3]] bit-flip layer sweep  |  p={P_NOISE}  "
             f"|  layerwise warm-start={USE_LAYERWISE}",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"bitflip_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved : bitflip_fidelity_{FILE_TAG}.png")

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
ax.plot(P_EVAL, fqae, 'o-', color='#2ecc71', lw=2, label=f"SRL (L={best_nl})")
#ax.fill_between(P_EVAL, fqae-fqae_std, fqae+fqae_std, color='#2ecc71', alpha=0.15)
#ax.plot(P_EVAL, fstd, 's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(P_EVAL, fstd, '--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(P_EVAL, fnoi, '^-', color='#e74c3c', lw=2, label="No correction")
ax.axvline(P_NOISE, color='black', ls=':', lw=1.2, alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(P_NOISE + 0.005, _ylo + 0.05*(_yhi-_ylo), annot_fig_p, fontsize=12, va='bottom')
#ax.set_xlabel("Bit-flip error rate  p"); ax.set_ylabel("Mean fidelity")
ax.set_xlabel("p", fontsize=15); ax.set_ylabel(r"$\langle F \rangle$", fontsize=15)
#ax.set_title(title_fig_p, fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
fig2.savefig(f"bitflip_fidelity_vs_p_{FILE_TAG}.pdf", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved : bitflip_fidelity_vs_p_{FILE_TAG}.pdf")

print("\nDone ✓")