"""
Reload & re-plot script for the Spectral QEC LNCY [[4,1,2]] amplitude-damping
experiment (optional Petz-map benchmark).
Loads the saved .npz files from data/ and reconstructs the editable figures.

Requires:
  data/ampdamp4q_results_<tag>.npz
  data/ampdamp4q_fidelity_vs_gamma_<tag>.npz
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 0. USER-EDITABLE CONFIG  —  change plot text/style here
# ================================================================
FILE_TAG = "layerwise"           # "layerwise" or "multistart"

DATA_DIR    = "data"             # input folder
RESULTS_NPZ = f"{DATA_DIR}/ampdamp4q_results_{FILE_TAG}.npz"
G_NPZ       = f"{DATA_DIR}/ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.npz"

# --- colors ---
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
LAYER_LS     = ['-', '--', '-', '--', '-']
PETZ_COLOR   = '#f39c12'         # Petz-map curve/bar/line color

# --- editable titles ---
TITLE_FIG_TRAIN   = "Training fidelity vs epoch (±1σ)"
TITLE_FIG_LOSS    = "Baseline loss vs epoch"
TITLE_FIG_TESTBAR = "Test fidelity vs layer count"
TITLE_FIG_LOGICAL = "Logical spectrum error vs layer count"

# Fig-2 title + annotation are auto-filled from data via {L}, {g}.
TITLE_FIG_G_TMPL = "LNCY [[4,1,2]] amplitude damping — best model (L={L})"
ANNOT_FIG_G_TMPL = "trained @ γ={g}"

# baseline loss weights (only used for the loss-panel subtitle text)
W_STAB_BASE = 1.0
W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

# ================================================================
# 1. Load main results
# ================================================================
D = np.load(RESULTS_NPZ, allow_pickle=True)
_keys = set(D.files)

GAMMA_DAMP    = float(D['meta_gamma'])
LAYER_SWEEP   = list(np.array(D['meta_layers']).tolist())
USE_LAYERWISE = bool(int(D['meta_layerwise']))
best_nl_save  = int(D['best_n_layers'])

f_noisy_test = D['f_noisy_test']
f_std_test   = D['f_std_test']
N_TEST_STATES = f_noisy_test.shape[0]

# Petz benchmark is optional — auto-detect
HAS_PETZ    = 'f_petz_test' in _keys
f_petz_test = D['f_petz_test'] if HAS_PETZ else None

# INIT_EPOCHS marker: recover from where phase switches 1 -> 2 in the history
_hist_phase = D[f"L{LAYER_SWEEP[0]}_hist_phase"]
_hist_epoch = D[f"L{LAYER_SWEEP[0]}_hist_epoch"]
_p2 = np.where(_hist_phase == 2)[0]
INIT_EPOCHS = int(_hist_epoch[_p2[0]]) if len(_p2) else int(_hist_epoch.max())

# Train-state count is not stored; label generically. (Original run used 100.)
N_TRAIN_STATES = 100

# Rebuild the per-layer results dict
hist_keys = ['epoch', 'phase', 'loss', 'loss_std', 'fid', 'fid_std',
             'dzl', 'dzl_std', 'dxl', 'dxl_std', 'dyl', 'dyl_std']
sweep = {}
for nl in LAYER_SWEEP:
    res = {
        'n_layers':   nl,
        'params':     D[f'L{nl}_params'],
        'f_qae_test': D[f'L{nl}_f_qae_test'],
        'dzl_test':   D[f'L{nl}_dzl_test'],
        'dxl_test':   D[f'L{nl}_dxl_test'],
        'dyl_test':   D[f'L{nl}_dyl_test'],
        'hist':       {k: D[f'L{nl}_hist_{k}'] for k in hist_keys},
    }
    res['n_params']     = int(res['params'].shape[0])
    res['f_noisy_test'] = f_noisy_test
    res['f_std_test']   = f_std_test
    res['f_petz_test']  = f_petz_test
    sweep[nl] = res

print(f"Loaded {RESULTS_NPZ}")
print(f"  γ={GAMMA_DAMP}, layers={LAYER_SWEEP}, best L={best_nl_save}, "
      f"test states={N_TEST_STATES}, layerwise={USE_LAYERWISE}, Petz={HAS_PETZ}")

# ================================================================
# 2. 4-panel comparison figure
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- (0,0) training fidelity (±1σ) ---
ax = axes[0, 0]
for (nl, res), c, ls in zip(sweep.items(), LAYER_COLORS, LAYER_LS):
    h = res['hist']
    ax.plot(h['epoch'], h['fid'], color=c, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(h['epoch'], h['fid']-h['fid_std'], h['fid']+h['fid_std'],
                    color=c, alpha=0.12)
ax.axhline(f_std_test.mean(),  color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_test.mean():.4f}")
if HAS_PETZ:
    ax.axhline(f_petz_test.mean(), color=PETZ_COLOR, ls='-.', lw=1.5, label=f"Petz map {f_petz_test.mean():.4f}")
ax.axhline(f_noisy_test.mean(), color='gray', ls=':', lw=1.5, label=f"No corr {f_noisy_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel(f"Avg fidelity ({N_TRAIN_STATES} train)")
ax.set_title(TITLE_FIG_TRAIN, fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

# --- (0,1) baseline loss ---
ax = axes[0, 1]
for (nl, res), c, ls in zip(sweep.items(), LAYER_COLORS, LAYER_LS):
    h = res['hist']
    ax.plot(h['epoch'], h['loss'], color=c, ls=ls, lw=2, label=f"L={nl}")
    ax.fill_between(h['epoch'], h['loss']-h['loss_std'], h['loss']+h['loss_std'],
                    color=c, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel("Fixed-weight baseline spectral loss")
ax.set_title(f"{TITLE_FIG_LOSS}\nS={W_STAB_BASE}, ZL={W_ZL_BASE}, XL={W_XL_BASE}, YL={W_YL_BASE}",
             fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- (1,0) test fidelity bars (3 or 4 groups depending on Petz) ---
ax = axes[1, 0]; x = np.arange(len(LAYER_SWEEP))
fn_m = [sweep[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep[nl]['f_std_test'].mean()   for nl in LAYER_SWEEP]
fq_m = [sweep[nl]['f_qae_test'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep[nl]['f_qae_test'].std()    for nl in LAYER_SWEEP]

if HAS_PETZ:
    fp_m = [sweep[nl]['f_petz_test'].mean() for nl in LAYER_SWEEP]
    bw = 0.2
    ax.bar(x-1.5*bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
    ax.bar(x-0.5*bw, fs_m, bw, label="Standard QEC (LNCY)", color='#3498db', alpha=0.8)
    ax.bar(x+0.5*bw, fp_m, bw, label="Petz map", color=PETZ_COLOR, alpha=0.8)
    bars = ax.bar(x+1.5*bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
    qx = x+1.5*bw
else:
    bw = 0.25
    ax.bar(x-bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
    ax.bar(x,    fs_m, bw, label="Standard QEC (LNCY)", color='#3498db', alpha=0.8)
    bars = ax.bar(x+bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
    qx = x+bw

ax.errorbar(qx, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4)
for b, v in zip(bars, fq_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test)")
ax.set_title(TITLE_FIG_TESTBAR, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# --- (1,1) logical spectrum error bars ---
ax = axes[1, 1]; bw2 = 0.25
dz = [sweep[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dx = [sweep[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dy = [sweep[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
ax.bar(x-bw2, dz, bw2, label="ΔZL", color='#e74c3c', alpha=0.85)
ax.bar(x,     dx, bw2, label="ΔXL", color='#3498db', alpha=0.85)
ax.bar(x+bw2, dy, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel("Mean |target − recovered|")
ax.set_title(TITLE_FIG_LOGICAL, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Spectral QAE — LNCY [[4,1,2]] amplitude damping "
             f"(γ={GAMMA_DAMP}) | layerwise={USE_LAYERWISE}",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"ampdamp4q_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved : ampdamp4q_fidelity_{FILE_TAG}.png")

# ================================================================
# 3. Fidelity vs gamma figure  (title + annotation auto-filled from data)
# ================================================================
G = np.load(G_NPZ, allow_pickle=True)
GAMMA_SWEEP = G['gamma']
fqae        = G['f_qae']
fqae_std    = G['f_qae_std']
fnoi        = G['f_noisy']
fstd        = G['f_std']
best_nl     = int(G['best_n_layers'])
has_petz_g  = 'f_petz' in set(G.files)
fpetz       = G['f_petz'] if has_petz_g else None

title_fig_g = TITLE_FIG_G_TMPL.format(L=best_nl)
annot_fig_g = ANNOT_FIG_G_TMPL.format(g=f"{GAMMA_DAMP:g}")

fig2, ax = plt.subplots(figsize=(9, 6))
ax.plot(GAMMA_SWEEP, fqae, 'o-', color='#2ecc71', lw=2, label=f"SRL (L={best_nl})")
#ax.fill_between(GAMMA_SWEEP, fqae-fqae_std, fqae+fqae_std, color='#2ecc71', alpha=0.15)
#ax.plot(GAMMA_SWEEP, fstd,  's--', color='#3498db', lw=2, label="Standard QEC (LNCY)")
ax.plot(GAMMA_SWEEP, fstd,  '--', color='#3498db', lw=2, label="Standard QEC (LNCY)")
if has_petz_g:
    ax.plot(GAMMA_SWEEP, fpetz, 'd-.', color=PETZ_COLOR, lw=2, label="Petz map")
ax.plot(GAMMA_SWEEP, fnoi,  '^-',  color='#e74c3c', lw=2, label="No correction")
ax.axvline(GAMMA_DAMP, color='black', ls=':', alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(GAMMA_DAMP + 0.02*(GAMMA_SWEEP.max()-GAMMA_SWEEP.min()),
        _ylo + 0.05*(_yhi-_ylo), annot_fig_g, fontsize=12, va='bottom')
ax.set_xlabel("γ",fontsize=15); ax.set_ylabel(r"$\langle F \rangle$",fontsize=15)
#ax.set_title(title_fig_g, fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
fig2.savefig(f"ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.pdf", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved : ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.pdf")

print("\nDone ✓")