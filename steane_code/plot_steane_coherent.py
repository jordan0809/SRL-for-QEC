"""
Rebuild Steane correlated-dephasing figures from saved data.
"""

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- must match the training run ----
USE_LAYERWISE = True
FILE_TAG = "layerwise" if USE_LAYERWISE else "multistart"
PETZ_COLOR = '#f39c12'         

# ================================================================
# Load saved data
# ================================================================
res = np.load(f"data/steane_coherent_results_{FILE_TAG}.npz", allow_pickle=True)

LAYER_SWEEP = list(res['meta_layers'])
SIGMA_THETA = float(res['meta_sigma'])
RHO_CORR    = float(res['meta_rho_corr'])
MU_COHERENT = float(res['meta_mu'])
best_nl     = int(res['best_n_layers'])
N_TRAIN_STATES = int(res['meta_test_thetas'].shape[0]) if 'meta_train' in res else None

f_noisy_test = res['f_noisy_test']
f_std_test   = res['f_std_test']
f_petz_test  = res['f_petz_test']

N_TEST_STATES = f_noisy_test.shape[0]

# rebuild per-layer sweep dict from saved arrays
def load_hist(nl):
    keys = ['epoch','phase','loss','loss_std','fid','fid_std','dzl','dzl_std',
            'dxl','dxl_std','dyl','dyl_std']
    return {k: res[f'L{nl}_hist_{k}'] for k in keys}

sweep = {}
for nl in LAYER_SWEEP:
    sweep[nl] = dict(
        n_params   = int(res[f'L{nl}_params'].shape[0]),
        f_qae_test = res[f'L{nl}_f_qae_test'],
        dzl_test   = res[f'L{nl}_dzl_test'],
        dxl_test   = res[f'L{nl}_dxl_test'],
        dyl_test   = res[f'L{nl}_dyl_test'],
        f_noisy_test = f_noisy_test,
        f_std_test   = f_std_test,
        f_petz_test  = f_petz_test,
        hist       = load_hist(nl),
    )

# n_params in the results npz is the param vector length; that IS count_params
# (kept as-is from the saved 'L{nl}_params' array length)

# ================================================================
# 4-panel plot
# ================================================================
colors = ['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']; ls_ops = ['-','--','-','--','-']
try:
    import jax
    backend = jax.default_backend().upper()
except Exception:
    backend = "N/A"

INIT_EPOCHS = int(sweep[LAYER_SWEEP[0]]['hist']['epoch'][
    np.argmax(sweep[LAYER_SWEEP[0]]['hist']['phase'] == 2)])  # first phase-2 epoch

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- panel [0,0] : training fidelity vs epoch ---
ax = axes[0, 0]
for (nl, r), c, ls in zip(sweep.items(), colors, ls_ops):
    h = r['hist']
    ax.plot(h['epoch'], h['fid'], color=c, ls=ls, lw=2,
            label=f"L={nl} ({r['n_params']}p) F̄={r['f_qae_test'].mean():.4f}")
    ax.fill_between(h['epoch'], h['fid']-h['fid_std'], h['fid']+h['fid_std'], color=c, alpha=0.12)
ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_test.mean():.4f}")
ax.axhline(f_petz_test.mean(), color=PETZ_COLOR, ls='-.', lw=1.5, label=f"Petz map {f_petz_test.mean():.4f}")
ax.axhline(f_noisy_test.mean(), color='gray', ls=':', lw=1.5, label=f"No corr {f_noisy_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel("Avg fidelity (train)")
ax.set_title("Training fidelity vs epoch (±1σ)", fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

# --- panel [0,1] : baseline loss vs epoch ---
ax = axes[0, 1]
for (nl, r), c, ls in zip(sweep.items(), colors, ls_ops):
    h = r['hist']
    ax.plot(h['epoch'], h['loss'], color=c, ls=ls, lw=2, label=f"L={nl}")
    ax.fill_between(h['epoch'], h['loss']-h['loss_std'], h['loss']+h['loss_std'], color=c, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel("Fixed-weight baseline spectral loss")
ax.set_title("Baseline loss vs epoch", fontweight='bold'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- panel [1,0] : test fidelity vs layer (4 bars incl. Petz) ---
ax = axes[1, 0]; x = np.arange(len(LAYER_SWEEP)); bw = 0.2
fn_m = [sweep[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep[nl]['f_std_test'].mean() for nl in LAYER_SWEEP]
fp_m = [sweep[nl]['f_petz_test'].mean() for nl in LAYER_SWEEP]
fq_m = [sweep[nl]['f_qae_test'].mean() for nl in LAYER_SWEEP]
fq_s = [sweep[nl]['f_qae_test'].std() for nl in LAYER_SWEEP]
ax.bar(x-1.5*bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x-0.5*bw, fs_m, bw, label="Standard QEC", color='#3498db', alpha=0.8)
ax.bar(x+0.5*bw, fp_m, bw, label="Petz map", color=PETZ_COLOR, alpha=0.8)
bars = ax.bar(x+1.5*bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
ax.errorbar(x+1.5*bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4)
for b, v in zip(bars, fq_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test)")
ax.set_title("Test fidelity vs layer count", fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# --- panel [1,1] : logical spectrum error vs layer ---
ax = axes[1, 1]; bw2 = 0.25
dz = [sweep[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dx = [sweep[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dy = [sweep[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
ax.bar(x-bw2, dz, bw2, label="ΔZL", color='#e74c3c', alpha=0.85)
ax.bar(x,     dx, bw2, label="ΔXL", color='#3498db', alpha=0.85)
ax.bar(x+bw2, dy, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel("Mean |target − recovered|")
ax.set_title("Logical spectrum error vs layer count", fontweight='bold'); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Spectral QAE — Steane [[7,1,3]] coherent dephasing "
             f"(σ={SIGMA_THETA}, ρ={RHO_CORR}, μ={MU_COHERENT} "
             f"[{'NON-diag PTM' if MU_COHERENT!=0 else 'diag PTM'}]) | JAX {backend} | layerwise={USE_LAYERWISE}",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"steane_coherent_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : steane_coherent_fidelity_{FILE_TAG}.png  (Petz color {PETZ_COLOR})")

# ================================================================
# Fidelity vs sigma plot
# ================================================================
sig_npz = np.load(f"data/steane_coherent_fidelity_vs_sigma_{FILE_TAG}.npz", allow_pickle=True)
SIGMA_SWEEP = sig_npz['sigma']
fqae     = sig_npz['f_qae']
fqae_std = sig_npz['f_qae_std']
fnoi     = sig_npz['f_noisy']
fstd     = sig_npz['f_std']
fpetz    = sig_npz['f_petz']

fig2, ax = plt.subplots(figsize=(9, 6))
ax.plot(SIGMA_SWEEP, fqae, 'o-', color='#2ecc71', lw=2, label=f"SRL (L={best_nl})")
#ax.fill_between(SIGMA_SWEEP, fqae-fqae_std, fqae+fqae_std, color='#2ecc71', alpha=0.15)
ax.plot(SIGMA_SWEEP, fstd,  's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(SIGMA_SWEEP, fpetz, 'd-.', color=PETZ_COLOR, lw=2, label="Petz map")
ax.plot(SIGMA_SWEEP, fnoi,  '^-',  color='#e74c3c', lw=2, label="No correction")
ax.axvline(SIGMA_THETA, color='black', ls=':', alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(SIGMA_THETA + 0.02, _ylo + 0.05*(_yhi-_ylo),
        f"trained @ σ={SIGMA_THETA}", fontsize=12, rotation=0, va='bottom')
#ax.text(SIGMA_THETA+0.02, 0.6, f"trained @ σ={SIGMA_THETA}", fontsize=8)
ax.set_xlabel("σ", fontsize=15); ax.set_ylabel(r"$\langle F \rangle$", fontsize=15)
#ax.set_title(f"Steane [[7,1,3]] coherent dephasing — best model (L={best_nl}, ρ={RHO_CORR}, μ={MU_COHERENT})",
#             fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
fig2.savefig(f"steane_coherent_fidelity_vs_sigma_{FILE_TAG}.pdf", dpi=150, bbox_inches='tight')
print(f"  Saved : steane_coherent_fidelity_vs_sigma_{FILE_TAG}.pdf  (Petz color {PETZ_COLOR})")
print("\n  Done ✓ — figures rebuilt from saved data, Petz color updated.")