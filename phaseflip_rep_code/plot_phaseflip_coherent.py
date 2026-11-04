"""
Reload & re-plot script for the Spectral QEC dephasing experiment.

Requires:
  dephasing_results_<tag>.npz
  dephasing_fidelity_vs_sigma_<tag>.npz
"""

import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 0. USER-EDITABLE CONFIG  —  change plot text/style here
# ================================================================
FILE_TAG = "layerwise"          # "multistart" or "layerwise" (match your run)

RESULTS_NPZ = f"data/dephasing_results_{FILE_TAG}.npz"
SIGMA_NPZ   = f"data/dephasing_fidelity_vs_sigma_{FILE_TAG}.npz"

# --- colors ---
PETZ_COLOR = '#f39c12'
LAYER_COLORS = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
LAYER_LS     = ['-', '--', '-', '--', '-']

# --- editable titles ---
TITLE_PTM_SUFFIX   = "X↔Y mixing appears only for μ≠0"
TITLE_FIG_TRAIN    = "Training fidelity vs epoch (±1σ band)"
TITLE_FIG_LOSS     = "Baseline loss vs epoch"
TITLE_FIG_TESTBAR  = "Test fidelity vs layer count"
TITLE_FIG_LOGICAL  = "Logical spectrum error vs layer count"
# >>> requested change: <<<
TITLE_FIG_SIGMA    = "Phase-flip [[3,1,3]] coherent dephasing — best model"

# fixed baseline weights (only used for the loss-panel subtitle text)
W_STAB_BASE = 1.0
W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

# ================================================================
# 1. Load main results
# ================================================================
D = np.load(RESULTS_NPZ, allow_pickle=True)

SIGMA_THETA = float(D['meta_sigma'])
RHO_CORR    = float(D['meta_rho_corr'])
MU_COHERENT = float(D['meta_mu'])
LAYER_SWEEP = list(np.array(D['meta_layers']).tolist())
INIT_EPOCHS = int(D['meta_INIT_EPOCHS'])
USE_LAYERWISE = bool(int(D['meta_layerwise']))

f_noisy_test = D['f_noisy_test']
f_std_test   = D['f_std_test']
f_petz_test  = D['f_petz_test']

N_TEST_STATES  = f_noisy_test.shape[0]
best_nl_save   = int(D['best_n_layers'])

# Rebuild the per-layer results dict
sweep_results = {}
per_layer_keys = ['params', 'hist_epoch', 'hist_phase',
                  'hist_loss', 'hist_loss_std', 'hist_fid', 'hist_fid_std',
                  'hist_dzl', 'hist_dzl_std', 'hist_dxl', 'hist_dxl_std',
                  'hist_dyl', 'hist_dyl_std',
                  'f_qae_test', 'dzl_test', 'dxl_test', 'dyl_test']
for nl in LAYER_SWEEP:
    res = {k: D[f'L{nl}_{k}'] for k in per_layer_keys}
    res['n_layers'] = nl
    res['n_params'] = int(res['params'].shape[0])
    # these three are stored globally in the .npz, not per-layer:
    res['f_noisy_test'] = f_noisy_test
    res['f_std_test']   = f_std_test
    res['f_petz_test']  = f_petz_test
    sweep_results[nl] = res

print(f"Loaded {RESULTS_NPZ}")
print(f"  sigma={SIGMA_THETA}, rho_corr={RHO_CORR}, mu={MU_COHERENT}")
print(f"  layers={LAYER_SWEEP}, best L={best_nl_save}, test states={N_TEST_STATES}")

# ================================================================
# 2. Single-qubit PTM figure (reconstructed analytically)
# ================================================================
def single_qubit_ptm(sigma, mu):
    c = np.exp(-0.5*sigma**2)*np.cos(mu)
    s = np.exp(-0.5*sigma**2)*np.sin(mu)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])

fig0, axp = plt.subplots(1, 2, figsize=(10, 4))
for a_, m_, ttl in [(axp[0], 0.0, "μ=0  (diagonal)"),
                    (axp[1], MU_COHERENT, f"μ={MU_COHERENT}  (non-diagonal)")]:
    K = single_qubit_ptm(SIGMA_THETA, m_)
    im = a_.imshow(K, cmap="RdBu", vmin=-1, vmax=1)
    a_.set_xticks(range(4)); a_.set_xticklabels(["I", "X", "Y", "Z"])
    a_.set_yticks(range(4)); a_.set_yticklabels(["I", "X", "Y", "Z"])
    a_.set_title(ttl, fontweight="bold")
    for i in range(4):
        for j in range(4):
            a_.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig0.colorbar(im, ax=axp, fraction=0.046)
fig0.suptitle(f"Single-qubit PTM (σ={SIGMA_THETA}): {TITLE_PTM_SUFFIX}",
              fontweight="bold")
fig0.savefig(f"dephasing_ptm_{FILE_TAG}.png", dpi=150, bbox_inches="tight")
plt.close(fig0)
print(f"  Saved : dephasing_ptm_{FILE_TAG}.png")

# ================================================================
# 3. 4-panel comparison figure
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- (0,0) training fidelity ---
ax = axes[0, 0]
for (nl, res), col, ls in zip(sweep_results.items(), LAYER_COLORS, LAYER_LS):
    ax.plot(res['hist_epoch'], res['hist_fid'], color=col, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(res['hist_epoch'], res['hist_fid']-res['hist_fid_std'],
                    res['hist_fid']+res['hist_fid_std'], color=col, alpha=0.12)
ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_test.mean():.4f}")
ax.axhline(f_petz_test.mean(), color=PETZ_COLOR, ls='-.', lw=1.5, label=f"Petz map {f_petz_test.mean():.4f}")
ax.axhline(f_noisy_test.mean(), color='gray', ls=':', lw=1.5, label=f"No corr {f_noisy_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)"); ax.set_ylabel("Avg fidelity (train)")
ax.set_title(TITLE_FIG_TRAIN, fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

# --- (0,1) baseline loss ---
ax = axes[0, 1]
for (nl, res), col, ls in zip(sweep_results.items(), LAYER_COLORS, LAYER_LS):
    ax.plot(res['hist_epoch'], res['hist_loss'], color=col, ls=ls, lw=2, label=f"L={nl}")
    ax.fill_between(res['hist_epoch'], res['hist_loss']-res['hist_loss_std'],
                    res['hist_loss']+res['hist_loss_std'], color=col, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel("Fixed-weight baseline spectral loss")
ax.set_title(f"{TITLE_FIG_LOSS}\nS={W_STAB_BASE}, ZL={W_ZL_BASE}, XL={W_XL_BASE}, YL={W_YL_BASE}",
             fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# --- (1,0) test fidelity bars ---
ax = axes[1, 0]
x = np.arange(len(LAYER_SWEEP)); bw = 0.2
fn_m = [sweep_results[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep_results[nl]['f_std_test'].mean()   for nl in LAYER_SWEEP]
fp_m = [sweep_results[nl]['f_petz_test'].mean()  for nl in LAYER_SWEEP]
fq_m = [sweep_results[nl]['f_qae_test'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep_results[nl]['f_qae_test'].std()    for nl in LAYER_SWEEP]
ax.bar(x-1.5*bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x-0.5*bw, fs_m, bw, label="Standard QEC", color='#3498db', alpha=0.8)
ax.bar(x+0.5*bw, fp_m, bw, label="Petz map", color=PETZ_COLOR, alpha=0.8)
bars = ax.bar(x+1.5*bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
ax.errorbar(x+1.5*bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4, lw=1.5)
for bar, val in zip(bars, fq_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{val:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test)")
ax.set_title(TITLE_FIG_TESTBAR, fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# --- (1,1) logical spectrum error bars ---
ax = axes[1, 1]
dzl_m = [sweep_results[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dxl_m = [sweep_results[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dyl_m = [sweep_results[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
bw2 = 0.25
ax.bar(x-bw2, dzl_m, bw2, label="ΔZL", color='#e74c3c', alpha=0.85)
ax.bar(x,     dxl_m, bw2, label="ΔXL", color='#3498db', alpha=0.85)
ax.bar(x+bw2, dyl_m, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel("Mean |target − recovered|")
ax.set_title(TITLE_FIG_LOGICAL, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Spectral QAE — [[3,1,3]] Phase-Flip, correlated coherent dephasing "
             f"(σ={SIGMA_THETA}, ρ_corr={RHO_CORR}, μ={MU_COHERENT} "
             f"[{'NON-diag PTM' if MU_COHERENT!=0 else 'diag PTM'}])  |  layerwise={USE_LAYERWISE}",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"dephasing_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved : dephasing_fidelity_{FILE_TAG}.png")

# ================================================================
# 4. Fidelity vs sigma figure  (title change applied here)
# ================================================================
S = np.load(SIGMA_NPZ, allow_pickle=True)
SIGMA_SWEEP   = S['sigma']
fqae_vs_s     = S['f_qae']
fqae_std_vs_s = S['f_qae_std']
fnoisy_vs_s   = S['f_noisy']
fstd_vs_s     = S['f_std']
fpetz_vs_s    = S['f_petz']
best_nl       = int(S['best_n_layers'])
rho_corr_s    = float(S['rho_corr'])
mu_s          = float(S['mu'])

fig2, ax = plt.subplots(figsize=(9, 6))
#ax.plot(SIGMA_SWEEP, fqae_vs_s, 'o-', color='#2ecc71', lw=2, label=f"SRL (L={best_nl})")
ax.plot(SIGMA_SWEEP, fqae_vs_s, 'o-', color='#2ecc71', lw=2, label="SRL (L=2)")
#ax.fill_between(SIGMA_SWEEP, fqae_vs_s-fqae_std_vs_s, fqae_vs_s+fqae_std_vs_s,
#                color='#2ecc71', alpha=0.15)
ax.plot(SIGMA_SWEEP, fstd_vs_s,  's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(SIGMA_SWEEP, fpetz_vs_s, 'd-.', color=PETZ_COLOR, lw=2, label="Petz map")
ax.plot(SIGMA_SWEEP, fnoisy_vs_s, '^-', color='#e74c3c', lw=2, label="No correction")
ax.axvline(SIGMA_THETA, color='black', ls=':', lw=1.2, alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(SIGMA_THETA + 0.02, _ylo + 0.05*(_yhi-_ylo),
        f"trained @ σ={SIGMA_THETA}", fontsize=12, va='bottom')
ax.set_xlabel("σ",fontsize=15); ax.set_ylabel(r"$\langle F \rangle$", fontsize=15)
#ax.set_title(f"{TITLE_FIG_SIGMA} "
#             f"(L={best_nl}, ρ_corr={rho_corr_s}, μ={mu_s})",
#             fontweight='bold')
ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
fig2.savefig(f"dephasing_fidelity_vs_sigma_{FILE_TAG}.pdf", dpi=150, bbox_inches='tight')
plt.close(fig2)
print(f"  Saved : dephasing_fidelity_vs_sigma_{FILE_TAG}.pdf")

print("\nDone ✓")