# ============================================================================
# plot_noisy_noiseless.py -- memory-protocol figures for the [[3,1,3]] phase-flip code
#                    with CLOSED-FORM baselines and analytic break-even curves
#
#  FIG A : grid, one panel per p2q, x = sigma_i, baselines + QAE
#  FIG B : P vs p2q at sigma_i = sigma_train (+ net gain, + noisy-training benefit)
#  FIG C : (sigma_i, p2q) 3-regime phase diagram; white contours = NUMERIC
#          boundaries from the data, coloured curves = ANALYTIC
# ============================================================================
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib.colors import ListedColormap, BoundaryNorm

# ------------------------------------------------------------------ files
C_NL, C_NS = "internal_noiseless_trained.npz", "internal_noisy_trained.npz"
NCOL, NROW = 3, 3
SHOW_BANDS = False
SHOW_IDEAL = True

COL = dict(uncorr='#e41a1c', bare='#ff7f00', nltrain='#4daf4a',
           nstrain='#377eb8', ideal='#4d4d4d', train='k')
# LBL = dict(uncorr="No QAE (logical)", bare="Bare physical qubit",
#            nltrain="Noiseless trained QAE", nstrain="Noisy trained QAE",
#            ideal=r"ideal QAE ($p_{2q}=0$)")
LBL = dict(uncorr="Uncorrected logical", bare="Bare physical",
           nltrain="Noiseless-trained SRL", nstrain="Noisy-trained SRL",
           ideal=r"ideal SRL ($p_{2q}=0$)")

# ======================= ANALYTIC-MODEL SWITCHES ============================
SHOW_FIT_REPORT    = True      # mu, rho, C, eps, model-vs-data tables
ANALYTIC_ON_PHASE  = True      # overlay analytic curves on FIG C
ANALYTIC_SOURCE    = 'model'   # 'model' (closed form) | 'data' | 'both'
ANALYTIC_IN_FIGB   = True
SHOW_MODEL_IN_FIGA = False
ANALYTIC_LEGEND    = True

MU_SOURCE   = 'fit'            # 'fit' (from P_bare) | 'fixed'
MU_FIXED    = 0.3
RHO_SOURCE  = 'fit'            # 'fit' (from P_uncorr) | 'fixed'
RHO_FIXED   = 0.3

P_INF       = 0.5              # depolarizing floor of the INTERNAL gate noise
C_SOURCE    = 'fit'            # 'fit' | 'file' (c_eff) 
C_FIT_MODE  = 'nonlinear'      # 'nonlinear' | 'linear'
C_FIT_CPMAX = 1.0              # drop saturated points with C*p2q > CPMAX
C_FIT_FLOOR = 2e-3
MODEL_USE_PANEL_C = False      # False -> one C (noiseless fit) on both panels

EPS_MODE    = 'per_sigma'      # 'global' | 'per_sigma' (piecewise linear)
EPS_MAX     = 3.0
NFINE_ANA   = 1200
ANA_COL     = dict(unc='#ff00ff', bare='#00e5ff')
ANA_STROKE  = 3.2


def idx(a, v):
    return int(np.argmin(np.abs(np.asarray(a) - v)))


def get(d, key, default=None):
    return np.asarray(d[key]) if key in d.files else default


# ============================================================================
# 1. exact closed forms (no expansion in mu, sigma or eps)
# ============================================================================
def C1_of(m, v):
    return np.cos(m)*np.exp(-0.5*v)


def C2_of(m, v, rho):                      # not needed (cancels), kept as check
    return 0.5*(np.cos(2*m)*np.exp(-v*(1.0 + rho))
                + np.exp(-v*(1.0 - rho)))


def C3_of(m, v, rho):
    return 0.25*(np.cos(3*m)*np.exp(-1.5*v*(1.0 + 2.0*rho))
                 + 3.0*np.cos(m)*np.exp(-0.5*v*(3.0 - 2.0*rho)))


def PL_code(m, v, rho):
    """majority-vote logical error probability, Eq. (PL)."""
    return 0.5 - 0.75*C1_of(m, v) + 0.25*C3_of(m, v, rho)


def P_code(m, v, rho):
    """averaged fidelity of the encoded qubit, Eq. (P-code)."""
    return 1.0 - (2.0/3.0)*PL_code(m, v, rho)


def P_bare_exact(sig, mu):
    sig = np.asarray(sig, float)
    return 2.0/3.0 + (1.0/3.0)*np.cos(2.0*mu)*np.exp(-sig**2)


def P_unc_exact(sig, mu, rho):
    sig = np.asarray(sig, float)
    return P_code(2.0*mu, 2.0*sig**2, rho)


def P_ideal_exact(sig, eps, mu, rho):
    """eps = 0 -> perfect recovery ; eps = 1 -> exactly P_unc_exact."""
    sig = np.asarray(sig, float)
    eps = np.asarray(eps, float)
    return P_code((1.0 + eps)*mu, (1.0 + eps**2)*sig**2, rho)


def P_decay(p2q, P_ideal, C, P_inf=P_INF):
    return P_inf + np.exp(-C*np.asarray(p2q, float))*(np.asarray(P_ideal) - P_inf)


def p_crit_eq28(P_ideal, P_ref, C, P_inf=P_INF):
    num = np.atleast_1d(np.asarray(P_ideal, float)) - P_inf
    den = np.atleast_1d(np.asarray(P_ref,   float)) - P_inf
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, np.nan)
    if not (np.isfinite(C) and C > 0):
        return out
    ok = (num > 0) & (den > 0) & (num > den)
    out[ok] = np.log(num[ok]/den[ok])/C
    return out


# ------------------------------- tiny solvers -------------------------------
def _root(f, a, b, tol=1e-13, itmax=300):
    fa, fb = f(a), f(b)
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa*fb > 0:
        return np.nan
    for _ in range(itmax):
        m = 0.5*(a + b); fm = f(m)
        if fa*fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
        if b - a < tol:
            break
    return 0.5*(a + b)


def _golden_min(f, a, b, tol=1e-11, itmax=400):
    gr = 0.5*(np.sqrt(5.0) - 1.0)
    c = b - gr*(b - a); d = a + gr*(b - a)
    fc, fd = f(c), f(d)
    for _ in range(itmax):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr*(b - a); fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr*(b - a); fd = f(d)
        if abs(b - a) < tol*(abs(a) + abs(b) + 1e-12):
            break
    return 0.5*(a + b)


# ---------------------------- parameter extraction --------------------------
def extract_mu(sig, P_ba):
    """cos 2mu = (3 P_bare - 2) exp(sigma^2)  -- sigma-independent by construction."""
    c = np.clip((3.0*np.asarray(P_ba, float) - 2.0)*np.exp(np.asarray(sig, float)**2),
                -1.0, 1.0)
    mu_each = 0.5*np.arccos(c)
    return float(np.median(mu_each)), mu_each, c


def fit_rho(sig, P_un, mu):
    """single-parameter least squares of the uncorrected baseline."""
    sse = lambda r: float(np.sum((P_unc_exact(sig, mu, r) - P_un)**2))
    r = _golden_min(sse, 0.0, 0.999)
    return float(np.clip(r, 0.0, 0.999)), float(np.sqrt(sse(r)/len(sig)))


def eps_at(sig, P_target, mu, rho, lo=0.0, hi=EPS_MAX):
    """invert P_ideal(sigma, eps) = P_target ; monotone decreasing in eps."""
    f = lambda e: float(P_ideal_exact(sig, e, mu, rho)) - float(P_target)
    if f(lo) <= 0.0:
        return lo
    if f(hi) >= 0.0:
        return hi
    return _root(f, lo, hi)


def fit_eps_global(sig, P_id, mu, rho):
    sse = lambda e: float(np.sum((P_ideal_exact(sig, e, mu, rho) - P_id)**2))
    e = _golden_min(sse, 0.0, EPS_MAX)
    return float(np.clip(e, 0.0, EPS_MAX)), float(np.sqrt(sse(e)/len(sig)))


def fit_C(p2q, P, P_ideal, P_inf=P_INF, cp_max=C_FIT_CPMAX,
          floor=C_FIT_FLOOR, mode=C_FIT_MODE, iters=6):
    """Fit P = P_inf + exp(-C p2q)(P_ideal - P_inf); saturated points dropped."""
    p2q = np.asarray(p2q, float)
    P = np.atleast_2d(np.asarray(P, float))
    if P.shape[0] != p2q.size:
        P = P.T
    P_ideal = np.asarray(P_ideal, float).ravel()
    X = np.repeat(p2q[:, None], P.shape[1], axis=1)
    num = P - P_inf
    den = np.broadcast_to(P_ideal - P_inf, P.shape)
    ok = (X > 0) & (num > floor) & (den > floor)
    if not ok.any():
        return dict(C=np.nan, C_lin=np.nan, C_nl=np.nan, rms=np.nan, n_used=0,
                    C_per_sig=np.full(P.shape[1], np.nan))
    Y = np.zeros_like(num); Y[ok] = -np.log(num[ok]/den[ok])
    w = ok.astype(float)
    C_lin = float(np.sum(w*X*Y)/np.sum(w*X*X))
    for _ in range(iters):
        w_new = np.where(ok & (C_lin*X < cp_max), 1.0, 0.0)
        if w_new.sum() < 2:
            break
        C_new = float(np.sum(w_new*X*Y)/np.sum(w_new*X*X))
        if abs(C_new - C_lin) < 1e-9*max(1.0, abs(C_lin)):
            C_lin, w = C_new, w_new
            break
        C_lin, w = C_new, w_new
    use = w > 0
    sse = lambda C: float(np.sum((P_inf + np.exp(-C*X[use])*den[use] - P[use])**2))
    C_nl = _golden_min(sse, max(1e-3, 0.2*C_lin), 5.0*C_lin)
    C = C_nl if mode == 'nonlinear' else C_lin
    rms = float(np.sqrt(sse(C)/max(1, int(use.sum()))))
    C_s = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        mj = use[:, j]
        if mj.sum() >= 2:
            C_s[j] = float(np.sum(X[mj, j]*Y[mj, j])/np.sum(X[mj, j]**2))
    return dict(C=C, C_lin=C_lin, C_nl=C_nl, rms=rms, n_used=int(use.sum()),
                C_per_sig=C_s)


def crossover(pv, fv, thr):
    pv = np.asarray(pv, float); fv = np.asarray(fv, float)
    o = np.argsort(pv); pv, fv = pv[o], fv[o]
    for i in range(len(pv)-1):
        a, b = fv[i]-thr, fv[i+1]-thr
        if a > 0 >= b and pv[i] > 0:
            w = a/(a-b)
            return 10**(np.log10(pv[i]) + w*(np.log10(pv[i+1])-np.log10(pv[i])))
    return None


def interp_sign_change(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.where(np.diff(np.sign(y)) != 0)[0]
    if not s.size:
        return np.nan
    i = s[0]; w = y[i]/(y[i] - y[i+1])
    return float(x[i] + w*(x[i+1] - x[i]))


# ============================================================================
# 2. load
# ============================================================================
dn = np.load(C_NL, allow_pickle=True)
HAVE_NS = os.path.exists(C_NS)
ds = np.load(C_NS, allow_pickle=True) if HAVE_NS else None

sig  = np.asarray(dn['sigma'], float)
p_nl = np.asarray(dn['p2q'], float)
P_nl = np.asarray(dn['p_corr'], float)
S_nl = np.asarray(get(dn, 'p_corr_std', np.zeros_like(P_nl)), float)
P_un = np.asarray(dn['p_uncorr'], float).ravel()
P_ba = np.asarray(dn['p_bare'], float).ravel()

_pid = get(dn, 'p_ideal_qae')
if _pid is None:
    _pid = P_nl[idx(p_nl, 0.0)]
P_id = np.asarray(_pid, float).ravel()

L     = int(dn['n_layers'])
nc    = int(get(dn, 'n_cnot_full', 0))
nr    = int(get(dn, 'n_rot_full', 0))
c_eff = float(get(dn, 'c_eff', np.nan))
sig_tr = float(get(ds if HAVE_NS else dn, 'sigma_train',
                   np.asarray(sig[len(sig)//2])))
i_tr = idx(sig, sig_tr)

if HAVE_NS:
    assert np.allclose(sig, np.asarray(ds['sigma'])), "sigma_i grid mismatch"
    p_ns = np.asarray(ds['p2q'], float)
    P_ns = np.asarray(ds['p_aware'], float)
    S_ns = np.asarray(get(ds, 'p_aware_std', np.zeros_like(P_ns)), float)
else:
    p_ns = np.array([]); P_ns = None; S_ns = None

mnl = p_nl > 0
mns = p_ns > 0 if HAVE_NS else np.array([], dtype=bool)

print(f"  sigma_i grid ({len(sig)}) : {np.round(sig, 3)}")
print(f"  L={L}, {nc} CNOT + {nr} rot, sigma_train={sig_tr}")
print(f"  noisy-trained file: {'yes' if HAVE_NS else 'no'}")

# ============================================================================
# 3. calibrate the closed-form model
# ============================================================================
MU_FIT, MU_EACH, COS2MU = extract_mu(sig, P_ba)
MU = MU_FIT if MU_SOURCE == 'fit' else MU_FIXED

RHO_FIT, RHO_RMS = fit_rho(sig, P_un, MU)
RHO = RHO_FIT if RHO_SOURCE == 'fit' else RHO_FIXED

EPS_G, EPS_RMS = fit_eps_global(sig, P_id, MU, RHO)
EPS_SIG = np.array([eps_at(s, P, MU, RHO) for s, P in zip(sig, P_id)])

FIT_NL = fit_C(p_nl, P_nl, P_id, P_INF)
FIT_NS = fit_C(p_ns, P_ns, P_id, P_INF) if HAVE_NS else None
C_MAIN = FIT_NL['C'] if (C_SOURCE == 'fit' and np.isfinite(FIT_NL['C'])) else c_eff

P_ba_x = P_bare_exact(sig, MU)
P_un_x = P_unc_exact(sig, MU, RHO)
P_id_x = P_ideal_exact(sig, EPS_SIG if EPS_MODE == 'per_sigma' else EPS_G, MU, RHO)


def eps_for(sig_vals):
    if EPS_MODE == 'per_sigma':
        return np.interp(np.asarray(sig_vals, float), sig, EPS_SIG)
    return np.full(np.shape(sig_vals), EPS_G, float)


# closure points of the analytic wedge
sig_max_model = _root(lambda s: float(P_ideal_exact(s, float(eps_for(s)), MU, RHO)
                                      - P_bare_exact(s, MU)),
                      1e-6, float(sig.max()))
sig_star_model = _root(lambda s: float(P_unc_exact(s, MU, RHO)
                                       - P_bare_exact(s, MU)),
                       1e-6, float(sig.max()))
sig_max_data = interp_sign_change(sig, P_id - P_ba)
sig_star_data = interp_sign_change(sig, P_un - P_ba)

p_all = np.array([p for p in p_nl if p > 0.0])
if len(p_all) > NROW*NCOL:
    sel = np.unique(np.linspace(0, len(p_all)-1, NROW*NCOL).round().astype(int))
    p_show = p_all[sel]
else:
    p_show = p_all


def nsrow(p):
    if not HAVE_NS or not np.any(np.isclose(p_ns, p)):
        return None, None
    j = idx(p_ns, p)
    return P_ns[j], S_ns[j]


# ------------------------------------------------------------------- FIG A
figA, axes = plt.subplots(NROW, NCOL, figsize=(5.6*NCOL, 4.3*NROW),
                          squeeze=False, sharex=True, sharey=True)
for k in range(NROW*NCOL):
    ax = axes[k//NCOL][k % NCOL]
    if k >= len(p_show):
        ax.axis('off'); continue
    p = p_show[k]; iz = idx(p_nl, p)
    if SHOW_IDEAL:
        ax.plot(sig, P_id, '-', color=COL['ideal'], lw=1.2, alpha=.85,
                label=LBL['ideal'], zorder=1)
    ax.plot(sig, P_un, '^-', color=COL['uncorr'], lw=2, ms=5,
            label=LBL['uncorr'], zorder=3)
    ax.plot(sig, P_ba, '--', color=COL['bare'], lw=2.4, label=LBL['bare'],
            zorder=4)
    ax.plot(sig, P_nl[iz], 'v--', color=COL['nltrain'], lw=2, ms=5,
            label=LBL['nltrain'], zorder=5)
    if SHOW_BANDS:
        ax.fill_between(sig, P_nl[iz]-S_nl[iz], P_nl[iz]+S_nl[iz],
                        color=COL['nltrain'], alpha=.12, lw=0, zorder=0)
    ps, ss = nsrow(p)
    if ps is not None:
        ax.plot(sig, ps, 's-', color=COL['nstrain'], lw=2, ms=5,
                label=LBL['nstrain'], zorder=6)
        if SHOW_BANDS:
            ax.fill_between(sig, ps-ss, ps+ss, color=COL['nstrain'],
                            alpha=.12, lw=0, zorder=0)
    if SHOW_MODEL_IN_FIGA and np.isfinite(C_MAIN):
        sf = np.linspace(sig.min(), sig.max(), 200)
        ax.plot(sf, P_decay(p, P_ideal_exact(sf, eps_for(sf), MU, RHO),
                            C_MAIN, P_INF),
                ':', color='k', lw=1.6, alpha=.8, label="closed-form model",
                zorder=7)
    ax.axvline(sig_tr, color=COL['train'], ls=':', lw=3, alpha=.45)
    # ax.set_title(rf"$p_{{2q}}={p:.0e}$,  $p_{{1q}}={p*0.1:.0e}$",
    #              fontweight='bold', fontsize=11)
    ax.set_title(rf"$p_{{2q}}={p:.0e}$,  $p_{{1q}}={p*0.1:.0e}$",
                 fontweight='bold', fontsize=15)
    ax.grid(alpha=.3)
    ax.tick_params(axis='both', labelsize=12)
    # ax.legend(fontsize=7.5, loc='lower left', framealpha=.9)
    if k == 0:
        ax.legend(fontsize=12, loc='lower left', framealpha=.9)
# for c in range(NCOL):
#     axes[-1][c].set_xlabel(r"$\sigma_i$")
# for r in range(NROW):
#     axes[r][0].set_ylabel("Fidelity")
for c in range(NCOL):
    axes[-1][c].set_xlabel(r"$\sigma$", fontsize=15)
for r in range(NROW):
    axes[r][0].set_ylabel(r"$\langle F \rangle$", fontsize=15)
# figA.suptitle("Quantum memory with a noisy QAE — [[3,1,3]] phase-flip code\n"
#               r"$\mathcal{N}_{\sigma_i}\to$ QAE $\to\mathcal{N}_{\sigma_i}\to$"
#               rf" perfect QEC  |  $L={L}$ ({nc} CNOT + {nr} rot)",
#               fontsize=13, fontweight='bold')
figA.tight_layout()
# figA.savefig("internal3_noise_grid_vs_sigma.png", dpi=150, bbox_inches='tight')
# print("  Saved : internal3_noise_grid_vs_sigma.png")
figA.savefig("internal3_noise_grid_vs_sigma.pdf", dpi=150, bbox_inches='tight')
print("  Saved : internal3_noise_grid_vs_sigma.pdf")

# ------------------------------------------------------------------- FIG B
pnl, Pnl = p_nl[mnl], P_nl[mnl, i_tr]
ps_log_nl = crossover(pnl, Pnl, P_un[i_tr])
ps_bar_nl = crossover(pnl, Pnl, P_ba[i_tr])
if HAVE_NS:
    pns, Pns = p_ns[mns], P_ns[mns, i_tr]
    ps_log_ns = crossover(pns, Pns, P_un[i_tr])
    ps_bar_ns = crossover(pns, Pns, P_ba[i_tr])
else:
    pns = Pns = None; ps_log_ns = ps_bar_ns = None

e_tr = float(eps_for(sig_tr))
Pid_tr = float(P_ideal_exact(sig_tr, e_tr, MU, RHO))
pc_log_mod = float(p_crit_eq28(Pid_tr, float(P_unc_exact(sig_tr, MU, RHO)),
                               C_MAIN, P_INF)[0])
pc_bar_mod = float(p_crit_eq28(Pid_tr, float(P_bare_exact(sig_tr, MU)),
                               C_MAIN, P_INF)[0])

# FIG B plotting disabled (only Fig A and Fig C are needed); variable
# computations above (pnl, Pnl, ps_log_nl, ..., pc_bar_mod) are still used
# by the report printed later, so they remain active.

ncols = 3 if HAVE_NS else 2
figB, axs = plt.subplots(1, ncols, figsize=(7.0*ncols, 5.2), squeeze=False)
axs = axs[0]

ax = axs[0]
ax.axhline(P_un[i_tr], color=COL['uncorr'], lw=2, label=LBL['uncorr'])
ax.axhline(P_ba[i_tr], color=COL['bare'], ls='--', lw=2, label=LBL['bare'])
ax.axhline(P_id[i_tr], color=COL['ideal'], lw=1.2, label=LBL['ideal'])
ax.plot(pnl, Pnl, 'v--', color=COL['nltrain'], lw=2, ms=7, label=LBL['nltrain'])
if HAVE_NS:
    ax.plot(pns, Pns, 's-', color=COL['nstrain'], lw=2, ms=7, label=LBL['nstrain'])
if ANALYTIC_IN_FIGB and np.isfinite(C_MAIN):
    pf = np.logspace(np.log10(pnl.min()), np.log10(pnl.max()), 300)
    ax.plot(pf, P_decay(pf, Pid_tr, C_MAIN, P_INF), '-', color='k', lw=1.3,
            alpha=.65, label=rf"Eq. (25) model, $C={C_MAIN:.0f}$")
    if np.isfinite(pc_log_mod):
        ax.axvline(pc_log_mod, color=ANA_COL['unc'], lw=1.4, alpha=.9,
                   label="analytic model vs logical")
    if np.isfinite(pc_bar_mod):
        ax.axvline(pc_bar_mod, color=ANA_COL['bare'], lw=1.4, alpha=.9,
                   label="analytic model vs physical")
for pst, c in [(ps_log_nl, COL['nltrain']), (ps_log_ns, COL['nstrain'])]:
    if pst:
        ax.axvline(pst, color=c, ls=':', lw=1.5)
ax.set_xscale('log'); ax.grid(alpha=.3, which='both')
ax.set_xlabel(r"two-qubit gate error $p_{2q}$")
ax.set_ylabel(rf"$\langle P\rangle$ at $\sigma_i={sig_tr:g}$")
ax.set_title(rf"Memory quality vs gate error ($L={L}$)", fontweight='bold')
ax.legend(fontsize=8, loc='lower left')

ax = axs[1]
ax.plot(pnl, Pnl-P_un[i_tr], 'v--', color=COL['nltrain'], lw=2, ms=7,
        label="noiseless trained vs logical")
ax.plot(pnl, Pnl-P_ba[i_tr], 'v:', color=COL['nltrain'], lw=1.4, ms=5,
        alpha=.75, label="noiseless trained vs physical")
if HAVE_NS:
    ax.plot(pns, Pns-P_un[i_tr], 's-', color=COL['nstrain'], lw=2, ms=7,
            label="noisy trained vs logical")
    ax.plot(pns, Pns-P_ba[i_tr], 's:', color=COL['nstrain'], lw=1.4, ms=5,
            alpha=.75, label="noisy trained vs physical")
if ANALYTIC_IN_FIGB:
    if np.isfinite(pc_log_mod):
        ax.axvline(pc_log_mod, color=ANA_COL['unc'], lw=1.2, alpha=.8)
    if np.isfinite(pc_bar_mod):
        ax.axvline(pc_bar_mod, color=ANA_COL['bare'], lw=1.2, alpha=.8)
ax.axhline(0, color='k', lw=1.2); ax.set_xscale('log')
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
ax.set_xlabel(r"$p_{2q}$"); ax.set_ylabel(r"net gain in $\langle P\rangle$")
ax.set_title(rf"Break-even at $\sigma_i={sig_tr:g}$", fontweight='bold')

if HAVE_NS:
    ax = axs[2]
    common = [p for p in pns if np.any(np.isclose(pnl, p))]
    d = np.array([P_ns[idx(p_ns, p), i_tr] - P_nl[idx(p_nl, p), i_tr]
                  for p in common])
    ax.plot(common, d, 'o-', color=COL['nstrain'], lw=2, ms=8)
    ax.axhline(0, color='k', lw=1.2); ax.set_xscale('log')
    ax.grid(alpha=.3, which='both'); ax.set_xlabel(r"$p_{2q}$")
    ax.set_ylabel(r"$P_{\rm noisy\,trained}-P_{\rm noiseless\,trained}$")
    ax.set_title("Benefit of noise-aware training", fontweight='bold')
figB.tight_layout()
figB.savefig("internal3_noise_vs_p2q.png", dpi=150, bbox_inches='tight')
print("  Saved : internal3_noise_vs_p2q.png")

# ------------------------------------------------------------------- FIG C
P2Q_VIEW      = None
VIEW_PAD      = 1.5
NFINE_P       = 400
NFINE_S       = 400
SMOOTH_FILL   = True
SIGMA0_ANCHOR = False          # anchored with the ANALYTIC sigma=0 limits
CMAP_NAME     = 'summer'
CMAP_STOPS    = (0.10, 0.55, 0.92)
XAXIS_MODE    = 'index'        # 'index' | 'linear'
XTICK_FS      = 9
BOUNDARY_LW   = 1.2
STROKE_LW     = 2.2
MARK_CLOSURE  = False
DIAGNOSE      = True
REGIME_LABELS = ['worst', 'median', 'best']

# sigma_i = 0 is NOT P = 1 here: the coherent bias mu survives
sig_x = np.r_[0.0, sig] if SIGMA0_ANCHOR else sig
Pun_x = (np.r_[float(P_unc_exact(0.0, MU, RHO)), P_un] if SIGMA0_ANCHOR else P_un)
Pba_x = (np.r_[float(P_bare_exact(0.0, MU)),     P_ba] if SIGMA0_ANCHOR else P_ba)
Pid_x = (np.r_[float(P_ideal_exact(0.0, EPS_SIG[0], MU, RHO)), P_id]
         if SIGMA0_ANCHOR else P_id)

xpos = (np.arange(len(sig_x), dtype=float) if XAXIS_MODE == 'index'
        else sig_x.astype(float))


def x_of_sigma(s):
    return np.interp(np.asarray(s, float), sig_x, xpos)


def extend_sigma(Pg):
    if not SIGMA0_ANCHOR:
        return Pg
    slope = (Pg[:, 1] - Pg[:, 0])/(sig[1] - sig[0])
    F0 = np.clip(Pg[:, 0] - sig[0]*slope, 0.0, 1.0)
    return np.column_stack([F0, Pg])


def _regime(Pf, un, ba):
    hi = np.maximum(un, ba)[None, :]
    lo = np.minimum(un, ba)[None, :]
    return (Pf > lo).astype(int) + (Pf > hi).astype(int), lo, hi


def data_p_range(pgrids):
    pos = [pg[pg > 0] for pg in pgrids]
    return max(p.min() for p in pos), min(p.max() for p in pos)


def auto_view(pgrids, Pgrids):
    d_lo, d_hi = data_p_range(pgrids)
    lo_all, hi_all = [], []
    for pg, Pg in zip(pgrids, Pgrids):
        m = pg > 0
        R, _, _ = _regime(extend_sigma(Pg[m]), Pun_x, Pba_x)
        chg = np.where(np.any(np.diff(R, axis=0) != 0, axis=1))[0]
        if chg.size:
            pgp = pg[m]
            lo_all.append(pgp[chg.min()])
            hi_all.append(pgp[min(chg.max() + 1, len(pgp) - 1)])
    if not lo_all:
        return d_lo, d_hi
    return max(min(lo_all)/VIEW_PAD, d_lo), min(max(hi_all)*VIEW_PAD, d_hi)


def refine(pg, Pg, view):
    m = pg > 0
    pgp, Pgm = pg[m], extend_sigma(Pg[m])
    lo, hi = max(view[0], pgp.min()), min(view[1], pgp.max())
    pf = np.logspace(np.log10(lo), np.log10(hi), NFINE_P)
    xf = np.linspace(xpos.min(), xpos.max(), NFINE_S)
    tmp = np.array([np.interp(np.log10(pf), np.log10(pgp), Pgm[:, j])
                    for j in range(Pgm.shape[1])]).T
    Pf = np.array([np.interp(xf, xpos, tmp[i]) for i in range(len(pf))])
    return pf, xf, Pf


def analytic_curves(C):
    """Analytic boundaries, x mapped to the drawing coordinate."""
    if not (ANALYTIC_ON_PHASE and np.isfinite(C) and C > 0):
        return []
    sf = np.linspace(max(sig_x.min(), 0.0), sig_x.max(), NFINE_ANA)
    xf = x_of_sigma(sf)
    out = []
    if ANALYTIC_SOURCE in ('model', 'both'):
        Pi_m = P_ideal_exact(sf, eps_for(sf), MU, RHO)
        # out += [(xf, p_crit_eq28(Pi_m, P_unc_exact(sf, MU, RHO), C, P_INF),
        #          ANA_COL['unc'], '-', "analytic model vs logical"),
        #         (xf, p_crit_eq28(Pi_m, P_bare_exact(sf, MU), C, P_INF),
        #          ANA_COL['bare'], '-', "analytic model vs physical")]
        out += [(xf, p_crit_eq28(Pi_m, P_unc_exact(sf, MU, RHO), C, P_INF),
                 ANA_COL['unc'], '-', "Predicted SRL-uncorr"),
                (xf, p_crit_eq28(Pi_m, P_bare_exact(sf, MU), C, P_INF),
                 ANA_COL['bare'], '-', "Predicted SRL-bare")]
    if ANALYTIC_SOURCE in ('data', 'both'):
        Pi_d = np.interp(sf, sig_x, Pid_x)
        Pu_d = np.interp(sf, sig_x, Pun_x)
        Pb_d = np.interp(sf, sig_x, Pba_x)
        out += [(xf, p_crit_eq28(Pi_d, Pu_d, C, P_INF), ANA_COL['unc'], ':',
                 "measured P: vs logical"),
                (xf, p_crit_eq28(Pi_d, Pb_d, C, P_INF), ANA_COL['bare'], ':',
                 "measured P: vs physical")]
    return out


panels = [(p_nl[mnl], P_nl[mnl], "noiseless trained", FIT_NL)]
# if HAVE_NS:
#     panels.append((p_ns[mns], P_ns[mns], "noisy trained", FIT_NS))

view = P2Q_VIEW or auto_view([q[0] for q in panels], [q[1] for q in panels])
print(f"  Fig C p2q view range : {view[0]:.2e} … {view[1]:.2e}")

cmap = ListedColormap(plt.get_cmap(CMAP_NAME)(np.array(CMAP_STOPS)))
norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
stroke = ([mpe.Stroke(linewidth=STROKE_LW, foreground='0.15'), mpe.Normal()]
          if STROKE_LW > 0 else None)
ana_stroke = [mpe.Stroke(linewidth=ANA_STROKE, foreground='w'), mpe.Normal()]


def _apply_stroke(cs):
    if stroke is None:
        return
    try:
        cs.set(path_effects=stroke)
    except AttributeError:
        for coll in cs.collections:
            coll.set_path_effects(stroke)


if DIAGNOSE:
    dbl = P_un - P_ba
    for i0 in np.where(np.diff(np.sign(dbl)) != 0)[0]:
        w = dbl[i0]/(dbl[i0] - dbl[i0+1])
        sc = sig[i0] + w*(sig[i0+1] - sig[i0])
        print(f"  baselines cross at sigma_i = {sc:.4f}  -> solid and dashed "
              f"contours pinch here")

figC, axs = plt.subplots(1, len(panels), figsize=(8.6*len(panels), 5.8),
                         squeeze=False, sharey=True)
axs = axs[0]
im = None
for ax, (pg, Pg, ttl, FITp) in zip(axs, panels):
    C_PAN = (FITp['C'] if (MODEL_USE_PANEL_C and FITp is not None
                           and np.isfinite(FITp['C'])) else C_MAIN)
    pf, xf, Pf = refine(pg, Pg, view)
    un = np.interp(xf, xpos, Pun_x)
    ba = np.interp(xf, xpos, Pba_x)
    R, lo, hi = _regime(Pf, un, ba)

    if SMOOTH_FILL:
        im = ax.contourf(xf, pf, R.astype(float), levels=[-.5, .5, 1.5, 2.5],
                         cmap=cmap, norm=norm)
    else:
        im = ax.pcolormesh(xf, pf, R, cmap=cmap, norm=norm, shading='auto')

    # NUMERIC boundaries from the data
    for fld, ls in [(Pf - lo, '-'), (Pf - hi, '--')]:
        if fld.min() < 0 < fld.max():
            _apply_stroke(ax.contour(xf, pf, fld, levels=[0], colors='w',
                                     linewidths=BOUNDARY_LW, linestyles=ls))

    # ANALYTIC boundaries
    for xs, ys, col, ls, lab in analytic_curves(C_PAN):
        ax.plot(xs, ys, ls=ls, color=col, lw=2.0, zorder=9, label=lab,
                path_effects=ana_stroke)
    if MARK_CLOSURE and np.isfinite(sig_max_model):
        ax.axvline(x_of_sigma(sig_max_model), color=ANA_COL['bare'], ls='-.',
                   lw=1.2, alpha=.95, zorder=8,
                   label=rf"$P_{{\rm ideal}}=P_{{\rm bare}}$ at "
                         rf"$\sigma_i={sig_max_model:.3f}$")
    if ANALYTIC_ON_PHASE and ANALYTIC_LEGEND:
        # ax.legend(fontsize=7, loc='upper left', framealpha=.85)
        ax.legend(fontsize=12, loc='upper left', framealpha=.85)

    ax.set_yscale('log')
    ax.set_ylim(*view)
    ax.set_xlim(xpos.min(), xpos.max())
    ax.set_xticks(xpos)
    ax.set_xticks([], minor=True)
    # ax.set_xticklabels([f"{s:g}" for s in sig_x], fontsize=XTICK_FS)
    # ax.set_xlabel(r"$\sigma_i$")
    # ax.set_title(ttl + (rf"   ($C_{{\rm fit}}={C_PAN:.0f}$)"
    #                     if np.isfinite(C_PAN) else ""), fontweight='bold')
    ax.set_xticklabels([f"{s:g}" for s in sig_x], fontsize=12)
    ax.set_xlabel(r"$\sigma$", fontsize=15)
    ax.tick_params(axis='y', labelsize=12)
    #ax.axvline(x_of_sigma(sig_tr), color='w', ls=':', lw=3, alpha=.85,
    #           zorder=10)

# axs[0].set_ylabel(r"$p_{2q}$")
axs[0].set_ylabel(r"$p_{2q}$", fontsize=15)
cb = figC.colorbar(im, ax=axs, ticks=[0, 1, 2], pad=.02)
# cb.ax.set_yticklabels(REGIME_LABELS)
cb.ax.set_yticklabels(REGIME_LABELS, fontsize=12)
#figC.suptitle(rf"[[3,1,3]] phase-flip code   ($\mu={MU:.3f}$, "
#              rf"$\rho_c={RHO:.2f}$, $P_\infty={P_INF:g}$)",
#              fontsize=13, fontweight='bold')
# figC.suptitle(rf"[[3,1,3]] phase-flip code",
#               fontsize=13, fontweight='bold')
# figC.savefig("internal3_noise_phase_diagram.png", dpi=150, bbox_inches='tight')
# print("  Saved : internal3_noise_phase_diagram.png")
figC.savefig("internal3_noise_phase_diagram.pdf", dpi=150, bbox_inches='tight')
print("  Saved : internal3_noise_phase_diagram.pdf")

# ------------------------------------------------------------------- CSV (data)
with open("internal3_noise_merged.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "p2q", "sigma_i", "P_bare", "P_uncorr_logical",
                "P_ideal_qae", "P_noiseless_trained", "P_noisy_trained",
                "gain_nl_vs_logical", "gain_ns_vs_logical",
                "gain_nl_vs_physical", "gain_ns_vs_physical",
                "benefit_noisy_training"])
    for ip, p in enumerate(p_nl):
        if p <= 0:
            continue
        j = (idx(p_ns, p) if (HAVE_NS and np.any(np.isclose(p_ns, p))) else None)
        for i, sg in enumerate(sig):
            ps = P_ns[j, i] if j is not None else None
            w.writerow([L, f"{p:.6e}", f"{sg:.4f}", f"{P_ba[i]:.6f}",
                        f"{P_un[i]:.6f}", f"{P_id[i]:.6f}", f"{P_nl[ip,i]:.6f}",
                        "" if ps is None else f"{ps:.6f}",
                        f"{P_nl[ip,i]-P_un[i]:.6f}",
                        "" if ps is None else f"{ps-P_un[i]:.6f}",
                        f"{P_nl[ip,i]-P_ba[i]:.6f}",
                        "" if ps is None else f"{ps-P_ba[i]:.6f}",
                        "" if ps is None else f"{ps-P_nl[ip,i]:.6f}"])
print("  Saved : internal3_noise_merged.csv")

# ---------------- CSV: numeric vs analytic break-even ---------------
num_unc = np.array([crossover(p_nl[mnl], P_nl[mnl, i], P_un[i]) or np.nan
                    for i in range(len(sig))], float)
num_bar = np.array([crossover(p_nl[mnl], P_nl[mnl, i], P_ba[i]) or np.nan
                    for i in range(len(sig))], float)
if HAVE_NS:
    num_unc_ns = np.array([crossover(p_ns[mns], P_ns[mns, i], P_un[i]) or np.nan
                           for i in range(len(sig))], float)
    num_bar_ns = np.array([crossover(p_ns[mns], P_ns[mns, i], P_ba[i]) or np.nan
                           for i in range(len(sig))], float)
else:
    num_unc_ns = num_bar_ns = np.full(len(sig), np.nan)

dat_unc = p_crit_eq28(P_id, P_un, C_MAIN, P_INF)
dat_bar = p_crit_eq28(P_id, P_ba, C_MAIN, P_INF)
eps_grid = eps_for(sig)
mod_unc = p_crit_eq28(P_ideal_exact(sig, eps_grid, MU, RHO),
                      P_unc_exact(sig, MU, RHO), C_MAIN, P_INF)
mod_bar = p_crit_eq28(P_ideal_exact(sig, eps_grid, MU, RHO),
                      P_bare_exact(sig, MU), C_MAIN, P_INF)

with open("internal3_breakeven_eq28.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sigma_i", "mu", "rho_c", "C_fit", "epsilon_used", "P_inf",
                "P_ideal_exact", "P_uncorr_exact", "P_bare_exact",
                "p2q_crit_num_nl_vs_logical",
                "p2q_crit_eq28_measured_vs_logical",
                "p2q_crit_eq28_closedform_vs_logical",
                "p2q_crit_num_nl_vs_physical",
                "p2q_crit_eq28_measured_vs_physical",
                "p2q_crit_eq28_closedform_vs_physical",
                "p2q_crit_num_ns_vs_logical", "p2q_crit_num_ns_vs_physical"])

    def s(v):
        return "" if not np.isfinite(v) else f"{v:.6e}"

    for i, sg in enumerate(sig):
        w.writerow([f"{sg:.4f}", f"{MU:.6f}", f"{RHO:.4f}", f"{C_MAIN:.2f}",
                    f"{eps_grid[i]:.4f}", f"{P_INF:.4f}",
                    f"{float(P_ideal_exact(sg, eps_grid[i], MU, RHO)):.6f}",
                    f"{float(P_unc_exact(sg, MU, RHO)):.6f}",
                    f"{float(P_bare_exact(sg, MU)):.6f}",
                    s(num_unc[i]), s(dat_unc[i]), s(mod_unc[i]),
                    s(num_bar[i]), s(dat_bar[i]), s(mod_bar[i]),
                    s(num_unc_ns[i]), s(num_bar_ns[i])])
print("  Saved : internal3_breakeven_eq28.csv")

# ------------------------------------------------------------------- summary
print("\n" + "="*104)
print(f"  SUMMARY at sigma_i = {sig_tr:g}   (L={L}, {nc} CNOT + {nr} rot)")
print("="*104)
print(f"  P_bare = {P_ba[i_tr]:.6f}   P_uncorr(logical) = {P_un[i_tr]:.6f}"
      f"   P_ideal_QAE = {P_id[i_tr]:.6f}")
print(f"  {'p2q':>9} {'P_nlTrain':>10} {'P_nsTrain':>10} {'gain_nl':>9} "
      f"{'gain_ns':>9} {'benefit':>9} {'ns-bare':>9}")
for ip, p in enumerate(p_nl):
    if p <= 0:
        continue
    j = idx(p_ns, p) if (HAVE_NS and np.any(np.isclose(p_ns, p))) else None
    pn = P_nl[ip, i_tr]
    if j is None:
        print(f"  {p:>9.1e} {pn:>10.5f} {'--':>10} {pn-P_un[i_tr]:>+9.5f} "
              f"{'--':>9} {'--':>9} {'--':>9}")
    else:
        pv = P_ns[j, i_tr]
        print(f"  {p:>9.1e} {pn:>10.5f} {pv:>10.5f} {pn-P_un[i_tr]:>+9.5f} "
              f"{pv-P_un[i_tr]:>+9.5f} {pv-pn:>+9.5f} {pv-P_ba[i_tr]:>+9.5f}")
print(f"\n  p* vs logical  : noiseless "
      f"{f'{ps_log_nl:.3e}' if ps_log_nl else 'not crossed'}, noisy "
      f"{f'{ps_log_ns:.3e}' if ps_log_ns else ('not crossed' if HAVE_NS else 'n/a')}"
      f"   |  closed form {pc_log_mod:.3e}")
print(f"  p* vs physical : noiseless "
      f"{f'{ps_bar_nl:.3e}' if ps_bar_nl else 'not crossed'}, noisy "
      f"{f'{ps_bar_ns:.3e}' if ps_bar_ns else ('not crossed' if HAVE_NS else 'n/a')}"
      f"   |  closed form {pc_bar_mod:.3e}")

# ------------------------- optional calibration report ----------------------
if SHOW_FIT_REPORT:
    print("\n" + "="*104)
    print("  CALIBRATION OF THE CLOSED-FORM MODEL")
    print("="*104)
    print(f"  mu from P_bare      : {MU_FIT:.6f}   (used {MU:.6f}, "
          f"source '{MU_SOURCE}')")
    print("   cos(2mu) per sigma : " + ", ".join(f"{c:.6f}" for c in COS2MU))
    print(f"  rho_c from P_uncorr : {RHO_FIT:.4f}  (rms {RHO_RMS:.2e}, "
          f"used {RHO:.4f}, source '{RHO_SOURCE}')")
    print(f"  P_inf               : {P_INF:.4f}")
    print(f"  C_fit (noiseless)   : {FIT_NL['C']:.1f}  (log-slope "
          f"{FIT_NL['C_lin']:.1f}, P-space {FIT_NL['C_nl']:.1f}, rms "
          f"{FIT_NL['rms']:.2e}, {FIT_NL['n_used']} unsaturated pts)")
    if FIT_NS is not None:
        print(f"  C_fit (noisy tr.)   : {FIT_NS['C']:.1f}  (rms "
              f"{FIT_NS['rms']:.2e})")
    print("  C per sigma_i       : " + ", ".join(
        f"{s0:g}:{c0:.0f}" for s0, c0 in zip(sig, FIT_NL['C_per_sig'])
        if np.isfinite(c0)))
    if np.isfinite(FIT_NL['C_per_sig']).sum() > 1:
        cv = np.nanstd(FIT_NL['C_per_sig'])/np.nanmean(FIT_NL['C_per_sig'])
        print(f"  column scatter      : {100*cv:.1f} %  "
              f"(<10 % validates the single-C exponential)")
    print(f"  epsilon (global)    : {EPS_G:.4f} (rms in P {EPS_RMS:.2e}, "
          f"mode '{EPS_MODE}')")
    print("  epsilon per sigma   : " + ", ".join(
        f"{s0:g}:{e0:.3f}" for s0, e0 in zip(sig, EPS_SIG)))
    print(f"  sigma^max (P_ideal = P_bare) : model {sig_max_model:.4f} | data "
          f"{sig_max_data:.4f}")
    print(f"  sigma*    (P_unc  = P_bare)  : model {sig_star_model:.4f} | data "
          f"{sig_star_data:.4f}")
    print("\n  closed-form baselines vs measured")
    print(f"  {'sigma':>7} {'P_ba_ex':>10} {'P_ba_num':>10} {'dev':>9} "
          f"{'P_un_ex':>10} {'P_un_num':>10} {'dev':>9} {'P_id_ex':>10} "
          f"{'P_id_num':>10}")
    for i, sg in enumerate(sig):
        print(f"  {sg:>7.3f} {P_ba_x[i]:>10.6f} {P_ba[i]:>10.6f} "
              f"{P_ba_x[i]-P_ba[i]:>+9.2e} {P_un_x[i]:>10.6f} {P_un[i]:>10.6f} "
              f"{P_un_x[i]-P_un[i]:>+9.2e} {P_id_x[i]:>10.6f} {P_id[i]:>10.6f}")
    print("\n  break-even p2q: numeric vs analytic")
    print(f"  {'sigma':>7} {'num|log':>11} {'meas|log':>11} {'form|log':>11} "
          f"{'num|bare':>11} {'meas|bare':>11} {'form|bare':>11}")

    def _fv(v):
        return f"{v:11.3e}" if np.isfinite(v) else f"{'--':>11}"

    for i, sg in enumerate(sig):
        print(f"  {sg:>7.3f} {_fv(num_unc[i])} {_fv(dat_unc[i])} "
              f"{_fv(mod_unc[i])} {_fv(num_bar[i])} {_fv(dat_bar[i])} "
              f"{_fv(mod_bar[i])}")

print("\n  Done")