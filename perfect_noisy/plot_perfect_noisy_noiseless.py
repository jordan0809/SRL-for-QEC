# ============================================================================
# plot_perfect_noisy_noiseless.py -- memory-protocol figures for the [[5,1,3]] perfect code
#
#   FIG A : 3x3 grid, one panel per p2q, x = p_i
#   FIG B : P vs p2q at p_i = PEXT_TRAIN (+ gain, + benefit of noisy training)
#   FIG C : (p_i, p2q) 3-regime phase diagram.  The white contours are the
#           NUMERIC boundaries from the data.  On top of them we
#           draw the ANALYTIC break-even curves
# ============================================================================
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe          # 'pe' is used as a loop var below
from matplotlib.colors import ListedColormap, BoundaryNorm

C_NL, C_NS = "internal5_noiseless_trained.npz", "internal5_noisy_trained.npz"
NCOL, NROW = 3, 3
SHOW_BANDS = True
SHOW_IDEAL = True

COL = dict(uncorr='#e41a1c', bare='#ff7f00', nltrain='#4daf4a',
           nstrain='#377eb8', ideal='#4d4d4d')
# LBL = dict(uncorr="No QAE (logical)",
#            bare="Bare physical qubit",
#            nltrain="Noiseless trained QAE", nstrain="Noisy trained QAE",
#            ideal=r"ideal QAE ($p_{2q}=0$)")
LBL = dict(uncorr="Uncorrected logical",
           bare="Bare physical",
           nltrain="Noiseless-trained SRL", nstrain="Noisy-trained SRL",
           ideal=r"ideal SRL ($p_{2q}=0$)")

# ============================ ANALYTIC-MODEL SWITCHES =======================
SHOW_FIT_REPORT   = False     # <-- C, eta, epsilon, alpha, beta, gamma, ...
ANALYTIC_ON_PHASE = True      # overlay analytic curves on FIG C
ANALYTIC_SOURCE   = 'model'   # 'model' (closed form) | 'data' | 'both'
ANALYTIC_IN_FIGB  = True      # decay law + thresholds in FIG B
SHOW_MODEL_IN_FIGA = False    # closed-form P(p_i) per panel in FIG A
ANALYTIC_LEGEND   = True
MARK_CLOSURE      = False      # vertical marks at p_i^max and p_i^*

P_INF_MODE   = 'fixed'        # 'fixed' -> P_INF_FIXED ; 'auto' -> plateau
P_INF_FIXED  = 0.5            # maximally mixed logical qubit (depolarizing)
C_SOURCE     = 'fit'          # 'fit' -> fit Eq. (25) ; 'file' -> stored c_eff
C_FIT_MODE   = 'nonlinear'    # 'nonlinear' (P-space LSQ) | 'linear' (log slope)
C_FIT_CPMAX  = 1.0            # drop saturated points with C*p2q > CPMAX
C_FIT_FLOOR  = 2e-3           # need P - P_inf > floor on both sides of the log
MODEL_USE_PANEL_C = False     # False -> one C (noiseless fit) in both panels

EPS_MODE     = 'per_pi'       # 'global' | 'per_pi' (global: all p_i share one eps, per_pi: each p_i has its own eps (peicewise linear))
EPS_FIT_PMAX = np.inf         # only fit eps on p_i <= this
EPS_MAX      = 3.0            # bracket for the eps root/fit
N_FAIL2      = 90             # uncorrectable weight-2 Paulis of [[5,1,3]]
N_QUAD_FIT   = 4              # # of smallest p_i used for the p_i^2 diagnostics
NFINE_ANA    = 1200           # resolution of the analytic curves
ANA_COL      = dict(unc='#ff00ff', bare='#00e5ff')
ANA_STROKE   = 3.2            # white halo width of the analytic curves

N_CODE = 5
ENUM5  = np.array([1, 15, 0, 60, 135, 45], dtype=float)   # sum = 256
assert ENUM5.sum() == 256


def idx(a, v): return int(np.argmin(np.abs(np.asarray(a) - v)))


# ============================================================================
# 1. EXACT analytic reference curves (no expansion in p_i)
# ============================================================================
def lam_of_p(p):
    """Bloch shrink factor of a single-qubit depolarizing channel."""
    return 1.0 - 4.0*np.asarray(p, float)/3.0


def p_of_lam(l):
    return 0.75*(1.0 - np.asarray(l, float))


def compose_depol(*ps):
    """Exact composition of single-qubit depolarizing channels."""
    lam = 1.0
    for p in ps:
        lam = lam*lam_of_p(p)
    return p_of_lam(lam)


def p_eff_two(p):
    """Two statistically independent half intervals (exact, = 2p - 4p^2/3)."""
    return compose_depol(p, p)


def P_bare_exact(p):
    """1 - (2/3) p_eff."""
    return 1.0 - (2.0/3.0)*p_eff_two(p)


def P_from_enum(q, enum=ENUM5, n=N_CODE):
    """EXACT <P> after perfect final QEC for i.i.d. depolarizing rate q.

    Every Pauli outside the correctable set is a logical X/Y/Z, each worth 1/3
    of the Bloch average -> P = (1 + 2 P_I)/3."""
    q = np.asarray(q, float)
    a, b = 1.0 - q, q/3.0
    pI = np.zeros_like(a)
    for w, Aw in enumerate(np.asarray(enum, float)):
        if Aw:
            pI = pI + Aw*a**(n - w)*b**w
    return (1.0 + 2.0*pI)/3.0


def P_unc_exact(p):
    """Encoded, NO mid-storage recovery, perfect final QEC."""
    return P_from_enum(p_eff_two(p))


def P_ideal_exact(p, eps):
    """Noiseless QAE: recovery leaves a fraction eps of the first interval.
    eps = 0 -> perfect recovery ; eps = 1 -> exactly P_unc_exact."""
    return P_from_enum(compose_depol(np.asarray(eps, float)*np.asarray(p, float),
                                     p))


def P_decay(p2q, P_ideal, C, P_inf=P_INF_FIXED):
    """the logical polarisation decays exponentially in p2q."""
    return P_inf + np.exp(-C*np.asarray(p2q, float))*(np.asarray(P_ideal) - P_inf)


def p_crit_eq28(P_ideal, P_ref, C, P_inf=P_INF_FIXED):
    """NaN wherever no positive internal-noise budget exists."""
    num = np.atleast_1d(np.asarray(P_ideal, float)) - P_inf
    den = np.atleast_1d(np.asarray(P_ref,   float)) - P_inf
    num, den = np.broadcast_arrays(num, den)
    out = np.full(num.shape, np.nan)
    if not (np.isfinite(C) and C > 0):
        return out
    ok = (num > 0) & (den > 0) & (num > den)
    out[ok] = np.log(num[ok]/den[ok])/C
    return out


def _root(f, a, b, tol=1e-13, itmax=300):
    """bisection; NaN if f does not change sign on [a,b]."""
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


def _golden_min(f, a, b, tol=1e-10, itmax=400):
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


# ------------------------- calibration of the model -------------------------
def eps_at(pi, P_target, lo=0.0, hi=EPS_MAX):
    """Solve P_ideal_exact(pi, eps) = P_target for eps (monotone decreasing)."""
    f = lambda e: float(P_ideal_exact(pi, e)) - float(P_target)
    if f(lo) <= 0.0:
        return lo
    if f(hi) >= 0.0:
        return hi
    return _root(f, lo, hi)


def fit_eps(pi, P_id, pmax=EPS_FIT_PMAX):
    """One global eps by least squares on the exact model."""
    pi = np.asarray(pi, float); P_id = np.asarray(P_id, float)
    m = pi <= pmax
    if m.sum() == 0:
        m = np.ones_like(pi, bool)
    sse = lambda e: float(np.sum((P_ideal_exact(pi[m], e) - P_id[m])**2))
    e = _golden_min(sse, 0.0, EPS_MAX)
    return float(np.clip(e, 0.0, EPS_MAX)), float(np.sqrt(sse(e)/m.sum()))


def estimate_Pinf(p2q, P, frac=0.25):
    """Empirical saturation floor: mean P over the largest-p2q rows."""
    m = np.asarray(p2q) > 0
    pv, Pv = np.asarray(p2q)[m], np.asarray(P)[m]
    k = max(1, int(np.ceil(frac*len(pv))))
    return float(np.mean(Pv[np.argsort(pv)[-k:]]))


def fit_C(p2q, P, P_ideal, P_inf=P_INF_FIXED, cp_max=C_FIT_CPMAX,
          floor=C_FIT_FLOOR, mode=C_FIT_MODE, iters=6):
    """Fit  P = P_inf + exp(-C p2q)(P_ideal - P_inf)  over the whole grid.

    Iteratively re-weighted log-slope fit through the origin; points with
    C*p2q > cp_max are SATURATED (P has hit the floor) and carry no slope
    information, so they are dropped.  P-space refinement on the same mask."""
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
        return dict(C=np.nan, C_lin=np.nan, C_nl=np.nan, rms=np.nan,
                    n_used=0, C_per_pi=np.full(P.shape[1], np.nan))
    Y = np.zeros_like(num); Y[ok] = -np.log(num[ok]/den[ok])
    w = ok.astype(float)
    C_lin = float(np.sum(w*X*Y)/np.sum(w*X*X))
    for _ in range(iters):
        w_new = np.where(ok & (C_lin*X < cp_max), 1.0, 0.0)
        if w_new.sum() < 2:
            break
        C_new = float(np.sum(w_new*X*Y)/np.sum(w_new*X*X))
        if abs(C_new - C_lin) < 1e-9*max(1.0, abs(C_lin)):
            C_lin = C_new; w = w_new; break
        C_lin, w = C_new, w_new
    use = w > 0
    sse = lambda C: float(np.sum((P_inf + np.exp(-C*X[use])*den[use]
                                  - P[use])**2))
    C_nl = _golden_min(sse, max(1e-3, 0.2*C_lin), 5.0*C_lin)
    C = C_nl if mode == 'nonlinear' else C_lin
    rms = float(np.sqrt(sse(C)/max(1, int(use.sum()))))
    C_pi = np.full(P.shape[1], np.nan)
    for j in range(P.shape[1]):
        mj = use[:, j]
        if mj.sum() >= 2:
            C_pi[j] = float(np.sum(X[mj, j]*Y[mj, j])/np.sum(X[mj, j]**2))
    return dict(C=C, C_lin=C_lin, C_nl=C_nl, rms=rms, n_used=int(use.sum()),
                C_per_pi=C_pi)


def fit_quad_coeffs(pi, P_id, P_un, P_ba, nq=N_QUAD_FIT):
    """alpha, beta, gamma of Sec. 1.8.6 (diagnostics only, small-p_i window)."""
    pi = np.asarray(pi, float)
    o = np.argsort(pi)[:max(2, min(int(nq), pi.size))]
    p = pi[o]
    quad = lambda y: float(np.sum(p**2*np.asarray(y, float)[o])/np.sum(p**4))
    alpha = quad(np.asarray(P_id) - np.asarray(P_un))
    gamma = quad(1.0 - np.asarray(P_un))
    beta = quad((4.0/3.0)*pi - (np.asarray(P_id) - np.asarray(P_ba)))
    return dict(alpha=alpha, beta=beta, gamma=gamma, p_fit=p)


def crossover(pv, fv, thr):
    """numeric break-even in log p2q (first downward crossing of thr)."""
    pv = np.asarray(pv, float); fv = np.asarray(fv, float)
    o = np.argsort(pv); pv, fv = pv[o], fv[o]
    for i in range(len(pv)-1):
        a, b = fv[i]-thr, fv[i+1]-thr
        if a > 0 >= b and pv[i] > 0:
            w = a/(a-b)
            return 10**(np.log10(pv[i]) + w*(np.log10(pv[i+1])-np.log10(pv[i])))
    return None


def interp_sign_change(x, y):
    """first x where y changes sign (linear interpolation), else NaN."""
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
ds = np.load(C_NS, allow_pickle=True)

pi_g = np.asarray(dn['p_ext'], float)
assert np.allclose(pi_g, np.asarray(ds['p_ext'])), "p_i grid mismatch"
p_nl = np.asarray(dn['p2q'], float);  p_ns = np.asarray(ds['p2q'], float)
P_nl = np.asarray(dn['p_corr']);      S_nl = np.asarray(dn['p_corr_std'])
P_ns = np.asarray(ds['p_aware']);     S_ns = np.asarray(ds['p_aware_std'])
P_un = np.asarray(dn['p_uncorr']).ravel()
P_ba = np.asarray(dn['p_bare']).ravel()
P_id = np.asarray(dn['p_ideal_qae']).ravel()

L      = int(dn['n_layers'])
nc, nr = int(dn['n_cnot_full']), int(dn['n_rot_full'])
c_eff  = float(dn['c_eff'])
pi_tr  = float(ds['pext_train']); i_tr = idx(pi_g, pi_tr)

print(f"  p_i grid ({len(pi_g)}): {np.round(pi_g,3)}")
print(f"  L={L}, {nc} CNOT + {nr} rot, C_eff={c_eff:.0f}, p_i_train={pi_tr}")

mnl = p_nl > 0; mns = p_ns > 0

# ============================================================================
# 3. calibrate the model
# ============================================================================
P_inf_auto = estimate_Pinf(p_nl, P_nl)
P_INF = P_INF_FIXED if P_INF_MODE == 'fixed' else P_inf_auto

C_COUNT = nc + nr/10.0                        
FIT_NL = fit_C(p_nl, P_nl, P_id, P_INF)
FIT_NS = fit_C(p_ns, P_ns, P_id, P_INF)
FIT_NL['eta'] = FIT_NL['C']/C_COUNT
FIT_NS['eta'] = FIT_NS['C']/C_COUNT
C_MAIN = FIT_NL['C'] if (C_SOURCE == 'fit' and np.isfinite(FIT_NL['C'])) else c_eff

EPS_G, EPS_RMS = fit_eps(pi_g, P_id)
EPS_PI = np.array([eps_at(pv, Pv) for pv, Pv in zip(pi_g, P_id)])
FQ = fit_quad_coeffs(pi_g, P_id, P_un, P_ba)

# exact closed-form baselines on the measured grid (cross-check of the data)
P_un_x = P_unc_exact(pi_g)
P_ba_x = P_bare_exact(pi_g)
P_id_x = P_ideal_exact(pi_g, EPS_G)

# structural points of the wedge (closed form and from the data)
pi_max_model = _root(lambda p: float(P_ideal_exact(p, EPS_G) - P_bare_exact(p)),
                     1e-6, 0.6)
pi_star_model = _root(lambda p: float(P_unc_exact(p) - P_bare_exact(p)),
                      1e-6, 0.6)
pi_max_data  = interp_sign_change(pi_g, P_id - P_ba)
pi_star_data = interp_sign_change(pi_g, P_un - P_ba)


def eps_for(pi_vals):
    """eps used by the model at the requested p_i values."""
    if EPS_MODE == 'per_pi':
        return np.interp(np.asarray(pi_vals, float), pi_g, EPS_PI)
    return np.full(np.shape(pi_vals), EPS_G, float)


p_all = np.array([p for p in p_nl if p > 0.0])
if len(p_all) > NROW*NCOL:
    sel = np.unique(np.linspace(0, len(p_all)-1, NROW*NCOL).round().astype(int))
    p_show = p_all[sel]
else:
    p_show = p_all


def nsrow(p):
    if not np.any(np.isclose(p_ns, p)): return None, None
    j = idx(p_ns, p); return P_ns[j], S_ns[j]


# ------------------------------------------------------------------- FIG A
figA, axes = plt.subplots(NROW, NCOL, figsize=(5.6*NCOL, 4.3*NROW),
                          squeeze=False, sharex=True, sharey=True)
for k in range(NROW*NCOL):
    ax = axes[k//NCOL][k % NCOL]
    if k >= len(p_show): ax.axis('off'); continue
    p = p_show[k]; iz = idx(p_nl, p)
    if SHOW_IDEAL:
        ax.plot(pi_g, P_id, '-', color=COL['ideal'], lw=1.2, alpha=.85,
                label=LBL['ideal'], zorder=1)
    ax.plot(pi_g, P_un, '^-', color=COL['uncorr'], lw=2, ms=5,
            label=LBL['uncorr'], zorder=3)
    ax.plot(pi_g, P_ba, '--', color=COL['bare'], lw=2.4, label=LBL['bare'],
            zorder=4)
    ax.plot(pi_g, P_nl[iz], 'v--', color=COL['nltrain'], lw=2, ms=5,
            label=LBL['nltrain'], zorder=5)
    if SHOW_BANDS:
        ax.fill_between(pi_g, P_nl[iz]-S_nl[iz], P_nl[iz]+S_nl[iz],
                        color=COL['nltrain'], alpha=.12, lw=0, zorder=0)
    ps, ss = nsrow(p)
    if ps is not None:
        ax.plot(pi_g, ps, 's-', color=COL['nstrain'], lw=2, ms=5,
                label=LBL['nstrain'], zorder=6)
        if SHOW_BANDS:
            ax.fill_between(pi_g, ps-ss, ps+ss, color=COL['nstrain'],
                            alpha=.12, lw=0, zorder=0)
    if SHOW_MODEL_IN_FIGA and np.isfinite(C_MAIN):
        pfi = np.linspace(max(pi_g.min(), 1e-6), pi_g.max(), 200)
        ax.plot(pfi, P_decay(p, P_ideal_exact(pfi, eps_for(pfi)), C_MAIN, P_INF),
                ':', color='k', lw=1.6, alpha=.8, label="closed-form model",
                zorder=7)
    ax.axvline(pi_tr, color='k', ls=':', lw=3, alpha=.45)
    # ax.set_title(rf"$p_{{2q}}={p:.0e}$,  $p_{{1q}}={p*0.1:.0e}$",
    #              fontweight='bold', fontsize=11)
    # ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc='lower left', framealpha=.9)
    ax.set_title(rf"$p_{{2q}}={p:.0e}$,  $p_{{1q}}={p*0.1:.0e}$",
                 fontweight='bold', fontsize=15)
    ax.grid(alpha=.3)
    ax.tick_params(axis='both', labelsize=12)
    if k == 0:
        ax.legend(fontsize=12, loc='lower left', framealpha=.9)
# for c in range(NCOL):
#     axes[-1][c].set_xlabel(r"$p_i$")
# for r in range(NROW):
#     axes[r][0].set_ylabel("Fidelity")
for c in range(NCOL):
    axes[-1][c].set_xlabel(r"$p_{\mathrm{ext}}$", fontsize=15)
for r in range(NROW):
    axes[r][0].set_ylabel(r"$\langle F \rangle$", fontsize=15)
# figA.suptitle("Quantum memory with a noisy QAE — [[5,1,3]] perfect code\n"
#               r"$\mathcal{N}_{p_i}\to$ QAE $\to\mathcal{N}_{p_i}\to$ perfect QEC"
#               rf"  |  $L={L}$ ({nc} CNOT + {nr} rot)",
#               fontsize=13, fontweight='bold')
figA.tight_layout()
# figA.savefig("internal5_noise_grid_vs_pi.png", dpi=150, bbox_inches='tight')
# print("  Saved : internal5_noise_grid_vs_pi.png")
figA.savefig("internal5_noise_grid_vs_pi.pdf", dpi=150, bbox_inches='tight')
print("  Saved : internal5_noise_grid_vs_pi.pdf")

# ------------------------------------------------------------------- FIG B
pnl, Pnl = p_nl[mnl], P_nl[mnl, i_tr]
pns, Pns = p_ns[mns], P_ns[mns, i_tr]
ps_log_nl = crossover(pnl, Pnl, P_un[i_tr]); ps_log_ns = crossover(pns, Pns, P_un[i_tr])
ps_bar_nl = crossover(pnl, Pnl, P_ba[i_tr]); ps_bar_ns = crossover(pns, Pns, P_ba[i_tr])

e_tr = float(eps_for(pi_tr))
pc_log_mod = float(p_crit_eq28(P_ideal_exact(pi_tr, e_tr),
                               P_unc_exact(pi_tr), C_MAIN, P_INF)[0])
pc_bar_mod = float(p_crit_eq28(P_ideal_exact(pi_tr, e_tr),
                               P_bare_exact(pi_tr), C_MAIN, P_INF)[0])

# FIG B 
figB, axs = plt.subplots(1, 3, figsize=(7.0*3, 5.2))
ax = axs[0]
ax.axhline(P_un[i_tr], color=COL['uncorr'], lw=2, label=LBL['uncorr'])
ax.axhline(P_ba[i_tr], color=COL['bare'], ls='--', lw=2, label=LBL['bare'])
ax.axhline(P_id[i_tr], color=COL['ideal'], lw=1.2, label=LBL['ideal'])
ax.plot(pnl, Pnl, 'v--', color=COL['nltrain'], lw=2, ms=7, label=LBL['nltrain'])
ax.plot(pns, Pns, 's-',  color=COL['nstrain'], lw=2, ms=7, label=LBL['nstrain'])
if ANALYTIC_IN_FIGB and np.isfinite(C_MAIN):
    pf = np.logspace(np.log10(pnl.min()), np.log10(pnl.max()), 300)
    ax.plot(pf, P_decay(pf, P_ideal_exact(pi_tr, e_tr), C_MAIN, P_INF), '-',
            color='k', lw=1.3, alpha=.65,
            label=rf"$C={C_MAIN:.0f}$")
    if np.isfinite(pc_log_mod):
        ax.axvline(pc_log_mod, color=ANA_COL['unc'], ls='-', lw=1.4, alpha=.9,
                   label="analytic model vs logical")
    if np.isfinite(pc_bar_mod):
        ax.axvline(pc_bar_mod, color=ANA_COL['bare'], ls='-', lw=1.4, alpha=.9,
                   label="analytic model vs physical")
for pst, c in [(ps_log_nl, COL['nltrain']), (ps_log_ns, COL['nstrain'])]:
    if pst: ax.axvline(pst, color=c, ls=':', lw=1.5)
ax.set_xscale('log'); ax.grid(alpha=.3, which='both')
ax.set_xlabel(r"two-qubit gate error $p_{2q}$")
ax.set_ylabel(rf"$\langle P\rangle$ at $p_i={pi_tr:g}$")
ax.set_title(rf"Memory quality vs gate error ($L={L}$)", fontweight='bold')
ax.legend(fontsize=8, loc='lower left')

ax = axs[1]
ax.plot(pnl, Pnl-P_un[i_tr], 'v--', color=COL['nltrain'], lw=2, ms=7,
        label="noiseless vs logical")
ax.plot(pns, Pns-P_un[i_tr], 's-', color=COL['nstrain'], lw=2, ms=7,
        label="noisy vs logical")
ax.plot(pnl, Pnl-P_ba[i_tr], 'v:', color=COL['nltrain'], lw=1.4, ms=5, alpha=.7,
        label="noiseless vs physical")
ax.plot(pns, Pns-P_ba[i_tr], 's:', color=COL['nstrain'], lw=1.4, ms=5, alpha=.7,
        label="noisy vs physical")
ax.axhline(0, color='k', lw=1.2); ax.set_xscale('log')
ax.grid(alpha=.3, which='both'); ax.legend(fontsize=8)
ax.set_xlabel(r"$p_{2q}$"); ax.set_ylabel(r"net gain in $\langle P\rangle$")
ax.set_title(rf"Break-even at $p_i={pi_tr:g}$", fontweight='bold')

ax = axs[2]
common = [p for p in pns if np.any(np.isclose(pnl, p))]
d = np.array([P_ns[idx(p_ns, p), i_tr] - P_nl[idx(p_nl, p), i_tr] for p in common])
ax.plot(common, d, 'o-', color=COL['nstrain'], lw=2, ms=8)
ax.axhline(0, color='k', lw=1.2); ax.set_xscale('log')
ax.grid(alpha=.3, which='both')
ax.set_xlabel(r"$p_{2q}$")
ax.set_ylabel(r"$P_{\rm noisy\,trained}-P_{\rm noiseless\,trained}$")
ax.set_title("Benefit of noise-aware training", fontweight='bold')
figB.tight_layout()
figB.savefig("internal5_noise_vs_p2q.png", dpi=150, bbox_inches='tight')
print("  Saved : internal5_noise_vs_p2q.png")

# ------------------------------------------------------------------- FIG C
# ------------------------------ switches -----------------------------
P2Q_VIEW    = None            # (lo, hi) manual window; None -> auto, data-clipped
VIEW_PAD    = 1.5             # padding factor, never beyond the data range
NFINE_P     = 400             # resampling resolution along p2q
NFINE_S     = 400             # resampling resolution along p_i
SMOOTH_FILL = True            # contourf (smooth) instead of pcolormesh
PI0_ANCHOR  = False           # True -> prepend the analytic p_i = 0 limit
CMAP_NAME   = 'summer'
CMAP_STOPS  = (0.10, 0.55, 0.92)

BOUNDARY_LW = 1.2             # white contour width
STROKE_LW   = 2.2             # dark halo width (0 -> no halo, crispest pinch)
XTICK_ROT   = 35              # x tick label rotation, degrees
XTICK_EVERY = 1               # 2 -> label every other p_i value
DIAGNOSE    = True            # print baseline crossing + re-entrant regimes

REGIME_LABELS = ['worst',
                 'median',
                 'best']

# p_i = 0  =>  no external noise  =>  both baselines are exactly 1
pi_x  = np.r_[0.0, pi_g]     if PI0_ANCHOR else pi_g
Pun_x = np.r_[1.0, P_un]     if PI0_ANCHOR else P_un
Pba_x = np.r_[1.0, P_ba]     if PI0_ANCHOR else P_ba
Pid_x = np.r_[1.0, P_id]     if PI0_ANCHOR else P_id


def extend_pi(Pg):
    """Prepend a p_i = 0 column (linear extrapolation) if anchoring is on."""
    if not PI0_ANCHOR:
        return Pg
    slope = (Pg[:, 1] - Pg[:, 0]) / (pi_g[1] - pi_g[0])
    F0 = np.clip(Pg[:, 0] - pi_g[0] * slope, 0.0, 1.0)
    return np.column_stack([F0, Pg])


def _regime(Pf, un, ba):
    """0 = worse than both, 1 = better than the worse one, 2 = better than both."""
    hi = np.maximum(un, ba)[None, :]
    lo = np.minimum(un, ba)[None, :]
    return (Pf > lo).astype(int) + (Pf > hi).astype(int), lo, hi


def data_p_range(pgrids):
    """p2q window covered by *all* panels (log axis -> drop p2q = 0)."""
    pos = [pg[pg > 0] for pg in pgrids]
    return max(p.min() for p in pos), min(p.max() for p in pos)


def auto_view(pgrids, Pgrids):
    """p2q band containing the regime changes, padded, clipped to the data."""
    d_lo, d_hi = data_p_range(pgrids)
    lo_all, hi_all = [], []
    for pg, Pg in zip(pgrids, Pgrids):
        m = pg > 0
        R, _, _ = _regime(extend_pi(Pg[m]), Pun_x, Pba_x)
        chg = np.where(np.any(np.diff(R, axis=0) != 0, axis=1))[0]
        if chg.size:
            pgp = pg[m]
            lo_all.append(pgp[chg.min()])
            hi_all.append(pgp[min(chg.max() + 1, len(pgp) - 1)])
    if not lo_all:
        return d_lo, d_hi
    return max(min(lo_all) / VIEW_PAD, d_lo), min(max(hi_all) * VIEW_PAD, d_hi)


def refine(pg, Pg, view):
    """Bilinear resampling: linear in log10(p2q), linear in p_i."""
    m = pg > 0
    pg, Pg = pg[m], extend_pi(Pg[m])
    lo = max(view[0], pg.min())
    hi = min(view[1], pg.max())
    pf = np.logspace(np.log10(lo), np.log10(hi), NFINE_P)
    sf = np.linspace(pi_x.min(), pi_x.max(), NFINE_S)
    tmp = np.array([np.interp(np.log10(pf), np.log10(pg), Pg[:, j])
                    for j in range(Pg.shape[1])]).T              # (NFINE_P, n_pi)
    Pf = np.array([np.interp(sf, pi_x, tmp[i]) for i in range(len(pf))])
    return pf, sf, Pf


def analytic_curves(C):
    """Analytic boundaries on a fine p_i grid.
       [(p_i, p2q_crit, colour, linestyle, label), ...]"""
    if not (ANALYTIC_ON_PHASE and np.isfinite(C) and C > 0):
        return []
    pfi = np.linspace(max(pi_x.min(), 1e-6), pi_x.max(), NFINE_ANA)
    out = []
    if ANALYTIC_SOURCE in ('model', 'both'):
        ef = eps_for(pfi)
        Pi_m = P_ideal_exact(pfi, ef)
        tag = (rf"$\epsilon$={EPS_G:.2f}" if EPS_MODE == 'global'
               else r"$\epsilon(p_i)$")
        out += [(pfi, p_crit_eq28(Pi_m, P_unc_exact(pfi), C, P_INF),
                 ANA_COL['unc'],  '-',
                 #rf"Eq. (28) closed form ({tag}): vs logical"),
                 # rf"analytic model vs logical"),
                 rf"Predicted SRL-uncorr"),
                (pfi, p_crit_eq28(Pi_m, P_bare_exact(pfi), C, P_INF),
                 ANA_COL['bare'], '-',
                 #rf"Eq. (28) closed form ({tag}): vs physical")
                 # rf"analytic model vs physical")]
                 rf"Predicted SRL-bare")]
    if ANALYTIC_SOURCE in ('data', 'both'):
        Pi_d = np.interp(pfi, pi_x, Pid_x)
        Pu_d = np.interp(pfi, pi_x, Pun_x)
        Pb_d = np.interp(pfi, pi_x, Pba_x)
        out += [(pfi, p_crit_eq28(Pi_d, Pu_d, C, P_INF), ANA_COL['unc'],  ':',
                 "measured P: vs logical"),
                (pfi, p_crit_eq28(Pi_d, Pb_d, C, P_INF), ANA_COL['bare'], ':',
                 "measured P: vs physical")]
    return out


# ------------------------------ figure -------------------------------
# panels = [(p_nl[mnl], P_nl[mnl], "noiseless trained", FIT_NL),
#           (p_ns[mns], P_ns[mns], "noisy trained",     FIT_NS)]
panels = [(p_nl[mnl], P_nl[mnl], "noiseless trained", FIT_NL)]

view = P2Q_VIEW or auto_view([q[0] for q in panels], [q[1] for q in panels])
print(f"  Fig C p2q view range : {view[0]:.2e} … {view[1]:.2e}")
if MARK_CLOSURE:
    print(f"  wedge closes (P_ideal = P_bare) : model p_i^max = "
          f"{pi_max_model:.4f}, data {pi_max_data:.4f}   -> there "
          f"p2q_crit -> 0, i.e. log10 -> -inf; the curve leaves the frame at "
          f"the bottom instead of touching p2q = 0")
    print(f"  baselines cross (pseudothreshold): model p_i* = "
          f"{pi_star_model:.4f}, data {pi_star_data:.4f}")

cmap = ListedColormap(plt.get_cmap(CMAP_NAME)(np.array(CMAP_STOPS)))
norm = BoundaryNorm([-.5, .5, 1.5, 2.5], cmap.N)
stroke = ([mpe.Stroke(linewidth=STROKE_LW, foreground='0.15'), mpe.Normal()]
          if STROKE_LW > 0 else None)
ana_stroke = [mpe.Stroke(linewidth=ANA_STROKE, foreground='w'), mpe.Normal()]

# baseline crossing p_i -> the point where solid and dashed contours must pinch
if DIAGNOSE:
    dbl = P_un - P_ba
    ix_cross = np.where(np.diff(np.sign(dbl)) != 0)[0]   # NOTE: not 'idx'!
    for i0 in ix_cross:
        w = dbl[i0] / (dbl[i0] - dbl[i0 + 1])
        pc = pi_g[i0] + w * (pi_g[i0 + 1] - pi_g[i0])
        print(f"  baselines cross at p_i = {pc:.4f}"
              f"  -> solid and dashed contours pinch here")

# figC, axsC = plt.subplots(1, 2, figsize=(16.0, 5.6), sharey=True)
figC, axsC = plt.subplots(1, len(panels), figsize=(8.6*len(panels), 5.6),
                          squeeze=False, sharey=True)
axsC = axsC[0]
im = None

for ax, (pg, Pg, ttl, FITp) in zip(axsC, panels):
    pf, sf, Pf = refine(pg, Pg, view)
    un = np.interp(sf, pi_x, Pun_x)
    ba = np.interp(sf, pi_x, Pba_x)
    R, lo, hi = _regime(Pf, un, ba)

    if DIAGNOSE:                       # per-p_i regime sequence on the raw grid
        mraw = pg > 0
        Rr, _, _ = _regime(extend_pi(Pg[mraw]), Pun_x, Pba_x)
        for jc, pv in enumerate(pi_x):
            seq = Rr[:, jc].astype(int).tolist()
            if np.any(np.diff(seq) > 0):        # non-monotonic -> re-entrant
                print(f"  [{ttl}] p_i={pv:g}: regimes vs p2q = "
                      f"{seq}  (re-entrant -> outside the single-C model)")

    if SMOOTH_FILL:
        im = ax.contourf(sf, pf, R.astype(float), levels=[-.5, .5, 1.5, 2.5],
                         cmap=cmap, norm=norm)
    else:
        im = ax.pcolormesh(sf, pf, R, cmap=cmap, norm=norm, shading='auto')

    # ---------- NUMERIC boundaries from the data (unchanged) ----------
    for fld, ls in [(Pf - lo, '-'), (Pf - hi, '--')]:
        if fld.min() < 0 < fld.max():
            cs = ax.contour(sf, pf, fld, levels=[0], colors='w',
                            linewidths=BOUNDARY_LW, linestyles=ls)
            if stroke is not None:
                cs.set(path_effects=stroke)

    # ---------- ANALYTIC break-even curves ----------
    Cp = FITp['C'] if (MODEL_USE_PANEL_C and np.isfinite(FITp['C'])) else C_MAIN
    for xs, ys, col, ls, lab in analytic_curves(Cp):
        ax.plot(xs, ys, ls=ls, color=col, lw=2.0, zorder=9,
                label=lab, path_effects=ana_stroke)
    if MARK_CLOSURE and np.isfinite(pi_max_model):
        ax.axvline(pi_max_model, color=ANA_COL['bare'], ls='-.', lw=3,
                   alpha=.95, zorder=8,
                   label=rf"$P_{{\rm ideal}}=P_{{\rm bare}}$ at "
                         rf"$p_i={pi_max_model:.3f}$ (wedge closes)")
    if MARK_CLOSURE and np.isfinite(pi_star_model):
        ax.axvline(pi_star_model, color='0.15', ls='-.', lw=3.0, alpha=.9,
                   zorder=8)
    if ANALYTIC_ON_PHASE and ANALYTIC_LEGEND:
        # ax.legend(fontsize=7, loc='upper left', framealpha=.85)
        ax.legend(fontsize=12, loc='upper left', framealpha=.85)

    ax.set_yscale('log')
    ax.set_ylim(*view)
    ax.set_xlim(pi_x.min(), pi_x.max())
    ticks = pi_x[::XTICK_EVERY]
    ax.set_xticks(ticks)
    # ax.set_xticklabels([f"{v:g}" for v in ticks], rotation=XTICK_ROT,
    #                    ha='right', rotation_mode='anchor', fontsize=9)
    # ax.set_xlabel(r"$p_i$")
    # ttl2 = ttl + (rf"   ($C_{{\rm fit}}={Cp:.0f}$)" if np.isfinite(Cp) else "")
    # ax.set_title(ttl2, fontweight='bold')
    ax.set_xticklabels([f"{v:g}" for v in ticks], rotation=XTICK_ROT,
                       ha='right', rotation_mode='anchor', fontsize=12)
    ax.set_xlabel(r"$p_{\mathrm{ext}}$", fontsize=15)
    #ax.axvline(pi_tr, color='w', ls=':', lw=1.4, alpha=.85)

# axsC[0].set_ylabel(r"$p_{2q}$")
axsC[0].set_ylabel(r"$p_{2q}$", fontsize=15)

cb = figC.colorbar(im, ax=axsC, ticks=[0, 1, 2], pad=.02)
# cb.ax.set_yticklabels(REGIME_LABELS)
cb.ax.set_yticklabels(REGIME_LABELS, fontsize=12)

# figC.suptitle("[[5,1,3]] perfect code", fontsize=13, fontweight='bold')
# figC.savefig("internal5_noise_phase_diagram.png", dpi=150, bbox_inches='tight')
# print("  Saved : internal5_noise_phase_diagram.png")
figC.savefig("internal5_noise_phase_diagram.pdf", dpi=150, bbox_inches='tight')
print("  Saved : internal5_noise_phase_diagram.pdf")

# ------------------------------------------------------------------- CSV
with open("internal5_noise_merged.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "p2q", "p_i", "P_bare", "P_uncorr_logical",
                "P_ideal_qae", "P_noiseless_trained", "P_noisy_trained",
                "gain_nl_vs_logical", "gain_ns_vs_logical",
                "gain_nl_vs_physical", "gain_ns_vs_physical",
                "benefit_noisy_training"])
    for ip, p in enumerate(p_nl):
        if p <= 0: continue
        j = idx(p_ns, p) if np.any(np.isclose(p_ns, p)) else None
        for i, pe in enumerate(pi_g):
            ps = P_ns[j, i] if j is not None else None
            w.writerow([L, f"{p:.6e}", f"{pe:.4f}", f"{P_ba[i]:.6f}",
                        f"{P_un[i]:.6f}", f"{P_id[i]:.6f}", f"{P_nl[ip,i]:.6f}",
                        "" if ps is None else f"{ps:.6f}",
                        f"{P_nl[ip,i]-P_un[i]:.6f}",
                        "" if ps is None else f"{ps-P_un[i]:.6f}",
                        f"{P_nl[ip,i]-P_ba[i]:.6f}",
                        "" if ps is None else f"{ps-P_ba[i]:.6f}",
                        "" if ps is None else f"{ps-P_nl[ip,i]:.6f}"])
print("  Saved : internal5_noise_merged.csv")

# ------------------- CSV: numeric vs analytic break-even --------------------
num_unc = np.array([crossover(p_nl[mnl], P_nl[mnl, i], P_un[i]) or np.nan
                    for i in range(len(pi_g))], float)
num_bar = np.array([crossover(p_nl[mnl], P_nl[mnl, i], P_ba[i]) or np.nan
                    for i in range(len(pi_g))], float)
dat_unc = p_crit_eq28(P_id, P_un, C_MAIN, P_INF)
dat_bar = p_crit_eq28(P_id, P_ba, C_MAIN, P_INF)
eps_grid = eps_for(pi_g)
mod_unc = p_crit_eq28(P_ideal_exact(pi_g, eps_grid), P_unc_exact(pi_g),
                      C_MAIN, P_INF)
mod_bar = p_crit_eq28(P_ideal_exact(pi_g, eps_grid), P_bare_exact(pi_g),
                      C_MAIN, P_INF)

with open("internal5_breakeven_eq28.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p_i", "C_fit", "eta", "epsilon_used", "P_inf",
                "P_ideal_exact", "P_uncorr_exact", "P_bare_exact",
                "p2q_crit_num_vs_logical", "p2q_crit_eq28_measured_vs_logical",
                "p2q_crit_eq28_closedform_vs_logical",
                "p2q_crit_num_vs_physical", "p2q_crit_eq28_measured_vs_physical",
                "p2q_crit_eq28_closedform_vs_physical"])
    for i, pe in enumerate(pi_g):
        def s(v): return "" if not np.isfinite(v) else f"{v:.6e}"
        w.writerow([f"{pe:.4f}", f"{C_MAIN:.2f}", f"{C_MAIN/C_COUNT:.4f}",
                    f"{eps_grid[i]:.4f}", f"{P_INF:.4f}",
                    f"{P_ideal_exact(pe, eps_grid[i]):.6f}",
                    f"{float(P_unc_exact(pe)):.6f}",
                    f"{float(P_bare_exact(pe)):.6f}",
                    s(num_unc[i]), s(dat_unc[i]), s(mod_unc[i]),
                    s(num_bar[i]), s(dat_bar[i]), s(mod_bar[i])])
print("  Saved : internal5_breakeven_eq28.csv")

# ------------------------------------------------------------------- summary
print("\n" + "="*104)
print(f"  SUMMARY at p_i = {pi_tr:g}   (L={L}, {nc} CNOT + {nr} rot, "
      f"C_eff={c_eff:.0f})")
print("="*104)
print(f"  P_bare = {P_ba[i_tr]:.6f}   P_uncorr(logical) = {P_un[i_tr]:.6f}"
      f"   P_ideal_QAE = {P_id[i_tr]:.6f}")
print(f"  {'p2q':>9} {'lambda':>8} {'P_nlTrain':>10} {'P_nsTrain':>10} "
      f"{'gain_nl':>9} {'gain_ns':>9} {'benefit':>9} {'ns-bare':>9}")
for ip, p in enumerate(p_nl):
    if p <= 0: continue
    j = idx(p_ns, p) if np.any(np.isclose(p_ns, p)) else None
    pn = P_nl[ip, i_tr]; lam = np.exp(-c_eff*p)
    if j is None:
        print(f"  {p:>9.1e} {lam:>8.4f} {pn:>10.5f} {'--':>10} "
              f"{pn-P_un[i_tr]:>+9.5f} {'--':>9} {'--':>9} {'--':>9}")
    else:
        pv = P_ns[j, i_tr]
        print(f"  {p:>9.1e} {lam:>8.4f} {pn:>10.5f} {pv:>10.5f} "
              f"{pn-P_un[i_tr]:>+9.5f} {pv-P_un[i_tr]:>+9.5f} "
              f"{pv-pn:>+9.5f} {pv-P_ba[i_tr]:>+9.5f}")
print(f"\n  p* vs logical  : noiseless "
      f"{f'{ps_log_nl:.3e}' if ps_log_nl else 'not crossed'}, noisy "
      f"{f'{ps_log_ns:.3e}' if ps_log_ns else 'not crossed'}"
      f"   |  closed form {pc_log_mod:.3e}")
print(f"  p* vs physical : noiseless "
      f"{f'{ps_bar_nl:.3e}' if ps_bar_nl else 'not crossed'}, noisy "
      f"{f'{ps_bar_ns:.3e}' if ps_bar_ns else 'not crossed'}"
      f"   |  closed form {pc_bar_mod:.3e}")
if ps_log_nl and ps_log_ns:
    print(f"  noisy training extends the useful p2q range by "
          f"{ps_log_ns/ps_log_nl:.2f}x (logical baseline)")

# ------------------------- OPTIONAL fit report ------------------------------
if SHOW_FIT_REPORT:
    print("\n" + "="*104)
    print("  CALIBRATION OF THE ANALYTIC MODEL")
    print("="*104)
    print(f"  P_inf              : {P_INF:.4f}  (mode '{P_INF_MODE}', "
          f"plateau estimate {P_inf_auto:.4f})")
    print(f"  gate counts        : N2q = {nc}, N1q = {nr}  -> "
          f"C_count = N2q + N1q/10 = {C_COUNT:.1f}")
    print(f"  C_fit (noiseless)  : {FIT_NL['C']:.1f}   (log slope "
          f"{FIT_NL['C_lin']:.1f}, P-space {FIT_NL['C_nl']:.1f}, rms "
          f"{FIT_NL['rms']:.2e}, {FIT_NL['n_used']} unsaturated pts)")
    print(f"  eta = C_fit/C_count: {FIT_NL['eta']:.3f}   (file c_eff "
          f"{c_eff:.1f}) -> using C = {C_MAIN:.1f} ('{C_SOURCE}')")
    print(f"  C_fit (noisy tr.)  : {FIT_NS['C']:.1f}  (eta "
          f"{FIT_NS['eta']:.3f}, rms {FIT_NS['rms']:.2e})  "
          f"[single-C model questionable: re-entrant]")
    print("  C per p_i          : " + ", ".join(
        f"{pv:g}:{cv:.0f}" for pv, cv in zip(pi_g, FIT_NL['C_per_pi'])
        if np.isfinite(cv)))
    if np.isfinite(FIT_NL['C_per_pi']).sum() > 1:
        cv = np.nanstd(FIT_NL['C_per_pi'])/np.nanmean(FIT_NL['C_per_pi'])
        print(f"  column scatter     : {100*cv:.1f} %  "
              f"(<10 % validates the single-C exponential)")
    print(f"\n  N_fail2 (weight-2 failures) : {N_FAIL2}   "
          f"enumerator A_w = {list(map(int, ENUM5))} (sum {int(ENUM5.sum())})")
    print(f"  epsilon (global fit)        : {EPS_G:.4f}  "
          f"(rms in P {EPS_RMS:.2e}, mode '{EPS_MODE}')")
    print("  epsilon per p_i             : " + ", ".join(
        f"{pv:g}:{ev:.3f}" for pv, ev in zip(pi_g, EPS_PI)))
    print(f"  alpha (P_id-P_unc)/p_i^2   : {FQ['alpha']:.2f}   "
          f"[0.13*N_fail2 = {0.13*N_FAIL2:.1f}]")
    print(f"  beta  ((4/3)p_i-(P_id-P_ba))/p_i^2 : {FQ['beta']:.2f}")
    print(f"  gamma (1-P_unc)/p_i^2      : {FQ['gamma']:.2f}   "
          f"[(8/27)*N_fail2 = {8*N_FAIL2/27:.1f}]")
    print(f"  quad-fit window in p_i     : {np.round(FQ['p_fit'],4)}")
    print(f"\n  p_i^max  (P_ideal = P_bare) : closed form "
          f"{pi_max_model:.4f}  |  data {pi_max_data:.4f}")
    print(f"  p_i*     (P_unc  = P_bare)  : closed form "
          f"{pi_star_model:.4f}  |  data {pi_star_data:.4f}")
    print("\n  closed-form baselines vs measured (sampling error only)")
    print(f"  {'p_i':>7} {'P_unc_ex':>10} {'P_unc_num':>10} {'dev':>9} "
          f"{'P_ba_ex':>9} {'P_ba_num':>9} {'P_id_ex':>9} {'P_id_num':>9}")
    for i, pe in enumerate(pi_g):
        print(f"  {pe:>7.3f} {P_un_x[i]:>10.6f} {P_un[i]:>10.6f} "
              f"{P_un_x[i]-P_un[i]:>+9.2e} {P_ba_x[i]:>9.6f} {P_ba[i]:>9.6f} "
              f"{P_id_x[i]:>9.6f} {P_id[i]:>9.6f}")
    print("\n  break-even p2q: numeric vs analytic")
    print(f"  {'p_i':>7} {'num|log':>11} {'meas|log':>11} {'form|log':>11} "
          f"{'num|bare':>11} {'meas|bare':>11} {'form|bare':>11}")
    def _f(v): return f"{v:11.3e}" if np.isfinite(v) else f"{'--':>11}"
    for i, pe in enumerate(pi_g):
        print(f"  {pe:>7.3f} {_f(num_unc[i])} {_f(dat_unc[i])} {_f(mod_unc[i])} "
              f"{_f(num_bar[i])} {_f(dat_bar[i])} {_f(mod_bar[i])}")

print("\n  Done")