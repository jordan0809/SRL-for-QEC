# ============================================================================
# plot_steane_depolar_noisy.py -- memory-protocol figures for the Steane [[7,1,3]] code
#   FIG A : grid, one panel per p2q, x = p_i, baselines + QAE
#   FIG B : (p_i, p2q) 3-regime phase diagram  +  ANALYTIC break-even curves
# ============================================================================
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as mpe
from matplotlib.colors import ListedColormap, BoundaryNorm

C_NL, C_NS = "internal7_noiseless_trained.npz", "internal7_noisy_trained.npz"
NCOL, NROW = 3, 3
SHOW_BANDS = False       
SHOW_IDEAL = True         # the p2q = 0 row as an ideal-QAE reference curve

# ----------------------------- analytic model switches ---------------------
P_INF_MODE   = 'fixed'    # 'fixed' -> 0.5 ; 'auto' -> read the large-p2q plateau
P_INF_FIXED  = 0.5
C_SOURCE     = 'fit'      # 'fit' -> fit C from the grid ; 'file' -> use c_eff
CP_MAX       = 1.0        # discard points with C*p2q > CP_MAX (saturated)
N2FAIL       = 147        # 3 -> 3q phase flip, 90 -> [[5,1,3]], 147 -> Steane
KMAX_PEXT    = 4          # number of smallest p_i points used in the p^2 fits
SHOW_ANALYTIC = True      # overlay the analytic boundaries on FIG B

COL = dict(uncorr='#e41a1c', bare='#ff7f00', nltrain='#4daf4a',
           nstrain='#377eb8', ideal='#4d4d4d', train='k')
LBL = dict(uncorr="Uncorrected logical", bare="Bare physical",
           nltrain="Noiseless-trained SRL", nstrain="Noisy-trained SRL",
           ideal=r"ideal SRL ($p_{2q}=0$)")


def idx(a, v):
    return int(np.argmin(np.abs(np.asarray(a) - v)))


def get(d, key, default=None):
    return np.asarray(d[key]) if key in d.files else default


# ========================================================== analytic helpers
def estimate_Pinf(p2q, P, frac=0.25):
    """Mean P over the largest-p2q rows: the empirical saturation floor."""
    m = np.asarray(p2q) > 0
    pv, Pv = np.asarray(p2q)[m], np.asarray(P)[m]
    k = max(1, int(np.ceil(frac * len(pv))))
    return float(np.mean(Pv[np.argsort(pv)[-k:]]))


def fit_C(p2q, P, P_ideal, P_inf, cp_max=CP_MAX, floor=1e-4, n_iter=5):
    """y = -ln[(P-P_inf)/(P_ideal-P_inf)] = C*p2q, least squares through 0."""
    p2q = np.asarray(p2q, float)
    P = np.atleast_2d(np.asarray(P, float))
    if P.shape[0] != p2q.size:
        P = P.T
    x = np.repeat(p2q[:, None], P.shape[1], axis=1)
    num = P - P_inf
    den = np.broadcast_to(np.asarray(P_ideal, float).ravel() - P_inf, P.shape)
    ok = (x > 0) & (num > floor) & (den > floor)
    if not ok.any():
        return np.nan
    y = np.zeros_like(num)
    y[ok] = -np.log(num[ok] / den[ok])
    w = ok.astype(float)
    C = np.nan
    for _ in range(n_iter):
        if w.sum() == 0:
            return np.nan
        C = float(np.sum(w * x * y) / np.sum(w * x * x))
        w_new = np.where(ok & (C * x < cp_max), 1.0, 0.0)
        if w_new.sum() == 0:
            break
        w = w_new
    return C


def p2q_crit(P_ideal, P_ref, C, P_inf):
    """Exact break-even threshold; NaN where no positive budget exists."""
    num = np.asarray(P_ideal, float) - P_inf
    den = np.asarray(P_ref, float) - P_inf
    out = np.full(np.broadcast(num, den).shape, np.nan)
    ok = (num > 0) & (den > 0) & (num > den) & np.isfinite(C) & (C > 0)
    out[ok] = np.log(np.broadcast_to(num, out.shape)[ok] /
                     np.broadcast_to(den, out.shape)[ok]) / C
    return out


def fit_power(x, y, kmax=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = (x > 0) & (y > 0)
    x, y = x[m], y[m]
    o = np.argsort(x); x, y = x[o], y[o]
    if kmax:
        x, y = x[:kmax], y[:kmax]
    if x.size < 2:
        return np.nan, np.nan
    s, lna = np.polyfit(np.log(x), np.log(y), 1)
    return float(s), float(np.exp(lna))


def fit_quadratic_coeff(x, y, kmax=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = x > 0
    x, y = x[m], y[m]
    o = np.argsort(x); x, y = x[o], y[o]
    if kmax:
        x, y = x[:kmax], y[:kmax]
    return float(np.sum(y * x**2) / np.sum(x**4))


def crossover(pv, fv, thr):
    """Log-interpolated p2q where fv drops through thr."""
    pv = np.asarray(pv, float); fv = np.asarray(fv, float)
    o = np.argsort(pv); pv, fv = pv[o], fv[o]
    for i in range(len(pv) - 1):
        a, b = fv[i] - thr, fv[i + 1] - thr
        if a > 0 >= b and pv[i] > 0:
            w = a / (a - b)
            return 10**(np.log10(pv[i]) +
                        w * (np.log10(pv[i + 1]) - np.log10(pv[i])))
    return None


# ================================================================ load data
dn = np.load(C_NL, allow_pickle=True)
HAVE_NS = os.path.exists(C_NS)
ds = np.load(C_NS, allow_pickle=True) if HAVE_NS else None

pi_g_full = np.asarray(dn['p_ext'], float)
p_nl = np.asarray(dn['p2q'], float)
P_nl_full = np.asarray(dn['p_corr'])
S_nl_full = np.asarray(dn['p_corr_std'])
P_un_full = np.asarray(dn['p_uncorr']).ravel()
P_ua_full = np.asarray(dn['p_uncorr_analytic']).ravel()
P_ba_full = np.asarray(dn['p_bare']).ravel()

_pid = get(dn, 'p_ideal_qae')
if _pid is None:
    _pid = P_nl_full[idx(p_nl, 0.0)]
P_id_full = np.asarray(_pid).ravel()

L      = int(dn['n_layers'])
nc     = int(get(dn, 'n_cnot_full', 0))
nr     = int(get(dn, 'n_rot_full', 0))
c_eff  = float(get(dn, 'c_eff', np.nan))
pi_tr  = float(dn['pext_train'])
LBL_TR = rf"training level $p_i={pi_tr:g}$"

# ---------------------------------------------------------------------------
# Reconcile the p_i grids between the noiseless- and noisy-trained files.
# The two training scripts can build slightly different p_i grids (e.g. the
# noiseless run adds extra points via PEXT_SWEEP_EXTRA). Rather than requiring
# an exact match, keep only the p_i values present in BOTH files.
# ---------------------------------------------------------------------------
if HAVE_NS:
    pi_g_ns = np.asarray(ds['p_ext'], float)
    keep_mask = np.array(
        [np.any(np.isclose(v, pi_g_ns, atol=1e-9)) for v in pi_g_full])
    if not keep_mask.all():
        dropped = pi_g_full[~keep_mask]
        print(f"  WARNING: p_i grid mismatch between '{C_NL}' and '{C_NS}'; "
              f"dropping points absent from the noisy-trained file: "
              f"{np.round(dropped, 6)}")
else:
    keep_mask = np.ones_like(pi_g_full, dtype=bool)

pi_g = pi_g_full[keep_mask]
P_nl = P_nl_full[:, keep_mask]
S_nl = S_nl_full[:, keep_mask]
P_un = P_un_full[keep_mask]
P_ua = P_ua_full[keep_mask]
P_ba = P_ba_full[keep_mask]
P_id = P_id_full[keep_mask]

i_tr = idx(pi_g, pi_tr)

if HAVE_NS:
    col_idx = [int(np.argmin(np.abs(pi_g_ns - v))) for v in pi_g]
    p_ns = np.asarray(ds['p2q'], float)
    P_ns = np.asarray(ds['p_aware'])[:, col_idx]
    S_ns = np.asarray(ds['p_aware_std'])[:, col_idx]
else:
    p_ns = np.array([]); P_ns = None; S_ns = None

print(f"  p_i grid ({len(pi_g)}) : {np.round(pi_g, 4)}")
print(f"  L={L}, {nc} CNOT + {nr} rot, C_eff(file)={c_eff:.0f}, "
      f"p_i_train={pi_tr}")
print(f"  noisy-trained file: {'yes' if HAVE_NS else 'no'}")


# ================================================== analytic model calibration
P_inf_auto = estimate_Pinf(p_nl, P_nl)
P_INF = P_INF_FIXED if P_INF_MODE == 'fixed' else P_inf_auto

C_fit = fit_C(p_nl, P_nl, P_id, P_INF)
C_col = np.array([fit_C(p_nl, P_nl[:, [j]], P_id[[j]], P_INF)
                  for j in range(len(pi_g))])
C_use = C_fit if (C_SOURCE == 'fit' and np.isfinite(C_fit)) else c_eff

B_unc  = P_id - P_un
B_bare = P_id - P_ba
s_unc, _ = fit_power(pi_g, B_unc, kmax=KMAX_PEXT)
a_unc    = fit_quadratic_coeff(pi_g, B_unc, kmax=KMAX_PEXT)
gamma    = fit_quadratic_coeff(pi_g, 1 - P_un, kmax=KMAX_PEXT)
beta     = fit_quadratic_coeff(pi_g, 1 - P_id, kmax=KMAX_PEXT)
s_bare, _ = fit_power(pi_g, np.clip(B_bare, 1e-12, None), kmax=KMAX_PEXT)

sat = gamma / ((2 / 27) * N2FAIL * 4)
_r  = beta / max(sat * (2 / 27) * N2FAIL, 1e-12)
eps = 2 - np.sqrt(_r) if _r > 0 else np.nan

pc_un_g = p2q_crit(P_id, P_un, C_use, P_INF)
pc_ba_g = p2q_crit(P_id, P_ba, C_use, P_INF)

print("\n" + "=" * 104)
print("  ANALYTIC MODEL CALIBRATION")
print("=" * 104)
print(f"  P_inf : used {P_INF:.4f}  (plateau estimate {P_inf_auto:.4f}, "
      f"mode='{P_INF_MODE}')")
print(f"  C     : fit {C_fit:.1f}   file {c_eff:.1f}   -> using {C_use:.1f}"
      f"   ({C_SOURCE})")
print(f"  C per p_i column : {np.array2string(C_col, precision=0)}")
if np.isfinite(C_col).sum() > 1:
    cv = np.nanstd(C_col) / np.nanmean(C_col)
    print(f"  column scatter   : {100*cv:.1f} %   "
          f"(<10 % validates the single-C exponential model)")
print(f"  benefit vs no-QAE : exponent s = {s_unc:.2f} (theory 2), "
      f"a = {a_unc:.2f}   [count 2*N2fail/9 = {2*N2FAIL/9:.2f}]")
print(f"  benefit vs bare   : exponent s = {s_bare:.2f} (theory 1)")
print(f"  gamma = {gamma:.2f}   beta = {beta:.2f}   "
      f"saturation = {sat:.2f}   eps = {eps:.2f}")
print(f"  B/p_i^2 flatness  : "
      f"{np.round(B_unc[pi_g>0]/pi_g[pi_g>0]**2, 2)}")
if beta + 8/9 > 0:
    print(f"  p_i(max)  [P_ideal = P_bare]   ~ {(4/3)/(beta+8/9):.4f}")
if gamma + 8/9 > 0:
    print(f"  p_i*      [P_uncorr = P_bare]  ~ {(4/3)/(gamma+8/9):.4f}")

print("\n  VALIDATION: analytic threshold vs numerical crossover")
print(f"  {'p_i':>8} {'P_ideal':>9} {'P_uncorr':>9} {'P_bare':>8} "
      f"{'an.vs_unc':>11} {'num.vs_unc':>11} {'an.vs_bare':>11} "
      f"{'num.vs_bare':>12}")
mnl = p_nl > 0
for j, pe in enumerate(pi_g):
    num_u = crossover(p_nl[mnl], P_nl[mnl, j], P_un[j])
    num_b = crossover(p_nl[mnl], P_nl[mnl, j], P_ba[j])
    f = lambda v: ("--" if v is None or not np.isfinite(v) else f"{v:.3e}")
    print(f"  {pe:>8.4f} {P_id[j]:>9.5f} {P_un[j]:>9.5f} {P_ba[j]:>8.5f} "
          f"{f(pc_un_g[j]):>11} {f(num_u):>11} "
          f"{f(pc_ba_g[j]):>11} {f(num_b):>12}")


# ==================================================================== FIG A
p_all = np.array([p for p in p_nl if p > 0.0])
if len(p_all) > NROW * NCOL:
    sel = np.unique(np.linspace(0, len(p_all) - 1,
                                NROW * NCOL).round().astype(int))
    p_show = p_all[sel]
else:
    p_show = p_all


def nsrow(p):
    if not HAVE_NS or not np.any(np.isclose(p_ns, p)):
        return None, None
    j = idx(p_ns, p)
    return P_ns[j], S_ns[j]


figA, axes = plt.subplots(NROW, NCOL, figsize=(5.6*NCOL, 4.3*NROW),
                          squeeze=False, sharex=True, sharey=True)
for k in range(NROW*NCOL):
    ax = axes[k//NCOL][k % NCOL]
    if k >= len(p_show):
        ax.axis('off'); continue
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
    ax.axvline(pi_tr, color=COL['train'], ls=':', lw=3, alpha=.55, zorder=2)
    ax.set_title(rf"$p_{{2q}}={p:.0e}$,  $p_{{1q}}={p*0.1:.0e}$",
                 fontweight='bold', fontsize=15)
    ax.grid(alpha=.3)
    ax.tick_params(axis='both', labelsize=12)
    if k == 0:
        ax.legend(fontsize=12, loc='lower left', framealpha=.9)
for c in range(NCOL):
    axes[-1][c].set_xlabel(r"$p_{\mathrm{ext}}$", fontsize=15)
for r in range(NROW):
    axes[r][0].set_ylabel(r"$\langle F \rangle$", fontsize=15)
figA.tight_layout()
figA.savefig("internal7_noise_grid_vs_pi.pdf", dpi=150, bbox_inches='tight')
print("\n  Saved : internal7_noise_grid_vs_pi.pdf")


# ==================================================================== FIG B
# Exact analytic phase diagram
P_INF_PHASE = 0.5
C_FIT_CPMAX_PHASE = 1.0
C_FIT_FLOOR_PHASE = 2e-3
EPS_MAX_PHASE = 3.0
NFINE_ANA_PHASE = 1200
ANA_COL = dict(unc='#ff00ff', bare='#00e5ff')

phase_stabilizers = [[0, 2, 4, 6], [1, 2, 5, 6], [3, 4, 5, 6]]
phase_rows = [sum(1 << (6-q) for q in support) for support in phase_stabilizers]
phase_space = {0}
for phase_row in phase_rows:
    phase_space |= {value ^ phase_row for value in phase_space}
phase_units = [0] + [1 << (6-q) for q in range(7)]
phase_allowed = sorted({value ^ unit for value in phase_space for unit in phase_units})
phase_or = (np.asarray(phase_allowed)[:, None] | np.asarray(phase_allowed)[None, :]).ravel()
phase_enum = np.bincount([int(value).bit_count() for value in phase_or], minlength=8).astype(float)

def phase_lam(p):
    return 1.0 - 4.0*np.asarray(p, float)/3.0

def phase_compose(*probabilities):
    lam = 1.0
    for probability in probabilities:
        lam *= phase_lam(probability)
    return 0.75*(1.0 - lam)

def phase_from_enum(probability):
    probability = np.asarray(probability, float)
    p_identity = np.zeros_like(probability)
    for weight, coefficient in enumerate(phase_enum):
        if coefficient:
            p_identity += coefficient*(1.0-probability)**(7-weight)*(probability/3.0)**weight
    return (1.0 + 2.0*p_identity)/3.0

def phase_uncorrected(p):
    return phase_from_enum(phase_compose(p, p))

def phase_ideal(p, epsilon):
    return phase_from_enum(phase_compose(np.asarray(epsilon)*np.asarray(p), p))

def phase_root(function, lower, upper, tol=1e-13):
    f_lower, f_upper = function(lower), function(upper)
    if not (np.isfinite(f_lower) and np.isfinite(f_upper)) or f_lower*f_upper > 0:
        return np.nan
    for _ in range(300):
        midpoint = 0.5*(lower + upper)
        f_midpoint = function(midpoint)
        if f_lower*f_midpoint <= 0:
            upper, f_upper = midpoint, f_midpoint
        else:
            lower, f_lower = midpoint, f_midpoint
        if upper-lower < tol:
            break
    return 0.5*(lower + upper)

def phase_golden_min(function, lower, upper, tol=1e-10):
    ratio = 0.5*(np.sqrt(5.0)-1.0)
    left = upper-ratio*(upper-lower)
    right = lower+ratio*(upper-lower)
    f_left, f_right = function(left), function(right)
    for _ in range(400):
        if f_left < f_right:
            upper, right, f_right = right, left, f_left
            left = upper-ratio*(upper-lower)
            f_left = function(left)
        else:
            lower, left, f_left = left, right, f_right
            right = lower+ratio*(upper-lower)
            f_right = function(right)
        if abs(upper-lower) < tol*(abs(lower)+abs(upper)+1e-12):
            break
    return 0.5*(lower+upper)

def phase_eps_at(p, target):
    difference = lambda epsilon: float(phase_ideal(p, epsilon)-target)
    if difference(0.0) <= 0:
        return 0.0
    if difference(EPS_MAX_PHASE) >= 0:
        return EPS_MAX_PHASE
    return phase_root(difference, 0.0, EPS_MAX_PHASE)

def phase_fit_c():
    matrix = np.asarray(P_nl, float)
    p2q = np.asarray(p_nl, float)
    coordinates = np.repeat(p2q[:, None], matrix.shape[1], axis=1)
    numerator = matrix-P_INF_PHASE
    denominator = np.broadcast_to(P_id-P_INF_PHASE, matrix.shape)
    valid = (coordinates > 0) & (numerator > C_FIT_FLOOR_PHASE) & (denominator > C_FIT_FLOOR_PHASE)
    transformed = np.zeros_like(numerator)
    transformed[valid] = -np.log(numerator[valid]/denominator[valid])
    weights = valid.astype(float)
    c_linear = float(np.sum(weights*coordinates*transformed)/np.sum(weights*coordinates**2))
    for _ in range(6):
        new_weights = np.where(valid & (c_linear*coordinates < C_FIT_CPMAX_PHASE), 1.0, 0.0)
        if new_weights.sum() < 2:
            break
        new_c = float(np.sum(new_weights*coordinates*transformed)/np.sum(new_weights*coordinates**2))
        if abs(new_c-c_linear) < 1e-9*max(1.0, abs(c_linear)):
            weights = new_weights
            break
        c_linear, weights = new_c, new_weights
    used = weights > 0
    objective = lambda c: float(np.sum((P_INF_PHASE + np.exp(-c*coordinates[used])*denominator[used] - matrix[used])**2))
    return phase_golden_min(objective, max(1e-3, 0.2*c_linear), 5.0*c_linear)

def phase_critical(ideal, reference, c_value):
    numerator = np.asarray(ideal)-P_INF_PHASE
    denominator = np.asarray(reference)-P_INF_PHASE
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    out = np.full(numerator.shape, np.nan)
    valid = (numerator > 0) & (denominator > 0) & (numerator > denominator)
    out[valid] = np.log(numerator[valid]/denominator[valid])/c_value
    return out

phase_c = phase_fit_c()
phase_eps = np.array([phase_eps_at(value, target) for value, target in zip(pi_g, P_id)])
phase_x = np.arange(len(pi_g), dtype=float)
phase_view_input = p_nl[mnl]
phase_regime_raw = P_nl[mnl]

def phase_regime(values, uncorrected, bare):
    high = np.maximum(uncorrected, bare)[None, :]
    low = np.minimum(uncorrected, bare)[None, :]
    return (values > low).astype(int)+(values > high).astype(int), low, high

raw_regime, _, _ = phase_regime(phase_regime_raw, P_un, P_ba)
changes = np.where(np.any(np.diff(raw_regime, axis=0) != 0, axis=1))[0]
if changes.size:
    phase_view = (max(phase_view_input[changes.min()]/1.5, phase_view_input.min()),
                  min(phase_view_input[min(changes.max()+1, len(phase_view_input)-1)]*1.5,
                      phase_view_input.max()))
else:
    phase_view = (phase_view_input.min(), phase_view_input.max())

phase_pf = np.logspace(np.log10(phase_view[0]), np.log10(phase_view[1]), 400)
phase_xf = np.linspace(phase_x.min(), phase_x.max(), 400)
phase_tmp = np.array([np.interp(np.log10(phase_pf), np.log10(phase_view_input), phase_regime_raw[:, column])
                      for column in range(phase_regime_raw.shape[1])]).T
phase_values = np.array([np.interp(phase_xf, phase_x, phase_tmp[row]) for row in range(len(phase_pf))])
phase_un = np.interp(phase_xf, phase_x, P_un)
phase_bare = np.interp(phase_xf, phase_x, P_ba)
phase_regions, phase_low, phase_high = phase_regime(phase_values, phase_un, phase_bare)

phase_cmap = ListedColormap(plt.get_cmap('summer')(np.array((0.10, 0.55, 0.92))))
phase_norm = BoundaryNorm([-.5, .5, 1.5, 2.5], phase_cmap.N)
phase_stroke = [mpe.Stroke(linewidth=2.2, foreground='0.15'), mpe.Normal()]
phase_analytic_stroke = [mpe.Stroke(linewidth=3.2, foreground='w'), mpe.Normal()]

figB, ax = plt.subplots(figsize=(9.2, 5.8))
im = ax.contourf(phase_xf, phase_pf, phase_regions.astype(float),
                 levels=[-.5, .5, 1.5, 2.5], cmap=phase_cmap, norm=phase_norm)
for field, linestyle in [(phase_values-phase_low, '-'), (phase_values-phase_high, '--')]:
    if field.min() < 0 < field.max():
        contour = ax.contour(phase_xf, phase_pf, field, levels=[0], colors='w',
                             linewidths=1.2, linestyles=linestyle)
        contour.set(path_effects=phase_stroke)

phase_fine_pi = np.linspace(max(pi_g.min(), 1e-6), pi_g.max(), NFINE_ANA_PHASE)
phase_fine_x = np.interp(phase_fine_pi, pi_g, phase_x)
phase_fine_eps = np.interp(phase_fine_pi, pi_g, phase_eps)
for curve, color, label in [
    (phase_critical(phase_ideal(phase_fine_pi, phase_fine_eps), phase_uncorrected(phase_fine_pi), phase_c),
     ANA_COL['unc'], 'Predicted SRL-uncorr'),
    (phase_critical(phase_ideal(phase_fine_pi, phase_fine_eps), 1.0-(2.0/3.0)*phase_compose(phase_fine_pi, phase_fine_pi), phase_c),
     ANA_COL['bare'], 'Predicted SRL-bare'),
]:
    ax.plot(phase_fine_x, curve, color=color, lw=2.0, zorder=9, label=label,
            path_effects=phase_analytic_stroke)

ax.set_yscale('log')
ax.set_ylim(*phase_view)
ax.set_xlim(phase_x.min(), phase_x.max())
ax.set_xticks(phase_x)
ax.set_xticklabels([f"{value:g}" for value in pi_g], fontsize=12)
ax.set_xlabel(r"$p_{\mathrm{ext}}$", fontsize=15)
ax.set_ylabel(r"$p_{2q}$", fontsize=15)
ax.tick_params(axis='y', labelsize=12)
ax.legend(fontsize=12, loc='upper left', framealpha=.85)
cb = figB.colorbar(im, ax=ax, ticks=[0, 1, 2], pad=.02)
cb.ax.set_yticklabels(['worst', 'median', 'best'], fontsize=12)
figB.savefig("internal7_noise_phase_diagram.pdf", dpi=150, bbox_inches='tight')
print("  Saved : internal7_noise_phase_diagram.pdf")


# ======================================================================= CSV
with open("internal7_noise_merged.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "p2q", "p_i", "P_bare", "P_uncorr_logical",
                "P_uncorr_analytic", "P_ideal_qae", "P_analytic_model",
                "P_noiseless_trained", "P_noisy_trained",
                "gain_nl_vs_logical", "gain_ns_vs_logical",
                "gain_nl_vs_physical", "gain_ns_vs_physical",
                "benefit_noisy_training", "p2q_crit_vs_logical",
                "p2q_crit_vs_physical"])
    for ip, p in enumerate(p_nl):
        if p <= 0:
            continue
        j = idx(p_ns, p) if (HAVE_NS and np.any(np.isclose(p_ns, p))) else None
        for i, pe in enumerate(pi_g):
            ps = P_ns[j, i] if j is not None else None
            Pmod = P_INF + np.exp(-C_use*p)*(P_id[i]-P_INF)
            w.writerow([L, f"{p:.6e}", f"{pe:.4f}", f"{P_ba[i]:.6f}",
                        f"{P_un[i]:.6f}", f"{P_ua[i]:.6f}", f"{P_id[i]:.6f}",
                        f"{Pmod:.6f}", f"{P_nl[ip,i]:.6f}",
                        "" if ps is None else f"{ps:.6f}",
                        f"{P_nl[ip,i]-P_un[i]:.6f}",
                        "" if ps is None else f"{ps-P_un[i]:.6f}",
                        f"{P_nl[ip,i]-P_ba[i]:.6f}",
                        "" if ps is None else f"{ps-P_ba[i]:.6f}",
                        "" if ps is None else f"{ps-P_nl[ip,i]:.6f}",
                        f"{pc_un_g[i]:.6e}" if np.isfinite(pc_un_g[i]) else "",
                        f"{pc_ba_g[i]:.6e}" if np.isfinite(pc_ba_g[i]) else ""])
print("  Saved : internal7_noise_merged.csv")


# =================================================================== summary
print("\n" + "=" * 104)
print(f"  SUMMARY at p_i = {pi_tr:g}  (L={L}, {nc} CNOT + {nr} rot, "
      f"C_used={C_use:.0f}, P_inf={P_INF:.3f})")
print("=" * 104)
print(f"  P_bare = {P_ba[i_tr]:.6f}   P_uncorr(logical) = {P_un[i_tr]:.6f}"
      f"   P_ideal_QAE = {P_id[i_tr]:.6f}")
print(f"  {'p2q':>9} {'lambda':>8} {'P_model':>9} {'P_nlTrain':>10} "
      f"{'P_nsTrain':>10} {'gain_nl':>9} {'gain_ns':>9} {'nl-bare':>9}")
for ip, p in enumerate(p_nl):
    if p <= 0:
        continue
    j = idx(p_ns, p) if (HAVE_NS and np.any(np.isclose(p_ns, p))) else None
    pn = P_nl[ip, i_tr]; lam = np.exp(-C_use*p)
    pm = P_INF + lam*(P_id[i_tr]-P_INF)
    pv = f"{P_ns[j,i_tr]:.5f}" if j is not None else "--"
    gs = f"{P_ns[j,i_tr]-P_un[i_tr]:+.5f}" if j is not None else "--"
    print(f"  {p:>9.1e} {lam:>8.4f} {pm:>9.5f} {pn:>10.5f} {pv:>10} "
          f"{pn-P_un[i_tr]:>+9.5f} {gs:>9} {pn-P_ba[i_tr]:>+9.5f}")
print(f"\n  p* vs logical  : numeric "
      f"{f'{ps_log_nl:.3e}' if ps_log_nl else 'not crossed'}"
      f",  analytic {pc_un_g[i_tr]:.3e}")
print(f"  p* vs physical : numeric "
      f"{f'{ps_bar_nl:.3e}' if ps_bar_nl else 'not crossed'}"
      f",  analytic {pc_ba_g[i_tr]:.3e}")
print("\n  Done")