# ============================================================================
# steane_noiseless_trained_prune.py  
# noiseless-trained params deployed with internal gate noise inside the 
# split-interval protocol.  No training.
# ============================================================================
import os, sys, time, csv, traceback
import numpy as np
import jax, jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import qae7_noisy_common_prune as C
from qae7_noisy_common_prune import *

# ============================== SWITCHES ====================================
VALIDATE         = False    
STRICT           = False    
PRUNE_ENABLE     = False    
N_TEST_STATES    = 1000     
CHUNK            = 20
OUT_TAG          = "noiseless_trained"
L_OVERRIDE       = None     # None -> L_TARGET
PEXT_SWEEP_EXTRA = [0.005, 0.02]
MAKE_PLOT        = True
SKIP_PENNYLANE   = True     # PennyLane cross-check is pure validation
# ============================================================================

TAG   = OUT_TAG
FNPZ  = f"internal7_{TAG}.npz"
FPART = f"internal7_{TAG}_partial.npz"
FCSV  = f"internal7_{TAG}.csv"
FPNG  = f"internal7_{TAG}.png"

print("="*78)
print("  [[7,1,3]] Steane -- noiseless-trained QAE, noisy deployment")
print("="*78)
print(f"  JAX backend : {jax.default_backend().upper()}")
print(f"  validation  : {'ON' if VALIDATE else 'OFF'}   "
      f"strict={STRICT}   pruning={'ON' if PRUNE_ENABLE else 'OFF'}")
try:
    print_config()
except Exception:
    pass


# ------------------------------------------------------------------ helpers
def soft(label, fn, *a, **k):
    """Run an optional step.  Never kills the run unless STRICT is True."""
    if fn is None:
        print(f"  [skip] {label}: not present in the common module")
        return None
    try:
        t = time.time()
        r = fn(*a, **k)
        print(f"  [ok]   {label}  ({time.time()-t:.0f}s)")
        return r
    except Exception as e:
        print(f"  [WARN] {label} FAILED -> ignored: {type(e).__name__}: {e}")
        if STRICT:
            raise
        return None


def get_params_zyz(params_old, n_layers):
    """ZYZ parameters, WITHOUT relying on the validation path succeeding."""
    fn = getattr(C, 'validate_decomposition', None)
    if fn is not None:
        try:
            err, p = fn(params_old, n_layers, skip=SKIP_PENNYLANE)
            if err is not None and np.isfinite(err):
                print(f"  [params] decomposition residual = {err:.2e}")
            return np.asarray(p, float)
        except Exception as e:
            print(f"  [WARN] validate_decomposition unusable ({e}); "
                  f"falling back to a direct converter")
    for nm in ('params_old_to_zyz', 'to_zyz_params', 'convert_params_zyz',
               'decompose_params'):
        g = getattr(C, nm, None)
        if g is not None:
            print(f"  [params] using C.{nm}()")
            return np.asarray(g(params_old, n_layers), float)
    raise SystemExit("ABORT: no way to obtain the ZYZ parameters")


def arr(x):
    return np.asarray(x)


# ------------------------------------------------- optional protocol checks
if VALIDATE:
    print("\n  [validation block]")
    soft("depolarizing channels",  getattr(C, 'validate_depol_channels', None))
    soft("external depolarizing",  getattr(C, 'validate_external_depol', None))
    soft("logical code",           getattr(C, 'validate_logical_code', None))
    soft("recovery decoder",       getattr(C, 'validate_recovery', None))
    soft("analytic baseline",      getattr(C, 'validate_uncorr_formula', None))
else:
    print("\n  [validation] skipped (VALIDATE = False)")

# --------------------------------------------------------- reference model
ref = load_noiseless_reference(l_override=L_OVERRIDE)
L   = int(ref['n_layers'])

PI_REF   = [float(x) for x in ref['p_ext']]
PI_SWEEP = sorted(set(PI_REF) | set(float(x) for x in PEXT_SWEEP_EXTRA))
IN_REF   = np.array([any(abs(p-r) < 1e-12 for r in PI_REF) for p in PI_SWEEP])
I_CTRL   = int(np.argmin(np.abs(np.array(PI_SWEEP) - PEXT_TRAIN)))

# p2q = 0 is FREE and gives the ideal-QAE reference row (depol(0) == identity)
P2Q = sorted(set([0.0]) | set(float(x) for x in P2Q_SWEEP))

try:
    BE = float(break_even_p_i())
except Exception:
    BE = float('nan')

print(f"\n  model     : L={L}  ({len(ref['params_old'])} raw params)  "
      f"| best L in npz = {ref['best_n_layers']}")
print(f"  p_i grid  : {PI_SWEEP}")
print(f"  p2q grid  : {P2Q}   (p1q = {P1Q_RATIO} * p2q)")
print(f"  states    : {N_TEST_STATES} per p_i")

params_zyz = get_params_zyz(ref['params_old'], L)
params_j   = jnp.asarray(params_zyz)

# ------------------------------------------------------- optional pruning
USE_PRUNE, PRUNE_ST, params_dep = False, None, params_zyz
if PRUNE_ENABLE:
    print("\n  [pruning]")
    soft("sparsity report", getattr(C, 'param_sparsity_report', None),
         params_zyz, L)
    out = soft("prune validation", getattr(C, 'validate_pruning', None),
               params_zyz, L, PRUNE_TOL_ROT, PRUNE_TOL_BLOCK)
    if out is not None:
        dev_U, _ops, PRUNE_ST, params_sp = out
        triv = getattr(C, 'prune_is_trivial', lambda st: False)
        if triv(PRUNE_ST):
            print("  [prune] nothing prunable -> full circuit")
        else:
            USE_PRUNE, params_dep = True, params_sp
            print(f"  [prune] applied, ||dU||_2 = {dev_U:.2e}")
params_dep_j = jnp.asarray(params_dep)
PRM = params_dep if USE_PRUNE else None

# ------------------------------------------------------------------ data
print(f"\n  Generating datasets ({N_TEST_STATES} x {len(PI_SWEEP)} p_i) ...")
DATA = {pe: generate_protocol_dataset(N_TEST_STATES, pe) for pe in PI_SWEEP}

P_UNCORR = np.array([arr(DATA[pe]['p_uncorr']).mean()      for pe in PI_SWEEP])
P_UN_ANA = np.array([uncorr_analytic_7q(pe)                for pe in PI_SWEEP])
P_BARE   = np.array([P_bare_physical(pe)                   for pe in PI_SWEEP])
F_RAW_NC = np.array([arr(DATA[pe]['f_raw_nocorr']).mean()  for pe in PI_SWEEP])

if VALIDATE:
    for pe in PI_SWEEP:
        soft(f"analytic vs simulated baseline @p_i={pe:.3f}",
             getattr(C, 'check_uncorr_against_analytic', None), DATA[pe])

print("\n    p_i    P_uncorr(b)  analytic    P_bare(c)   F_raw_1half")
for i, pe in enumerate(PI_SWEEP):
    print(f"    {pe:>5.3f} {P_UNCORR[i]:>12.6f} {P_UN_ANA[i]:>9.6f} "
          f"{P_BARE[i]:>11.6f} {F_RAW_NC[i]:>12.6f}")
print(f"    pseudothreshold p_i* = {BE:.4f}  |  max-mixed floor "
      f"<P> = {P_MAXMIX:.3f}")

# ----------------------------------------------------------------- compile
print("\n  Compiling the noisy graph (the expensive step, ONE graph) ...")
t0 = time.time()
qae = make_noisy_qae(L, with_noise=True, prune_params=PRM, verbose=True)
_ = eval_protocol_chunked(qae['protocol_metrics'], params_dep_j,
                          DATA[PI_SWEEP[0]], 0.0, 0.0, CHUNK)
print(f"    ready ({time.time()-t0:.0f}s)  [p1q/p2q traced -> no recompiles]")

C_DEP = float(qae['C_eff'])
try:
    _nr, _nc, _nrz = gate_counts(build_schedule(L))
    C_FULL = float(effective_C(2*_nc, 2*_nr, 2*_nrz))
except Exception:
    C_FULL = C_DEP
print(f"  C_eff deployed = {C_DEP:.0f}   (unpruned {C_FULL:.0f})")

# ------------------------------------------------------------------- sweep
NP, NE = len(P2Q), len(PI_SWEEP)
nan = lambda: np.full((NP, NE), np.nan)
P_mean, P_std, F_raw, S_mean = nan(), nan(), nan(), nan()
DZ, DX, DY, TRdev            = nan(), nan(), nan(), nan()


def save_all(path):
    np.savez(path,
             n_layers=np.array(L), l_target=np.array(L_TARGET),
             best_n_layers_noiseless=np.array(ref['best_n_layers']),
             params_zyz=params_zyz, params_deployed=params_dep,
             params_old=np.asarray(ref['params_old']),
             ket_0L=ket_0L, ket_1L=ket_1L,
             p_ext=np.array(PI_SWEEP), in_ref_grid=IN_REF,
             p2q=np.array(P2Q), p1q_ratio=np.array(P1Q_RATIO),
             pext_train=np.array(PEXT_TRAIN),
             noisy_rz=np.array(int(NOISY_RZ)),
             noisy_reset=np.array(int(NOISY_RESET)),
             p_reset=np.array(P_RESET),
             n_test=np.array(N_TEST_STATES),
             p_corr=P_mean, p_corr_std=P_std,
             p_uncorr=P_UNCORR, p_uncorr_analytic=P_UN_ANA, p_bare=P_BARE,
             p_ideal_qae=P_mean[0],            # the p2q = 0 row
             p_maxmix=np.array(P_MAXMIX), break_even_p_i=np.array(BE),
             f_raw=F_raw, f_raw_nocorr=F_RAW_NC,
             stab=S_mean, dzl=DZ, dxl=DX, dyl=DY, trace_dev=TRdev,
             corr_weight_enum=CORR_WEIGHT_ENUM,
             pruned=np.array(int(USE_PRUNE)),
             i_pext_ctrl=np.array(I_CTRL),
             n_cnot_full=np.array(2*qae['n_cnot_half']),
             n_rot_full=np.array(2*qae['n_rot_half']),
             c_eff=np.array(C_DEP), c_eff_unpruned=np.array(C_FULL),
             validated=np.array(int(VALIDATE)))


def make_plot():
    ok = np.isfinite(P_mean).any(axis=1)
    if not ok.any():
        print("  [plot] no data yet -> skipped"); return
    pi = np.array(PI_SWEEP)
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8.4))

    # (a) quantum-memory quality
    a = ax[0, 0]
    a.plot(pi, P_BARE,   'k-',  lw=2,   label='physical qubit')
    a.plot(pi, P_UNCORR, '--',  c='darkorange', lw=2,
           label='logical, uncorrected')
    cm = plt.cm.viridis(np.linspace(0, .88, NP))
    for ip, p2 in enumerate(P2Q):
        if not ok[ip]: continue
        a.plot(pi, P_mean[ip], 'o-', ms=4, c=cm[ip],
               label=rf'QAE, $p_{{2q}}$={p2:.1e}')
    a.axhline(P_MAXMIX, ls=':', c='gray', lw=1)
    if np.isfinite(BE): a.axvline(BE, ls=':', c='crimson', lw=1)
    a.set_xlabel(r'incoming error probability $p_i$')
    a.set_ylabel(r'$\langle P \rangle$')
    a.set_title('(a) quantum-memory quality')
    a.legend(fontsize=7, ncol=2); a.grid(alpha=.3)

    # (b) advantage over the uncorrected logical qubit
    b = ax[0, 1]
    b.axhline(0, c='k', lw=1)
    for ip, p2 in enumerate(P2Q):
        if not ok[ip]: continue
        b.plot(pi, P_mean[ip]-P_UNCORR, 'o-', ms=4, c=cm[ip],
               label=rf'$p_{{2q}}$={p2:.1e}')
    b.set_xlabel(r'$p_i$')
    b.set_ylabel(r'$\langle P\rangle_{\rm QAE}-\langle P\rangle_{\rm uncorr}$')
    b.set_title('(b) gain vs. uncorrected logical qubit')
    b.legend(fontsize=7); b.grid(alpha=.3)

    # (c) advantage over the bare physical qubit
    c = ax[1, 0]
    c.axhline(0, c='k', lw=1)
    for ip, p2 in enumerate(P2Q):
        if not ok[ip]: continue
        c.plot(pi, P_mean[ip]-P_BARE, 'o-', ms=4, c=cm[ip])
    c.set_xlabel(r'$p_i$')
    c.set_ylabel(r'$\langle P\rangle_{\rm QAE}-\langle P\rangle_{\rm bare}$')
    c.set_title('(c) gain vs. bare physical qubit')
    c.grid(alpha=.3)

    # (d) degradation with the internal gate noise
    d = ax[1, 1]
    p2a = np.array(P2Q)
    for ie, pe in enumerate(PI_SWEEP):
        col = np.isfinite(P_mean[:, ie])
        if col.sum() < 2: continue
        d.plot(p2a[col], P_mean[col, ie], 'o-', ms=4,
               label=rf'$p_i$={pe:.3f}')
    d.axhline(P_MAXMIX, ls=':', c='gray', lw=1)
    d.set_xlabel(r'two-qubit gate error $p_{2q}$')
    d.set_ylabel(r'$\langle P \rangle$')
    d.set_title('(d) internal-noise sensitivity')
    d.legend(fontsize=7); d.grid(alpha=.3)

    fig.suptitle(f'Steane [[7,1,3]] noiseless-trained QAE, L={L}, '
                 f'{N_TEST_STATES} states, '
                 f"{'pruned' if USE_PRUNE else 'full'} circuit", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(FPNG, dpi=160)
    fig.savefig(FPNG.replace('.png', '.pdf'))
    plt.close(fig)
    print(f"  Plot  : {FPNG} / .pdf")


HEADER = ["n_layers", "p2q", "p1q", "p_i", "P_corr", "P_corr_std",
          "P_uncorr_logical", "P_uncorr_analytic", "P_bare_physical",
          "gain_vs_logical", "gain_vs_physical", "F_raw_1half",
          "stab", "dzl", "dxl", "dyl", "pruned", "c_eff"]

print("\n" + "="*78)
print(f"  SWEEP  ({NP} x {NE} = {NP*NE} points)   -> live CSV: {FCSV}")
print("="*78)

fcsv = open(FCSV, "w", newline="")
wcsv = csv.writer(fcsv); wcsv.writerow(HEADER); fcsv.flush()
t_all = time.time()
try:
    for ip, p2 in enumerate(P2Q):
        p1 = P1Q_RATIO*p2
        print(f"\n  p2q={p2:.1e}  p1q={p1:.1e}  "
              f"lambda=exp(-C p2q)={np.exp(-C_DEP*p2):.4f}"
              + ("   [= ideal QAE reference row]" if p2 == 0 else ""))
        print(f"    {'p_i':>6} {'P_corr':>9} {'+-std':>8} {'P_uncorr':>9} "
              f"{'P_bare':>8} {'gain_b':>8} {'gain_c':>8} {'F_raw':>8} {'t':>6}")
        for ie, pe in enumerate(PI_SWEEP):
            t0 = time.time()
            mm = eval_protocol_chunked(qae['protocol_metrics'], params_dep_j,
                                       DATA[pe], p1, p2, CHUNK)
            P_mean[ip, ie] = arr(mm['P']).mean()
            P_std [ip, ie] = arr(mm['P']).std()
            F_raw [ip, ie] = arr(mm['f_raw']).mean()
            S_mean[ip, ie] = arr(mm['stab']).mean()
            DZ[ip, ie] = arr(mm['dzl']).mean()
            DX[ip, ie] = arr(mm['dxl']).mean()
            DY[ip, ie] = arr(mm['dyl']).mean()
            TRdev[ip, ie] = (np.max(np.abs(arr(mm['trace'])-1.0))
                             if 'trace' in mm else np.nan)
            wcsv.writerow([L, f"{p2:.6e}", f"{p1:.6e}", f"{pe:.4f}",
                           f"{P_mean[ip,ie]:.6f}", f"{P_std[ip,ie]:.6f}",
                           f"{P_UNCORR[ie]:.6f}", f"{P_UN_ANA[ie]:.6f}",
                           f"{P_BARE[ie]:.6f}",
                           f"{P_mean[ip,ie]-P_UNCORR[ie]:.6f}",
                           f"{P_mean[ip,ie]-P_BARE[ie]:.6f}",
                           f"{F_raw[ip,ie]:.6f}", f"{S_mean[ip,ie]:.6f}",
                           f"{DZ[ip,ie]:.6f}", f"{DX[ip,ie]:.6f}",
                           f"{DY[ip,ie]:.6f}", int(USE_PRUNE),
                           f"{C_DEP:.3f}"])
            fcsv.flush()
            print(f"    {pe:>6.3f} {P_mean[ip,ie]:>9.5f} {P_std[ip,ie]:>8.5f} "
                  f"{P_UNCORR[ie]:>9.5f} {P_BARE[ie]:>8.5f} "
                  f"{P_mean[ip,ie]-P_UNCORR[ie]:>+8.5f} "
                  f"{P_mean[ip,ie]-P_BARE[ie]:>+8.5f} "
                  f"{F_raw[ip,ie]:>8.5f} {time.time()-t0:>5.1f}s")
        save_all(FPART)                      # <- partial dump after each row
except KeyboardInterrupt:
    print("\n  [interrupted] keeping everything computed so far")
except Exception:
    print("\n  [ERROR] sweep aborted; keeping everything computed so far")
    traceback.print_exc()
finally:
    fcsv.close()
    save_all(FNPZ)
    if os.path.exists(FPART):
        try: os.remove(FPART)
        except OSError: pass
    print(f"\n  Total sweep time : {time.time()-t_all:.0f}s")
    print(f"  finished points  : {int(np.isfinite(P_mean).sum())} / {NP*NE}")
    if np.isfinite(TRdev).any():
        print(f"  max |trace-1|    : {np.nanmax(TRdev):.2e}")
    print(f"  Saved : {FNPZ} / {FCSV}")
    if MAKE_PLOT:
        try:
            make_plot()
        except Exception:
            print("  [WARN] plotting failed:")
            traceback.print_exc()
    print("\n  Done")