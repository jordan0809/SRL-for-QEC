# ============================================================================
# noiseless_trained_prune.py   
# noiseless-trained parameters, deployed WITH internal gate noise.
#
#   N_{sigma_i} -> noisy QAE(params from p2q=0 training) -> N_{sigma_i}
#               -> perfect QEC
# ============================================================================
import time, csv
import numpy as np
import jax.numpy as jnp
from qae_noisy_common_prune import *

N_TEST_STATES = 1000
CHUNK         = 100
OUT_TAG       = "noiseless_trained"
L_OVERRIDE    = None

PRUNE_ENABLE        = True
PR_TOL_ROT          = PRUNE_TOL_ROT
PR_TOL_BLK          = PRUNE_TOL_BLOCK
PR_DEV_WARN         = 5e-3
PRUNE_CONTROL_POINT = True

print("="*78)
print("  [[3,1,3]] - noiseless-trained QAE in the memory protocol")
print("="*78)
print(f"  JAX backend : {jax.default_backend().upper()}")

# ---------------------------------------------------------- protocol checks
validate_phi_composition()
validate_recovery()
validate_bare_physical()

ref = load_noiseless_reference(l_override=L_OVERRIDE)
L   = ref['n_layers']
SIGMA_SWEEP = list(ref['sigma'])          # <- reinterpreted as sigma_i
print(f"\n  Loaded noiseless model : L={L} ({len(ref['params_old'])} params) "
      f"| best L in npz = {ref['best_n_layers']}")
print(f"  sigma_i grid : {SIGMA_SWEEP}   (dephasing per HALF interval)")
print(f"  p2q sweep    : {P2Q_SWEEP}   (p1q = {P1Q_RATIO} * p2q)")
print(f"  NOISY_RZ={NOISY_RZ}  NOISY_RESET={NOISY_RESET}")
print(f"  pruning      : {'ON' if PRUNE_ENABLE else 'OFF'} "
      f"(tol_rot={PR_TOL_ROT:.0e}, tol_block={PR_TOL_BLK:.0e})")

err_dec, params_zyz = validate_decomposition(ref['params_old'], L)
if err_dec > 1e-11:
    raise SystemExit(f"ABORT: decomposition mismatch {err_dec:.3e}")
params_j = jnp.array(params_zyz)

# ---------------------------------------------------------------- pruning
print("\n  [prunability of the noiseless-trained parameters]")
param_sparsity_report(params_zyz, L)
USE_PRUNE, PRUNE_ST = False, None
params_dep = params_zyz
if PRUNE_ENABLE:
    dev, ops_p, PRUNE_ST, params_sp = validate_pruning(
        params_zyz, L, PR_TOL_ROT, PR_TOL_BLK)
    if prune_is_trivial(PRUNE_ST):
        print("  [prune] nothing is prunable -> deploying the full circuit")
    else:
        if dev > PR_DEV_WARN:
            print(f"  [prune] WARNING: unitary deviation {dev:.2e} > "
                  f"{PR_DEV_WARN:.0e} -> lower the tolerances")
        USE_PRUNE = True
        params_dep = params_sp
params_dep_j = jnp.array(params_dep)
PRM = params_dep if USE_PRUNE else None

# ---------------------------------------------------------------- datasets
print(f"\n  Generating protocol test sets "
      f"({N_TEST_STATES} states x {len(SIGMA_SWEEP)} sigma_i) ...")
DATA = {sg: generate_protocol_dataset(N_TEST_STATES, sg) for sg in SIGMA_SWEEP}
P_UNCORR = np.array([DATA[sg]['p_uncorr'].mean() for sg in SIGMA_SWEEP])
P_BARE   = np.array([P_bare_physical(sg)          for sg in SIGMA_SWEEP])
F_RAW_NC = np.array([DATA[sg]['f_raw_nocorr'].mean() for sg in SIGMA_SWEEP])
I_CTRL   = int(np.argmin(np.abs(np.array(SIGMA_SWEEP) - SIGMA_TRAIN)))
print("  done.")
print("\n    sigma_i   P_uncorr(b)   P_bare(c)   F_raw_1round(legacy)")
for i, sg in enumerate(SIGMA_SWEEP):
    print(f"    {sg:>7.3f} {P_UNCORR[i]:>13.6f} {P_BARE[i]:>11.6f} "
          f"{F_RAW_NC[i]:>16.6f}")

# ------------------------------------ validation: p2q=0 legacy raw fidelity
c0 = make_noisy_qae(L, p2q=0.0, p1q=0.0)          # FULL schedule, always
F0 = np.zeros(len(SIGMA_SWEEP)); P0 = np.zeros(len(SIGMA_SWEEP))
print("\n  [validate] p2q=0 forward map (noiseless limit, FULL schedule) ...")
t0 = time.time()
for i, sg in enumerate(SIGMA_SWEEP):
    m = eval_protocol_chunked(c0['protocol_metrics'], params_j, DATA[sg], CHUNK)
    F0[i] = m['f_raw'].mean(); P0[i] = m['P'].mean()
    tr = np.max(np.abs(m['trace'] - 1.0))
    assert tr < 1e-10, f"trace violation {tr:.2e} at sigma_i={sg}"

if ref['f_qae_is_target_L']:
    worst = float(np.max(np.abs(F0 - ref['f_noiseless_qae'])))
    print(f"    legacy raw-fidelity reproduction: max |diff| = {worst:.3e} "
          f"({time.time()-t0:.0f}s)")
    if worst > 1e-6:
        raise SystemExit("ABORT: p2q=0 simulator does not reproduce the stored "
                         "noiseless curve")
    print("  VALIDATION PASSED")
else:
    print(f"    [skip] stored f_qae belongs to L={ref['best_n_layers']}; "
          f"cross-check at sigma_i={SIGMA_SWEEP[I_CTRL]:.2f}: "
          f"F_new={F0[I_CTRL]:.6f} vs stored "
          f"{ref['f_test_at_train_sigma'][L]:.6f}")

print(f"\n  Full  QAE: {2*c0['n_cnot_half']} CNOT, {2*c0['n_rot_half']} rot "
      f"| C_eff={c0['C_eff']:.0f}")

# deployed (possibly pruned) p2q=0 reference
c0_dep = c0 if not USE_PRUNE else make_noisy_qae(L, p2q=0.0, p1q=0.0,
                                                prune_params=PRM)
if USE_PRUNE:
    print(f"  Pruned QAE: {2*c0_dep['n_cnot_half']} CNOT, "
          f"{2*c0_dep['n_rot_half']} rot | C_eff={c0_dep['C_eff']:.0f} "
          f"({100*(1-c0_dep['C_eff']/c0['C_eff']):.1f}% less)")

# ---------------------------------------------------------------- sweep
NP, NS = len(P2Q_SWEEP), len(SIGMA_SWEEP)
P_mean = np.zeros((NP, NS)); P_std = np.zeros((NP, NS))
F_raw  = np.zeros((NP, NS)); S_mean = np.zeros((NP, NS))
DZ = np.zeros((NP, NS)); DX = np.zeros((NP, NS)); DY = np.zeros((NP, NS))
TRdev = np.zeros((NP, NS)); P_CTRL = np.full(NP, np.nan)

print("\n" + "="*78)
print(f"  PROTOCOL SWEEP  (noiseless-trained, L={L}, "
      f"{'PRUNED' if USE_PRUNE else 'full'} circuit)")
print("="*78)
t_all = time.time()
for ip, p2 in enumerate(P2Q_SWEEP):
    comps = (c0_dep if p2 == 0.0 else
             make_noisy_qae(L, p2q=p2, prune_params=PRM, verbose=True))
    print(f"\n  p2q={p2:.1e}  p1q={comps['p1q']:.1e}  C_eff={comps['C_eff']:.0f}")
    print(f"    {'sig_i':>6} {'P_corr':>9} {'+-std':>8} {'P_uncorr':>9} "
          f"{'P_bare':>8} {'gain_b':>8} {'gain_c':>8} {'F_raw':>8} {'t':>6}")
    for isg, sg in enumerate(SIGMA_SWEEP):
        t0 = time.time()
        m = eval_protocol_chunked(comps['protocol_metrics'], params_dep_j,
                                  DATA[sg], CHUNK)
        P_mean[ip, isg] = m['P'].mean();  P_std[ip, isg] = m['P'].std()
        F_raw[ip, isg]  = m['f_raw'].mean(); S_mean[ip, isg] = m['stab'].mean()
        DZ[ip, isg] = m['dzl'].mean(); DX[ip, isg] = m['dxl'].mean()
        DY[ip, isg] = m['dyl'].mean()
        TRdev[ip, isg] = np.max(np.abs(m['trace'] - 1.0))
        print(f"    {sg:>6.2f} {P_mean[ip,isg]:>9.5f} {P_std[ip,isg]:>8.5f} "
              f"{P_UNCORR[isg]:>9.5f} {P_BARE[isg]:>8.5f} "
              f"{P_mean[ip,isg]-P_UNCORR[isg]:>+8.5f} "
              f"{P_mean[ip,isg]-P_BARE[isg]:>+8.5f} "
              f"{F_raw[ip,isg]:>8.5f} {time.time()-t0:>5.1f}s")
    # ---- control point: SAME unitary, FULL (unpruned) gate list -----------
    if PRUNE_CONTROL_POINT and USE_PRUNE and p2 > 0:
        cf = make_noisy_qae(L, p2q=p2)
        mf = eval_protocol_chunked(cf['protocol_metrics'], params_dep_j,
                                   DATA[SIGMA_SWEEP[I_CTRL]], CHUNK)
        P_CTRL[ip] = mf['P'].mean()
        print(f"    [prune gain @sigma_i={SIGMA_SWEEP[I_CTRL]:.2f}] "
              f"full {P_CTRL[ip]:.5f} -> pruned {P_mean[ip,I_CTRL]:.5f} "
              f"({P_mean[ip,I_CTRL]-P_CTRL[ip]:+.5f})")
        del cf
    if p2 > 0: del comps
print(f"\n  Total sweep time: {time.time()-t_all:.0f}s")
print(f"  max |trace-1| over everything: {TRdev.max():.2e}")

# ---------------------------------------------------------------- save
i_p0 = int(np.argmin(np.abs(np.array(P2Q_SWEEP))))
np.savez(f"internal_{OUT_TAG}.npz",
         n_layers=np.array(L), l_target=np.array(L_TARGET),
         best_n_layers_noiseless=np.array(ref['best_n_layers']),
         params_zyz=params_zyz, params_deployed=params_dep,
         params_old=np.asarray(ref['params_old']),
         sigma=np.array(SIGMA_SWEEP), p2q=np.array(P2Q_SWEEP),
         p1q_ratio=np.array(P1Q_RATIO), sigma_train=np.array(SIGMA_TRAIN),
         mu=np.array(MU_COHERENT), rho_corr=np.array(RHO_CORR),
         noisy_rz=np.array(int(NOISY_RZ)), n_test=np.array(N_TEST_STATES),
         p_corr=P_mean, p_corr_std=P_std,
         p_uncorr=P_UNCORR, p_bare=P_BARE,
         p_ideal_qae=P_mean[i_p0], p_ideal_qae_fullcirc=P0,
         f_raw=F_raw, f_raw_p0=F0, f_raw_nocorr=F_RAW_NC,
         stab=S_mean, dzl=DZ, dxl=DX, dyl=DY, trace_dev=TRdev,
         # ---- pruning bookkeeping ----
         pruned=np.array(int(USE_PRUNE)),
         prune_tol_rot=np.array(PR_TOL_ROT),
         prune_tol_block=np.array(PR_TOL_BLK),
         prune_unitary_dev=np.array(0.0 if PRUNE_ST is None
                                    else PRUNE_ST['unitary_dev']),
         n_blocks=np.array(-1 if PRUNE_ST is None else PRUNE_ST['n_blocks']),
         n_blocks_kept=np.array(-1 if PRUNE_ST is None
                                else PRUNE_ST['n_blocks_kept']),
         i_sigma_ctrl=np.array(I_CTRL), p_ctrl_fullcirc=P_CTRL,
         n_cnot_full=np.array(2*c0_dep['n_cnot_half']),
         n_rot_full=np.array(2*c0_dep['n_rot_half']),
         c_eff=np.array(c0_dep['C_eff']),
         n_cnot_full_unpruned=np.array(2*c0['n_cnot_half']),
         n_rot_full_unpruned=np.array(2*c0['n_rot_half']),
         c_eff_unpruned=np.array(c0['C_eff']))

with open(f"internal_{OUT_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "p2q", "p1q", "sigma_i", "P_corr", "P_corr_std",
                "P_uncorr_logical", "P_bare_physical",
                "gain_vs_logical", "gain_vs_physical",
                "F_raw_1round", "stab", "dzl", "dxl", "dyl",
                "pruned", "n_cnot_full", "n_rot_full", "c_eff",
                "P_corr_fullcirc_ctrl"])
    for ip, p2 in enumerate(P2Q_SWEEP):
        for isg, sg in enumerate(SIGMA_SWEEP):
            w.writerow([L, f"{p2:.6e}", f"{P1Q_RATIO*p2:.6e}", f"{sg:.4f}",
                        f"{P_mean[ip,isg]:.6f}", f"{P_std[ip,isg]:.6f}",
                        f"{P_UNCORR[isg]:.6f}", f"{P_BARE[isg]:.6f}",
                        f"{P_mean[ip,isg]-P_UNCORR[isg]:.6f}",
                        f"{P_mean[ip,isg]-P_BARE[isg]:.6f}",
                        f"{F_raw[ip,isg]:.6f}", f"{S_mean[ip,isg]:.6f}",
                        f"{DZ[ip,isg]:.6f}", f"{DX[ip,isg]:.6f}",
                        f"{DY[ip,isg]:.6f}", int(USE_PRUNE),
                        2*c0_dep['n_cnot_half'], 2*c0_dep['n_rot_half'],
                        f"{c0_dep['C_eff']:.3f}",
                        "" if (isg != I_CTRL or not np.isfinite(P_CTRL[ip]))
                        else f"{P_CTRL[ip]:.6f}"])
print(f"  Saved : internal_{OUT_TAG}.npz / .csv")
print("\n  Done")