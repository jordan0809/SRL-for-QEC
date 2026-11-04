# ============================================================================
# noisy_trained_prune.py  
# [[3,1,3]] phase-flip code, QAE retrained WITH internal gate noise.
# ============================================================================
import time, csv, gc
import numpy as np
import jax, jax.numpy as jnp
import optax
from qae_noisy_common_prune import *

N_TRAIN_STATES = 100
N_TEST_STATES  = 1000
BATCH_SIZE     = 20
CHUNK          = 100
WEIGHT_UPDATE_FREQ = 5
SEED           = 7
OUT_TAG        = "noisy_trained"
USE_PROTOCOL_LOSS  = False      
SWEEP_ONLY_FINAL_L = True       # intermediate stages: only the training point

# ---------------------------------------------------------------- schedule
# Mirrors the noiseless [[3,1,3]] / [[5,1,3]] reference scripts exactly.
LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
TOTAL_EPOCHS       = INIT_EPOCHS + MAIN_EPOCHS   # 200

# ---------------------------------------------------------------- pruning
PRUNE_ENABLE    = True
PR_TOL_ROT      = PRUNE_TOL_ROT
PR_TOL_BLK      = PRUNE_TOL_BLOCK
PR_DEV_WARN     = 5e-3
FT_EPOCHS       = 30
FT_LR           = 0.002
GROUP_L1        = 0.0           # >0: group lasso over (beta,gamma,delta)

TRAIN_ALL_P2Q = True
P2Q_SUBSET    = [1e-4, 1e-3, 1e-2]
P2Q_TRAIN = ([p for p in P2Q_SWEEP if p > 0.0] if TRAIN_ALL_P2Q
             else [p for p in P2Q_SWEEP if any(abs(p-q) < 1e-15
                                               for q in P2Q_SUBSET)])
LAYERS_PATH = LAYERWISE_PATH
L_FINAL     = LAYERS_PATH[-1]

print("="*78)
print("  [[3,1,3]] - NOISY layerwise training")
print("="*78)
print(f"  JAX backend : {jax.default_backend().upper()}")

validate_phi_composition(); validate_recovery(); validate_bare_physical()
assert all(any(abs(p-q) < 1e-15 for q in P2Q_SWEEP) for p in P2Q_TRAIN)

# ---------------------------------------------------------------- sigma grid
try:
    SIGMA_SWEEP = list(SIGMA_SWEEP)
    print(f"  SIGMA_SWEEP sourced from qae_noisy_common_prune: {SIGMA_SWEEP}")
except NameError:
    SIGMA_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]
    print(f"  SIGMA_SWEEP not found in qae_noisy_common_prune "
          f"-> using local default grid: {SIGMA_SWEEP}")

print(f"  layer path      : {LAYERS_PATH}   (L_final={L_FINAL})")
print(f"  schedule        : Phase1 = {N_INIT_ROUNDS}x{INIT_EPOCHS} multi-start "
      f"(L={LAYERS_PATH[0]}) OR {INIT_EPOCHS} warm-refine (L>{LAYERS_PATH[0]})  "
      f"|  Phase2 = {MAIN_EPOCHS}  |  LR={LR}  |  TOTAL_EPOCHS={TOTAL_EPOCHS}")
print(f"  train sigma_i   : {SIGMA_TRAIN}   (per HALF interval)")
print(f"  loss            : "
      f"{'PROTOCOL (1-P)' if USE_PROTOCOL_LOSS else 'denoiser / spectral'}"
      f"{'' if GROUP_L1 <= 0 else f' + group_l1={GROUP_L1:.1e}'}")
print(f"  pruning         : {'ON' if PRUNE_ENABLE else 'OFF'} "
      f"(tol_rot={PR_TOL_ROT:.0e}, tol_block={PR_TOL_BLK:.0e})")
print(f"  p2q training set: {[f'{p:.0e}' for p in P2Q_TRAIN]}")

# ---------------------------------------------------------------- data
TR = generate_protocol_dataset(N_TRAIN_STATES, SIGMA_TRAIN)
print(f"\n  Generating protocol test sets "
      f"({N_TEST_STATES} x {len(SIGMA_SWEEP)} sigma_i) ...")
DATA = {sg: generate_protocol_dataset(N_TEST_STATES, sg) for sg in SIGMA_SWEEP}
P_UNCORR = np.array([DATA[sg]['p_uncorr'].mean() for sg in SIGMA_SWEEP])
P_BARE   = np.array([P_bare_physical(sg)         for sg in SIGMA_SWEEP])
I_TRAIN  = int(np.argmin(np.abs(np.array(SIGMA_SWEEP) - SIGMA_TRAIN)))
print(f"  done.  training index in sweep: {I_TRAIN} "
      f"(sigma_i={SIGMA_SWEEP[I_TRAIN]:.3f})")

# ---------------------------------------------------------------- helpers
_H_KEYS = ['epoch', 'phase', 'loss', 'loss_std', 'P', 'P_std', 'fid',
           'fid_std', 'stab', 'dzl', 'dxl', 'dyl', 'gnorm']
_PROBE_KEYS = ['loss', 'loss_std', 'P', 'P_std', 'fid', 'fid_std', 'stab',
               'dzl', 'dxl', 'dyl']
_SW_KEYS = ('P', 'P_std', 'f_raw', 'stab', 'dzl', 'dxl', 'dyl')

def probe(comps, params):
    m = eval_protocol_chunked(comps['protocol_metrics'], params, TR, CHUNK)
    bl = baseline_loss_from_metrics(m['stab'], m['dzl'], m['dxl'], m['dyl'])
    return dict(loss=bl.mean(), loss_std=bl.std(),
                P=m['P'].mean(), P_std=m['P'].std(),
                fid=m['f_raw'].mean(), fid_std=m['f_raw'].std(),
                stab=m['stab'].mean(), dzl=m['dzl'].mean(),
                dxl=m['dxl'].mean(), dyl=m['dyl'].mean())

def protocol_sweep(comps, params):
    out = {k: np.zeros(len(SIGMA_SWEEP)) for k in _SW_KEYS}
    print(f"      {'sig_i':>6} {'P_corr':>9} {'+-std':>8} {'P_uncorr':>9} "
          f"{'P_bare':>8} {'F_raw':>8}")
    for i, sg in enumerate(SIGMA_SWEEP):
        m = eval_protocol_chunked(comps['protocol_metrics'], params,
                                  DATA[sg], CHUNK)
        out['P'][i] = m['P'].mean(); out['P_std'][i] = m['P'].std()
        out['f_raw'][i] = m['f_raw'].mean(); out['stab'][i] = m['stab'].mean()
        out['dzl'][i] = m['dzl'].mean(); out['dxl'][i] = m['dxl'].mean()
        out['dyl'][i] = m['dyl'].mean()
        print(f"      {sg:>6.2f} {out['P'][i]:>9.5f} {out['P_std'][i]:>8.5f} "
              f"{P_UNCORR[i]:>9.5f} {P_BARE[i]:>8.5f} {out['f_raw'][i]:>8.5f}")
    return out

def protocol_single(comps, params, i):
    out = {k: np.full(len(SIGMA_SWEEP), np.nan) for k in _SW_KEYS}
    m = eval_protocol_chunked(comps['protocol_metrics'], params,
                              DATA[SIGMA_SWEEP[i]], CHUNK)
    out['P'][i] = m['P'].mean(); out['P_std'][i] = m['P'].std()
    out['f_raw'][i] = m['f_raw'].mean(); out['stab'][i] = m['stab'].mean()
    out['dzl'][i] = m['dzl'].mean(); out['dxl'][i] = m['dxl'].mean()
    out['dyl'][i] = m['dyl'].mean()
    print(f"      warm-start stage: sigma_i={SIGMA_SWEEP[i]:.2f} only -> "
          f"P={out['P'][i]:.5f} (full sweep skipped)")
    return out

# ---------------------------------------------------------------- run_phase
def run_phase(comps, params, opt_state, optimizer, epochs, init_weights,
              H, phase, rng_key, label):
    """One optimisation phase (multi-start round / warm-refine / Phase-2
    continuation / prune fine-tune). Appends EVERY epoch to H. Returns the
    FINAL-epoch state -- mirrors the noiseless reference scripts, which
    always use the end-of-phase params with no best-epoch tracking."""
    ws, wz, wx, wy = init_weights
    if epochs <= 0:
        return params, opt_state, rng_key, ws, wz, wx, wy
    steps_per_ep = int(np.ceil(N_TRAIN_STATES / BATCH_SIZE))
    ep0 = (H['epoch'][-1] + 1) if H['epoch'] else 0
    print(f"    [{label}] {'Ep':>4} {'BaseLoss':>10} {'P_tr':>8} {'F_raw':>8} "
          f"{'<S>':>6} {'dZL':>7} {'dXL':>7} {'dYL':>7} {'|g|':>9} {'t':>6}")
    for ep in range(epochs):
        t0 = time.time()
        if ep % WEIGHT_UPDATE_FREQ == 0 and ep > 0:
            t_ws, t_wz, t_wx, t_wy = compute_target_weights(H['stab'][-1])
            ws = step_weight(ws, t_ws); wz = step_weight(wz, t_wz)
            wx = step_weight(wx, t_wx); wy = step_weight(wy, t_wy)
        jws, jwz = jnp.float64(ws), jnp.float64(wz)
        jwx, jwy = jnp.float64(wx), jnp.float64(wy)
        rng_key, sk = jax.random.split(rng_key)
        perm = np.array(jax.random.permutation(sk, N_TRAIN_STATES))
        gn_acc = 0.0
        for st in range(steps_per_ep):
            bi = jnp.array(perm[st*BATCH_SIZE:(st+1)*BATCH_SIZE])
            if USE_PROTOCOL_LOSS:
                lv, gv = comps['protocol_loss_and_grad'](
                    params, TR['rho_half'][bi], TR['phi'], TR['V'][bi])
            else:
                lv, gv = comps['loss_and_grad'](
                    params, TR['rho_half'][bi], TR['tgt'][bi],
                    jws, jwz, jwx, jwy)
            gv = jnp.where(jnp.isnan(gv), 0.0, gv)
            gn_acc += float(jnp.linalg.norm(gv))
            upd, opt_state = optimizer.update(gv, opt_state, params)
            params = optax.apply_updates(params, upd)
        pm = probe(comps, params)
        H['epoch'].append(ep0 + ep); H['phase'].append(int(phase))
        H['gnorm'].append(gn_acc / steps_per_ep)
        for k in _PROBE_KEYS: H[k].append(pm[k])
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"    {ep0+ep:>4} {pm['loss']:>10.3f} {pm['P']:>8.5f} "
                  f"{pm['fid']:>8.5f} {pm['stab']:>6.3f} {pm['dzl']:>7.4f} "
                  f"{pm['dxl']:>7.4f} {pm['dyl']:>7.4f} "
                  f"{H['gnorm'][-1]:>9.2e} {time.time()-t0:>5.1f}s")
        if ep % 20 == 0: gc.collect()
    return params, opt_state, rng_key, ws, wz, wx, wy

# ---------------------------------------------------------------- train_stage
def train_stage(L, p2, prev_params, prev_L, full_sweep=True):
    comps = make_noisy_qae(L, p2q=p2, group_l1=GROUP_L1, verbose=True)
    n_p = count_params(L)
    steps_per_ep = int(np.ceil(N_TRAIN_STATES / BATCH_SIZE))
    base_seed = SEED + 1000*L + int(round(-np.log10(p2)*10))

    print("    JIT warm-up ...", end=" ", flush=True); t0 = time.time()
    _p0 = jnp.zeros(n_p, dtype=jnp.float64)
    if USE_PROTOCOL_LOSS:
        comps['protocol_loss_and_grad'](_p0, TR['rho_half'][:BATCH_SIZE],
                                        TR['phi'], TR['V'][:BATCH_SIZE])
    else:
        comps['loss_and_grad'](_p0, TR['rho_half'][:BATCH_SIZE],
                               TR['tgt'][:BATCH_SIZE], jnp.float64(1.),
                               jnp.float64(1.), jnp.float64(1.), jnp.float64(1.))
    print(f"done {time.time()-t0:.0f}s")

    H = {k: [] for k in _H_KEYS}
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(LR, TOTAL_EPOCHS*steps_per_ep)))

    if prev_params is None:
        # ============== Phase 1: MULTI-START random search, UNDER NOISE ==============
        print(f"\n  Phase 1 : {N_INIT_ROUNDS} rounds x {INIT_EPOCHS} epochs "
              f"-- random init, NOISY training (L={L}, p2q={p2:.1e}) "
              f"-- NO noiseless-trained parameters used")
        print(f"  {'Rd':<3} {'Init style':<20} {'P_end':>8} {'F_raw':>8} {'<S>':>6}")
        rng = jax.random.PRNGKey(base_seed)
        best_P, best_params, best_opt_state = -1.0, None, None
        best_round_hist, best_w, best_rng = None, None, rng

        for rd in range(N_INIT_ROUNDS):
            rng, sk = jax.random.split(rng)
            style = rd % 4
            if   style == 0:
                tp = jax.random.uniform(sk, (n_p,), minval=-np.pi, maxval=np.pi)
                sn = "Uniform[-pi,pi]"
            elif style == 1:
                tp = jax.random.normal(sk, (n_p,)) * 0.15
                sn = "Normal(0,0.15)"
            elif style == 2:
                tp = jax.random.normal(sk, (n_p,)) * 1.0
                sn = "Normal(0,1.0)"
            else:
                tp = jax.random.uniform(sk, (n_p,), minval=-0.1, maxval=0.1)
                sn = "Unif[-0.1,0.1]"
            tp = jnp.array(tp, dtype=jnp.float64)
            opt0 = optimizer.init(tp)
            rng, rr = jax.random.split(rng)
            Hr = {k: [] for k in _H_KEYS}
            tp2, os2, rr, ws, wz, wx, wy = run_phase(
                comps, tp, opt0, optimizer, INIT_EPOCHS,
                (W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW),
                Hr, 0, rr, f"ms-rd{rd}")
            P_end = Hr['P'][-1]
            print(f"  {rd:<3} {sn:<20} {P_end:>8.5f} {Hr['fid'][-1]:>8.5f} "
                  f"{Hr['stab'][-1]:>6.3f}")
            if P_end > best_P:
                best_P = P_end; best_params = np.array(tp2)
                best_opt_state = os2; best_round_hist = Hr
                best_w = (ws, wz, wx, wy); best_rng = rr
            gc.collect()

        print(f"  Best warm-up (multi-start) P = {best_P:.5f}")
        H = best_round_hist
        params, opt_state = jnp.array(best_params, dtype=jnp.float64), best_opt_state
        ws, wz, wx, wy = best_w
        rng = best_rng
    else:
        # ============== Phase 1: LAYERWISE WARM-START (identity-grown, noisy) =========
        init = grow_params(prev_params, prev_L, L)
        print(f"\n  Phase 1 : LAYERWISE WARM-START -- single {INIT_EPOCHS}-epoch "
              f"refinement, identity-grown L={prev_L} -> L={L} (NOISY training)")
        params = jnp.array(init, dtype=jnp.float64)
        assert params.size == n_p
        param_sparsity_report(np.asarray(init, float).ravel(), L)
        p0 = probe(comps, params)
        print(f"    warm-start point: P={p0['P']:.5f} F_raw={p0['fid']:.5f} "
              f"<S>={p0['stab']:.3f}")
        opt_state = optimizer.init(params)
        rng = jax.random.PRNGKey(base_seed)
        rng, rr = jax.random.split(rng)
        params, opt_state, rr, ws, wz, wx, wy = run_phase(
            comps, params, opt_state, optimizer, INIT_EPOCHS,
            (W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW),
            H, 0, rr, "warm-refine")
        rng = rr
        print(f"  Warm-start refinement P = {H['P'][-1]:.5f}")

    # ============== Phase 2: continuation (same optimizer/schedule) ==============
    print(f"\n  Phase 2 : {MAIN_EPOCHS} continuation epochs "
          f"(global {INIT_EPOCHS} ... {TOTAL_EPOCHS-1})")
    params, opt_state, rng, ws, wz, wx, wy = run_phase(
        comps, params, opt_state, optimizer, MAIN_EPOCHS,
        (ws, wz, wx, wy), H, 1, rng, "train")
    best_params = np.array(params)
    best_P = H['P'][-1]
    print(f"    final protocol P (L={L}, full circuit) = {best_P:.5f}")

    use = dict(comps=comps, params=best_params, P=best_P, prune=None)
    if PRUNE_ENABLE:
        print("    --- prune + fine-tune " + "-"*40)
        param_sparsity_report(best_params, L)
        dev, _ops, pst, sp = validate_pruning(best_params, L,
                                              PR_TOL_ROT, PR_TOL_BLK)
        if prune_is_trivial(pst):
            print("      nothing prunable -> keeping the full circuit")
        else:
            if dev > PR_DEV_WARN:
                print(f"      WARNING: ||U_pruned - U_full||_2 = {dev:.2e} > "
                      f"{PR_DEV_WARN:.0e} -> lower the tolerances")
            comps_p = make_noisy_qae(L, p2q=p2, prune_params=sp,
                                     group_l1=GROUP_L1, verbose=True)
            pp = probe(comps_p, jnp.array(sp))
            print(f"      P: full {best_P:.5f} -> pruned(warm) {pp['P']:.5f} "
                  f"({pp['P']-best_P:+.5f}) | C_eff {comps['C_eff']:.0f} -> "
                  f"{comps_p['C_eff']:.0f}")
            ft_optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(optax.cosine_decay_schedule(
                    FT_LR, FT_EPOCHS*steps_per_ep)))
            sp_j = jnp.array(sp, dtype=jnp.float64)
            ft_opt_state = ft_optimizer.init(sp_j)
            ft_ws, ft_wz, ft_wx, ft_wy = compute_target_weights(pp['stab'])
            rng, rr = jax.random.split(rng)
            params_ft, ft_opt_state, rr, _, _, _, _ = run_phase(
                comps_p, sp_j, ft_opt_state, ft_optimizer, FT_EPOCHS,
                (ft_ws, ft_wz, ft_wx, ft_wy), H, 2, rr, "finetune")
            rng = rr
            bP = H['P'][-1]; bp = np.array(params_ft)
            pst['P_pruned_warm'] = float(pp['P'])
            pst['P_pruned_best'] = float(bP)
            if bP >= best_P:
                del comps; gc.collect()
                use = dict(comps=comps_p, params=bp, P=bP, prune=pst)
                print(f"      -> keeping the PRUNED circuit (P={bP:.5f}, "
                      f"{pst['n_cnot_half']}/{pst['n_cnot_half_full']} "
                      f"CNOT/half)")
            else:
                del comps_p; gc.collect()
                print(f"      -> keeping the FULL circuit (pruned {bP:.5f} "
                      f"< {best_P:.5f})")

    ncnot = 2*use['comps']['n_cnot_half']
    nrot  = 2*use['comps']['n_rot_half']
    ceff  = use['comps']['C_eff']
    sw = (protocol_sweep(use['comps'], jnp.array(use['params'])) if full_sweep
          else protocol_single(use['comps'], jnp.array(use['params']), I_TRAIN))
    pst = use['prune']
    del use['comps']; gc.collect()
    return dict(params=use['params'], hist=H, P_train_best=use['P'],
                n_cnot_full=ncnot, n_rot_full=nrot,
                c_eff=ceff, swept_full=bool(full_sweep),
                pruned=int(pst is not None),
                n_blocks=int(pst['n_blocks']) if pst else -1,
                n_blocks_kept=int(pst['n_blocks_kept']) if pst else -1,
                prune_dev=float(pst['unitary_dev']) if pst else 0.0, **sw)

# ---------------------------------------------------------------- run
RES = {}
t_all = time.time()
for p2 in P2Q_TRAIN:
    print("\n" + "="*78)
    print(f"  p2q = {p2:.1e}   (layerwise path {LAYERS_PATH})")
    print("="*78)
    RES[p2] = {}
    prev_params, prev_L = None, None
    for L in LAYERS_PATH:
        print(f"\n  --- stage L={L} " + "-"*54)
        st = train_stage(L, p2, prev_params, prev_L,
                         full_sweep=(L == L_FINAL or not SWEEP_ONLY_FINAL_L))
        RES[p2][L] = st
        prev_params, prev_L = st['params'], L
    pT = RES[p2][L_FINAL]['P'][I_TRAIN]
    p1 = RES[p2][LAYERS_PATH[0]]['P'][I_TRAIN]
    print(f"\n  >>> p2q={p2:.1e} @ sigma_i={SIGMA_SWEEP[I_TRAIN]:.2f}: "
          f"L={L_FINAL} P={pT:.5f} | L={LAYERS_PATH[0]} P={p1:.5f} "
          f"| depth gain {pT-p1:+.5f} | P_uncorr={P_UNCORR[I_TRAIN]:.5f} "
          f"| P_bare={P_BARE[I_TRAIN]:.5f} "
          f"| C_eff={RES[p2][L_FINAL]['c_eff']:.0f}")
print(f"\n  Total training time: {time.time()-t_all:.0f}s "
      f"({(time.time()-t_all)/60:.1f} min)")

# ---------------------------------------------------------------- save
def _key(p): return f"p{p:.0e}".replace("-", "m").replace("+", "")

P_aw  = np.stack([RES[p][L_FINAL]['P']     for p in P2Q_TRAIN])
Ps_aw = np.stack([RES[p][L_FINAL]['P_std'] for p in P2Q_TRAIN])
assert np.isfinite(P_aw).all(), "reported L_FINAL curve has NaNs"

sd = dict(n_layers=np.array(L_FINAL), l_target=np.array(L_TARGET),
          layers_path=np.array(LAYERS_PATH),
          sigma=np.array(SIGMA_SWEEP), p2q=np.array(P2Q_TRAIN),
          p1q_ratio=np.array(P1Q_RATIO), sigma_train=np.array(SIGMA_TRAIN),
          mu=np.array(MU_COHERENT), rho_corr=np.array(RHO_CORR),
          noisy_rz=np.array(int(NOISY_RZ)),
          protocol_loss=np.array(int(USE_PROTOCOL_LOSS)),
          prune_enable=np.array(int(PRUNE_ENABLE)),
          prune_tol_rot=np.array(PR_TOL_ROT),
          prune_tol_block=np.array(PR_TOL_BLK),
          group_l1=np.array(GROUP_L1),
          n_train=np.array(N_TRAIN_STATES), n_test=np.array(N_TEST_STATES),
          n_init_rounds=np.array(N_INIT_ROUNDS),
          init_epochs=np.array(INIT_EPOCHS), main_epochs=np.array(MAIN_EPOCHS),
          p_aware=P_aw, p_aware_std=Ps_aw,
          p_uncorr=P_UNCORR, p_bare=P_BARE,
          i_sigma_train=np.array(I_TRAIN),
          f_raw=np.stack([RES[p][L_FINAL]['f_raw'] for p in P2Q_TRAIN]),
          stab=np.stack([RES[p][L_FINAL]['stab'] for p in P2Q_TRAIN]),
          dzl=np.stack([RES[p][L_FINAL]['dzl'] for p in P2Q_TRAIN]),
          dxl=np.stack([RES[p][L_FINAL]['dxl'] for p in P2Q_TRAIN]),
          dyl=np.stack([RES[p][L_FINAL]['dyl'] for p in P2Q_TRAIN]),
          # pruning makes the gate count parameter dependent -> per-p2q arrays;
          # the legacy scalars keep old plotting scripts alive
          n_cnot_full=np.array(RES[P2Q_TRAIN[0]][L_FINAL]['n_cnot_full']),
          n_rot_full=np.array(RES[P2Q_TRAIN[0]][L_FINAL]['n_rot_full']),
          c_eff=np.array(RES[P2Q_TRAIN[0]][L_FINAL]['c_eff']),
          sweep_only_final=np.array(int(SWEEP_ONLY_FINAL_L)),
          stage_sweep_full=np.array(
              [int(RES[P2Q_TRAIN[0]][L]['swept_full']) for L in LAYERS_PATH]))
for L in LAYERS_PATH:
    sd[f"p_stage_L{L}"]      = np.stack([RES[p][L]['P']     for p in P2Q_TRAIN])
    sd[f"p_stage_L{L}_std"]  = np.stack([RES[p][L]['P_std'] for p in P2Q_TRAIN])
    sd[f"p_train_best_L{L}"] = np.array([RES[p][L]['P_train_best']
                                         for p in P2Q_TRAIN])
    for k in ('n_cnot_full', 'n_rot_full', 'c_eff', 'pruned',
              'n_blocks', 'n_blocks_kept', 'prune_dev'):
        sd[f"{k}_L{L}"] = np.array([RES[p][L][k] for p in P2Q_TRAIN])
for p in P2Q_TRAIN:
    for L in LAYERS_PATH:
        sd[f"{_key(p)}_L{L}_params"] = RES[p][L]['params']
        for k, v in RES[p][L]['hist'].items():
            sd[f"{_key(p)}_L{L}_hist_{k}"] = np.array(v)
np.savez(f"internal_{OUT_TAG}.npz", **sd)

with open(f"internal_{OUT_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    hdr = ["p2q", "p1q", "sigma_i", "P_corr", "P_corr_std",
           "P_uncorr_logical", "P_bare_physical", "gain_vs_logical",
           "gain_vs_physical", "F_raw_1round", "stab", "dzl", "dxl", "dyl",
           "pruned", "n_cnot_full", "n_rot_full", "c_eff",
           "n_blocks_kept", "n_blocks", "prune_unitary_dev"]
    hdr += [f"P_L{L}" for L in LAYERS_PATH]
    w.writerow(hdr)
    for ip, p2 in enumerate(P2Q_TRAIN):
        R = RES[p2][L_FINAL]
        for i, sg in enumerate(SIGMA_SWEEP):
            row = [f"{p2:.6e}", f"{P1Q_RATIO*p2:.6e}", f"{sg:.4f}",
                   f"{P_aw[ip,i]:.6f}", f"{Ps_aw[ip,i]:.6f}",
                   f"{P_UNCORR[i]:.6f}", f"{P_BARE[i]:.6f}",
                   f"{P_aw[ip,i]-P_UNCORR[i]:.6f}",
                   f"{P_aw[ip,i]-P_BARE[i]:.6f}",
                   f"{R['f_raw'][i]:.6f}", f"{R['stab'][i]:.6f}",
                   f"{R['dzl'][i]:.6f}", f"{R['dxl'][i]:.6f}",
                   f"{R['dyl'][i]:.6f}",
                   R['pruned'], R['n_cnot_full'], R['n_rot_full'],
                   f"{R['c_eff']:.3f}", R['n_blocks_kept'], R['n_blocks'],
                   f"{R['prune_dev']:.3e}"]
            row += ["" if not np.isfinite(RES[p2][L]['P'][i])
                    else f"{RES[p2][L]['P'][i]:.6f}" for L in LAYERS_PATH]
            w.writerow(row)

with open(f"internal_{OUT_TAG}_history.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p2q", "n_layers", "epoch", "phase", "baseline_loss",
                "baseline_loss_std", "P_protocol", "P_protocol_std",
                "F_raw", "F_raw_std", "stab", "dzl", "dxl", "dyl",
                "grad_norm"])
    for p2 in P2Q_TRAIN:
        for L in LAYERS_PATH:
            H = RES[p2][L]['hist']
            for i in range(len(H['epoch'])):
                w.writerow([f"{p2:.6e}", L, H['epoch'][i], H['phase'][i]] +
                           [f"{H[k][i]:.6f}" for k in
                            ['loss','loss_std','P','P_std','fid','fid_std',
                             'stab','dzl','dxl','dyl','gnorm']])
print(f"  Saved : internal_{OUT_TAG}.npz / .csv / _history.csv")
print("\n  Done")