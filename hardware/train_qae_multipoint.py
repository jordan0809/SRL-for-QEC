"""train_qae_multipoint.py -- train one QAE per (mu, sigma, gamma) operating
point, matched to a tau/2 hardware sweep.

Usage:
    python train_qae_multipoint.py \
        --tau-list 0,2,4,6,8,12,16,24 \
        --measured-csv cal/hw_deploy_cal.csv \
        --rho-corr 0.05 --L 2 --out dephasing_multipoint_L2.npz
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, gc, csv, argparse
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

import jax
jax.config.update("jax_enable_x64", True)
import logging; logging.getLogger("jax").setLevel(logging.WARNING)
try:
    jax.config.update("jax_platform_name", "gpu"); jax.devices()
except Exception:
    jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import optax
import pennylane as qml
import numpy as np
from functools import reduce

print(f"  JAX backend : {jax.default_backend().upper()}")

# ================================================================
# 1. Global constants 
# ================================================================
N_CODE, N_ANC = 3, 1
N_TOTAL = N_CODE + N_ANC
SYS_DIM, ANC_DIM = 2**N_CODE, 2**N_ANC

N_TRAIN_STATES = 100
N_TEST_STATES  = 1000
BATCH_SIZE     = 5
CHUNK_EVAL_SIZE = 200

LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
WEIGHT_UPDATE_FREQ = 5

W_STAB_HIGH, W_STAB_LOW = 5.0, 0.5
W_ZL_LOW, W_ZL_HIGH = 5.0, 20.0
W_XL_LOW, W_XL_HIGH = 5.0, 20.0
W_YL_LOW, W_YL_HIGH = 5.0, 20.0
STAB_SAT_LOW, STAB_SAT_HIGH = 0.70, 0.99
MAX_W_DELTA = 1.0

W_STAB_BASE = 1.0
W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

# ================================================================
# 2. Pauli primitives + phase-flip code operators
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]]); Z_ = np.array([[1, 0], [0, -1]], dtype=complex)
P1_ = np.array([[0, 0], [0, 1]], dtype=complex)
SM_ = np.array([[0, 1], [0, 0]], dtype=complex)

def kron_list(ops): return reduce(np.kron, ops)
def op_on(qs, mat, n=N_CODE):
    ops = [I2]*n
    for q in qs: ops[q] = mat
    return kron_list(ops)

S1_mat = kron_list([X_, X_, I2]); S2_mat = kron_list([I2, X_, X_])
XL_mat = kron_list([X_, X_, X_]); ZL_mat = kron_list([Z_, Z_, Z_])
YL_mat = kron_list([Y_, Y_, Y_])
S1j, S2j = jnp.array(S1_mat), jnp.array(S2_mat)
L_STACK = jnp.array(np.stack([ZL_mat, XL_mat, YL_mat]))

def expect_stack(op_stack, rho):
    return jnp.real(jnp.einsum('kij,ji->k', op_stack, rho))

ket_p = np.array([1, 1], dtype=complex)/np.sqrt(2)
ket_m = np.array([1, -1], dtype=complex)/np.sqrt(2)
ket_0L = kron_list([ket_p, ket_p, ket_p]).flatten()
ket_1L = kron_list([ket_m, ket_m, ket_m]).flatten()

def logical_state(th, ph):
    return np.cos(th/2)*ket_0L + np.exp(1j*ph)*np.sin(th/2)*ket_1L
def state_to_dm(psi):
    psi = psi.reshape(-1, 1); return psi @ psi.conj().T

ANC_0 = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0, 0].set(1.0)
FRESH_000 = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0, 0].set(1.0)

I8 = np.eye(SYS_DIM, dtype=complex)
P_CODE = ((I8 + S1_mat)/2) @ ((I8 + S2_mat)/2)
TR_P = float(np.real(np.trace(P_CODE)))

# ================================================================
# 3. Combined noise channel AD(gamma) o Dephase(mu,sigma,rho_corr)
# ================================================================
_zvecs = np.array([[1 - 2*((b >> (N_CODE-1-i)) & 1) for i in range(N_CODE)]
                   for b in range(SYS_DIM)], dtype=float)

def build_phi(sigma, rho_corr, mu):
    M = np.eye(N_CODE)*(1-rho_corr) + rho_corr*np.ones((N_CODE, N_CODE))
    Sigma = (sigma**2)*M
    mu_vec = np.full(N_CODE, mu)
    C_ = (_zvecs[:, None, :] - _zvecs[None, :, :]) / 2.0
    lin = C_ @ mu_vec
    quad = np.einsum('abi,ij,abj->ab', C_, Sigma, C_)
    return np.exp(-1j*lin - 0.5*quad)

_P1_EMB = [jnp.array(op_on([q], P1_)) for q in range(N_CODE)]
_SM_EMB = [jnp.array(op_on([q], SM_)) for q in range(N_CODE)]
_I_SYS = jnp.eye(SYS_DIM, dtype=complex)

@jax.jit
def combined_channel(rho, gamma, phi):
    rho = phi * rho
    s = jnp.sqrt(1.0 - gamma); sg = jnp.sqrt(gamma)
    for q in range(N_CODE):
        K0 = _I_SYS - (1.0 - s) * _P1_EMB[q]
        K1 = sg * _SM_EMB[q]
        rho = K0 @ rho @ jnp.conj(K0.T) + K1 @ rho @ jnp.conj(K1.T)
    return rho

@jax.jit
def combined_adjoint(A, gamma, phi_conj):
    s = jnp.sqrt(1.0 - gamma); sg = jnp.sqrt(gamma)
    for q in range(N_CODE):
        K0 = _I_SYS - (1.0 - s) * _P1_EMB[q]
        K1 = sg * _SM_EMB[q]
        A = jnp.conj(K0.T) @ A @ K0 + jnp.conj(K1.T) @ A @ K1
    return phi_conj * A

@jax.jit
def _chan_batch(rho_b, gamma, phi):
    return jax.vmap(lambda r: combined_channel(r, gamma, phi))(rho_b)

def make_noisy_stack(clean_np, gamma, phi, chunk=CHUNK_EVAL_SIZE):
    gj, phij = jnp.float64(gamma), jnp.array(phi)
    n = clean_np.shape[0]; out = np.empty_like(clean_np)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out[sl] = np.array(_chan_batch(jnp.array(clean_np[sl]), gj, phij))
    return out

# ================================================================
# 4. Standard QEC (phase-flip)
# ================================================================
Z0 = kron_list([Z_, I2, I2]); Z1 = kron_list([I2, Z_, I2]); Z2 = kron_list([I2, I2, Z_])

def _proj(s1, s2):
    return ((I8 + s1*S1_mat)/2) @ ((I8 + s2*S2_mat)/2)

_STD_CORR = [(_proj(+1, +1), I8),
             (_proj(-1, +1), Z0),
             (_proj(-1, -1), Z1),
             (_proj(+1, -1), Z2)]
CORR_PI = jnp.array(np.stack([Pi for Pi, _ in _STD_CORR]))
CORR_R  = jnp.array(np.stack([R  for _, R in _STD_CORR]))

@jax.jit
def _std_fid_batch(rn, rc):
    def per(rho, rcl):
        def one(Pi, R):
            out = R @ Pi @ rho @ Pi @ R
            return jnp.real(jnp.einsum('ij,ji->', out, rcl))
        return jnp.sum(jax.vmap(one)(CORR_PI, CORR_R))
    return jax.vmap(per)(rn, rc)

def f_std_chunked(rn, rc, chunk=CHUNK_EVAL_SIZE):
    n = rn.shape[0]; out = np.empty(n)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out[sl] = np.array(_std_fid_batch(rn[sl], rc[sl]))
    return out

# ================================================================
# 4b. Petz (transpose) recovery of the FULL composed channel
# ================================================================
def make_petz_fid(gamma, phi, chunk=CHUNK_EVAL_SIZE):
    sig = P_CODE / TR_P
    sig_half = P_CODE / np.sqrt(TR_P)
    Nsig = np.array(combined_channel(jnp.array(sig), jnp.float64(gamma), jnp.array(phi)))
    Nsig = (Nsig + Nsig.conj().T) / 2.0
    ev, evec = np.linalg.eigh(Nsig)
    tol = 1e-12 * float(ev.max())
    inv_sqrt = np.where(ev > tol, 1.0/np.sqrt(np.where(ev > tol, ev, 1.0)), 0.0)
    Nih = jnp.array((evec * inv_sqrt) @ evec.conj().T)
    Sh = jnp.array(sig_half)
    gj, pcj = jnp.float64(gamma), jnp.array(np.conj(phi))

    @jax.jit
    def _petz_out(rho):
        inner = Nih @ rho @ Nih
        out = Sh @ combined_adjoint(inner, gj, pcj) @ Sh
        return out / jnp.real(jnp.trace(out))

    @jax.jit
    def _fid_batch(rn, rc):
        def per(rho, rcl):
            return jnp.real(jnp.einsum('ij,ji->', _petz_out(rho), rcl))
        return jax.vmap(per)(rn, rc)

    def fid(rn, rc):
        n = rn.shape[0]; out = np.empty(n)
        for k in range(int(np.ceil(n/chunk))):
            sl = slice(k*chunk, min((k+1)*chunk, n))
            out[sl] = np.array(_fid_batch(rn[sl], rc[sl]))
        return out
    return fid

# ================================================================
# 5. Encoder ansatz
# ================================================================
def count_params(nl, n_total=N_TOTAL):
    return (nl+1)*n_total*2 + nl*n_total*(n_total-1)*3

def grow_params(params_prev, prev_nl, nl):
    """zero-pad a smaller layer count's trained params into the larger
    layer count's parameter vector -- IDENTICAL to the original script."""
    extra = count_params(nl) - count_params(prev_nl)
    return np.concatenate([np.asarray(params_prev), np.zeros(extra)])

def model_b_ansatz(params, nl, n_total=N_TOTAL):
    idx = 0
    for q in range(n_total):
        qml.RY(params[idx], wires=q); idx += 1
        qml.RZ(params[idx], wires=q); idx += 1
    for _ in range(nl):
        for c in range(n_total):
            for t in range(n_total):
                if c != t:
                    qml.CRX(params[idx], wires=[c, t]); idx += 1
                    qml.CRY(params[idx], wires=[c, t]); idx += 1
                    qml.CRZ(params[idx], wires=[c, t]); idx += 1
        for q in range(n_total):
            qml.RY(params[idx], wires=q); idx += 1
            qml.RZ(params[idx], wires=q); idx += 1
    return idx

def partial_trace_system(rho16):
    r = rho16.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=0, axis2=2)
def partial_trace_ancilla(rho16):
    r = rho16.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=1, axis2=3)

def make_layer_components(nl):
    n_p = count_params(nl)
    def circuit(params): model_b_ansatz(params, nl)
    encoder_fn = qml.matrix(circuit, wire_order=range(N_TOTAL))

    def _fwd(V, Vd, rho_code):
        rho = jnp.kron(rho_code.astype(complex), ANC_0)
        rho = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho = jnp.kron(FRESH_000, rho_a)
        rho = Vd @ rho @ V
        return partial_trace_ancilla(rho)
    _vmapped = jax.vmap(_fwd, in_axes=(None, None, 0))

    @jax.jit
    def qae_batch_chunk(params, rho_chunk):
        V = encoder_fn(params); Vd = jnp.conj(V.T)
        return _vmapped(V, Vd, rho_chunk)

    @jax.jit
    def batch_loss_and_grad(params, rho_b, tgt_b, ws, wz, wx, wy):
        def _loss(p):
            V = encoder_fn(p); Vd = jnp.conj(V.T)
            outs = _vmapped(V, Vd, rho_b)
            def per(rho, tgt):
                s1 = jnp.real(jnp.trace(S1j @ rho)); s2 = jnp.real(jnp.trace(S2j @ rho))
                l = expect_stack(L_STACK, rho)
                return (ws*(-(s1+s2)) + wz*(tgt[0]-l[0])**2
                        + wx*(tgt[1]-l[1])**2 + wy*(tgt[2]-l[2])**2)
            return jnp.mean(jax.vmap(per)(outs, tgt_b))
        return jax.value_and_grad(_loss)(params)

    @jax.jit
    def batch_stab_avg(params, rho_chunk):
        out = qae_batch_chunk(params, rho_chunk)
        def per(r): return (jnp.real(jnp.trace(S1j@r))+jnp.real(jnp.trace(S2j@r)))/2.0
        return jnp.mean(jax.vmap(per)(out))

    @jax.jit
    def batch_fidelity_arr(params, rn, rc):
        out = qae_batch_chunk(params, rn)
        return jax.vmap(lambda ro, r: jnp.real(jnp.einsum('ij,ji->', ro, r)))(out, rc)

    @jax.jit
    def batch_logical_errors_arr(params, rn, zt, xt, yt):
        out = qae_batch_chunk(params, rn)
        def per(rho, z, x, y):
            l = expect_stack(L_STACK, rho)
            return jnp.abs(z-l[0]), jnp.abs(x-l[1]), jnp.abs(y-l[2])
        return jax.vmap(per)(out, zt, xt, yt)

    @jax.jit
    def batch_probe_chunk(params, rn, rc, zt, xt, yt):
        out = qae_batch_chunk(params, rn)
        def per(rho, rcl, z, x, y):
            s1 = jnp.real(jnp.trace(S1j @ rho)); s2 = jnp.real(jnp.trace(S2j @ rho))
            l = expect_stack(L_STACK, rho)
            bl = (W_STAB_BASE*(-(s1+s2))
                  + W_ZL_BASE*(z-l[0])**2 + W_XL_BASE*(x-l[1])**2 + W_YL_BASE*(y-l[2])**2)
            fid = jnp.real(jnp.einsum('ij,ji->', rho, rcl))
            return bl, fid, jnp.abs(z-l[0]), jnp.abs(x-l[1]), jnp.abs(y-l[2])
        return jax.vmap(per)(out, rc, zt, xt, yt)

    return dict(n_params=n_p, qae_batch_chunk=qae_batch_chunk,
                batch_loss_and_grad=batch_loss_and_grad,
                batch_stab_avg=batch_stab_avg,
                batch_fidelity_arr=batch_fidelity_arr,
                batch_logical_errors_arr=batch_logical_errors_arr,
                batch_probe_chunk=batch_probe_chunk)

# ================================================================
# 6. Dataset helpers
# ================================================================
def fibonacci_sphere(n):
    golden = (1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/n), (2*np.pi*i/golden) % (2*np.pi)) for i in range(n)]

def generate_dataset(n_states, gamma, phi):
    angles = fibonacci_sphere(n_states)
    clean = np.stack([state_to_dm(logical_state(th, ph)) for th, ph in angles])
    noisy = make_noisy_stack(clean, gamma, phi)
    zl = np.array([float(np.real(np.trace(ZL_mat@r))) for r in clean])
    xl = np.array([float(np.real(np.trace(XL_mat@r))) for r in clean])
    yl = np.array([float(np.real(np.trace(YL_mat@r))) for r in clean])
    return clean, noisy, zl, xl, yl

# ================================================================
# 7. Adaptive weight helpers 
# ================================================================
def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab-STAB_SAT_LOW)/(STAB_SAT_HIGH-STAB_SAT_LOW), 0, 1))
    ws = W_STAB_HIGH - (W_STAB_HIGH-W_STAB_LOW)*sat
    wz = W_ZL_LOW + (W_ZL_HIGH-W_ZL_LOW)*sat
    wx = W_XL_LOW + (W_XL_HIGH-W_XL_LOW)*sat
    wy = W_YL_LOW + (W_YL_HIGH-W_YL_LOW)*sat
    return float(ws), float(wz), float(wx), float(wy)

def step_weight(cur, tgt):
    return cur + float(np.clip(tgt-cur, -MAX_W_DELTA, MAX_W_DELTA))

def run_epoch(params, opt_state, optimizer, ws, wz, wx, wy, rng_key, blg,
              rho_n, tgt, n_train):
    jws, jwz, jwx, jwy = map(jnp.float64, (ws, wz, wx, wy))
    steps = int(np.ceil(n_train/BATCH_SIZE))
    rng_key, sk = jax.random.split(rng_key)
    idx_all = np.array(jax.random.permutation(sk, n_train))
    for step in range(steps):
        b = idx_all[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        if len(b) == 0: continue
        lv, gv = blg(params, rho_n[b], tgt[b], jws, jwz, jwx, jwy)
        gv = jnp.where(jnp.isnan(gv), 0.0, gv)
        updates, opt_state = optimizer.update(gv, opt_state, params)
        params = optax.apply_updates(params, updates)
    return params, opt_state, rng_key


# ================================================================
# 8. train_one_layer_config -- train ONE (layer count) at ONE FIXED
#    noise point. This is train_one_config() from the original script,
#    unchanged, except sigma/gamma/mu/rho_corr are passed in instead of
#    read from module-level globals, and train/test stacks + benchmarks
#    are passed in.
# ========================================
def train_one_layer_config(nl, comps, pn, pc_j, tgt, rn_test, rc_test_j,
                           zt_j, xt_j, yt_j, f_std_test, f_noisy_test,
                           f_petz_test, rng_seed, init_params=None,
                           n_rounds=N_INIT_ROUNDS, init_epochs=INIT_EPOCHS,
                           main_epochs=MAIN_EPOCHS):
    n_p = comps['n_params']
    blg, bsa = comps['batch_loss_and_grad'], comps['batch_stab_avg']
    bfa, bla = comps['batch_fidelity_arr'], comps['batch_logical_errors_arr']
    bpc = comps['batch_probe_chunk']
    total_epochs = init_epochs + main_epochs

    print(f"\n  {'-'*60}\n  L={nl}  params={n_p}  "
          f"init={'LAYERWISE WARM-START' if init_params is not None else f'MULTI-START ({n_rounds} rounds)'}"
          f"\n  {'-'*60}")

    def probe_metrics(p):
        n = pn.shape[0]; BL=[]; FD=[]; DZ=[]; DX=[]; DY=[]
        for k in range(int(np.ceil(n/CHUNK_EVAL_SIZE))):
            sl = slice(k*CHUNK_EVAL_SIZE, min((k+1)*CHUNK_EVAL_SIZE, n))
            bl, fd, dz, dx, dy = bpc(p, pn[sl], pc_j[sl], tgt[sl, 0],
                                     tgt[sl, 1], tgt[sl, 2])
            BL.append(np.array(bl)); FD.append(np.array(fd))
            DZ.append(np.array(dz)); DX.append(np.array(dx)); DY.append(np.array(dy))
        BL=np.concatenate(BL); FD=np.concatenate(FD)
        DZ=np.concatenate(DZ); DX=np.concatenate(DX); DY=np.concatenate(DY)
        return (float(BL.mean()), float(BL.std()), float(FD.mean()), float(FD.std()),
                float(DZ.mean()), float(DZ.std()), float(DX.mean()), float(DX.std()),
                float(DY.mean()), float(DY.std()))

    steps_per_ep = int(np.ceil(N_TRAIN_STATES/BATCH_SIZE))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(LR, total_epochs*steps_per_ep)))
    rng = jax.random.PRNGKey(rng_seed)

    best_p = best_opt = None; best_f = -1.0
    best_ws, best_wz, best_wx, best_wy = W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW
    best_rng = rng; best_round_hist = None

    def phase1_round(tp, rr):
        to = optimizer.init(tp)
        ws, wz, wx, wy = W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW
        h = []
        for ep in range(init_epochs):
            if ep % WEIGHT_UPDATE_FREQ == 0:
                avg_s = float(bsa(tp, pn))
                tw, twz, twx, twy = compute_target_weights(avg_s)
                ws = step_weight(ws, tw); wz = step_weight(wz, twz)
                wx = step_weight(wx, twx); wy = step_weight(wy, twy)
            tp, to, rr = run_epoch(tp, to, optimizer, ws, wz, wx, wy, rr, blg,
                                   pn, tgt, N_TRAIN_STATES)
            h.append((ep, 1, *probe_metrics(tp)))
        return tp, to, (ws, wz, wx, wy), rr, h

    if init_params is not None:
        print(f"  Phase 1 : LAYERWISE WARM-START -- single "
              f"{init_epochs}-epoch refinement")
        rng, rr = jax.random.split(rng)
        tp, to, w, rr, h = phase1_round(jnp.asarray(init_params, dtype=jnp.float64), rr)
        best_f = h[-1][4]; best_p, best_opt = tp, to
        best_ws, best_wz, best_wx, best_wy = w
        best_rng = rr; best_round_hist = h
        print(f"  Warm-start refinement F = {best_f:.4f}")
    else:
        print(f"  Phase 1 : {n_rounds} rounds x {init_epochs} epochs "
              f"(FROM SCRATCH, avoids barren plateau at full param count)")
        print(f"  {'Rd':<3} {'Init style':<20} {'F_end':>7} {'F_std':>7} "
              f"{'<S>':>6} {'dZL':>6} {'dXL':>6} {'dYL':>6}")
        for rd in range(n_rounds):
            rng, sk = jax.random.split(rng); s = rd % 4
            if   s == 0: tp = jax.random.uniform(sk, (n_p,), minval=-np.pi, maxval=np.pi); sn = "Uniform[-pi,pi]"
            elif s == 1: tp = jax.random.normal(sk, (n_p,))*0.15; sn = "Normal(0,0.15)"
            elif s == 2: tp = jax.random.normal(sk, (n_p,))*1.0;  sn = "Normal(0,1.0)"
            else:        tp = jax.random.uniform(sk, (n_p,), minval=-0.1, maxval=0.1); sn = "Unif[-0.1,0.1]"
            rng, rr = jax.random.split(rng)
            tp, to, w, rr, h = phase1_round(tp, rr)
            f_end = h[-1][4]; avg_s = float(bsa(tp, pn))
            dzl, dxl, dyl = h[-1][6], h[-1][8], h[-1][10]
            star = ""
            if f_end > best_f:
                best_f = f_end; best_p, best_opt = tp, to
                best_ws, best_wz, best_wx, best_wy = w
                best_rng = rr; best_round_hist = h; star = " *"
            print(f"  {rd:<3} {sn:<20} {f_end:>7.4f} {f_std_test.mean():>7.4f} "
                  f"{avg_s:>6.3f} {dzl:>6.4f} {dxl:>6.4f} {dyl:>6.4f}{star}")
            if star == "": del to
            gc.collect()
        print(f"  Best warm-up F = {best_f:.4f}")

    hist_epoch = [h[0] for h in best_round_hist]
    hist_phase = [h[1] for h in best_round_hist]
    hist_loss, hist_loss_std = [h[2] for h in best_round_hist], [h[3] for h in best_round_hist]
    hist_fid, hist_fid_std = [h[4] for h in best_round_hist], [h[5] for h in best_round_hist]
    hist_dzl, hist_dzl_std = [h[6] for h in best_round_hist], [h[7] for h in best_round_hist]
    hist_dxl, hist_dxl_std = [h[8] for h in best_round_hist], [h[9] for h in best_round_hist]
    hist_dyl, hist_dyl_std = [h[10] for h in best_round_hist], [h[11] for h in best_round_hist]

    print(f"  Phase 2 : {main_epochs} continuation epochs")
    params, opt_state = best_p, best_opt
    ws, wz, wx, wy = best_ws, best_wz, best_wx, best_wy
    rng_key = best_rng
    print(f"  {'Ep':<5} {'BaseLoss':>9} {'F_tr':>7} {'F_std':>7} {'<S>':>6} "
          f"{'dZL':>7} {'dXL':>7} {'dYL':>7} {'t':>5}")
    for ep in range(main_epochs):
        t0 = time.time()
        if ep % WEIGHT_UPDATE_FREQ == 0:
            avg_s = float(bsa(params, pn))
            tw, twz, twx, twy = compute_target_weights(avg_s)
            ws = step_weight(ws, tw); wz = step_weight(wz, twz)
            wx = step_weight(wx, twx); wy = step_weight(wy, twy)
        params, opt_state, rng_key = run_epoch(
            params, opt_state, optimizer, ws, wz, wx, wy, rng_key, blg,
            pn, tgt, N_TRAIN_STATES)
        (bm, bs, fm, fs, zm, zs, xm, xs, ym, ys) = probe_metrics(params)
        avg_s = float(bsa(params, pn))
        te = init_epochs + ep
        hist_epoch.append(te); hist_phase.append(2)
        hist_loss.append(bm); hist_loss_std.append(bs)
        hist_fid.append(fm); hist_fid_std.append(fs)
        hist_dzl.append(zm); hist_dzl_std.append(zs)
        hist_dxl.append(xm); hist_dxl_std.append(xs)
        hist_dyl.append(ym); hist_dyl_std.append(ys)
        if ep % 10 == 0 or ep == main_epochs-1:
            beat = "BEATS std QEC" if fm > f_std_test.mean() else "below std QEC"
            print(f"  {te:<5} {bm:>9.3f} {fm:>7.4f} {f_std_test.mean():>7.4f} "
                  f"{avg_s:>6.3f} {zm:>7.4f} {xm:>7.4f} {ym:>7.4f} "
                  f"{time.time()-t0:>4.1f}s  [{beat}]")
        if ep % 20 == 0: gc.collect()

    to_arr = lambda l, dt: np.array(l, dtype=dt)
    hist = dict(
        hist_epoch=to_arr(hist_epoch, np.int64), hist_phase=to_arr(hist_phase, np.int64),
        hist_loss=to_arr(hist_loss, float), hist_loss_std=to_arr(hist_loss_std, float),
        hist_fid=to_arr(hist_fid, float), hist_fid_std=to_arr(hist_fid_std, float),
        hist_dzl=to_arr(hist_dzl, float), hist_dzl_std=to_arr(hist_dzl_std, float),
        hist_dxl=to_arr(hist_dxl, float), hist_dxl_std=to_arr(hist_dxl_std, float),
        hist_dyl=to_arr(hist_dyl, float), hist_dyl_std=to_arr(hist_dyl_std, float))

    f_qae = np.array(bfa(params, rn_test, rc_test_j))
    dzl_t, dxl_t, dyl_t = (np.array(a) for a in bla(params, rn_test, zt_j, xt_j, yt_j))
    gap = f_std_test.mean() - f_noisy_test.mean()
    pct = (f_qae.mean()-f_noisy_test.mean())/gap*100 if gap > 1e-6 else float("nan")
    verdict = "BEATS std QEC" if f_qae.mean() > f_std_test.mean() else "BELOW std QEC"
    print(f"  L={nl} test eval: F_qae={f_qae.mean():.5f}  F_std={f_std_test.mean():.5f}  "
          f"F_petz={f_petz_test.mean():.5f}  F_noisy={f_noisy_test.mean():.5f}  "
          f"gap_closed={pct:.1f}%  [{verdict}]")

    return dict(n_layers=nl, n_params=n_p, params=np.array(params),
                f_qae_test=f_qae, dzl_test=dzl_t, dxl_test=dxl_t, dyl_test=dyl_t,
                **hist)


def train_point_layerwise(L, tau_half, mu, sigma, gamma, rho_corr, layer_sweep,
                          comps_cache, rng_seed, n_rounds, init_epochs, main_epochs):
    """FULL layerwise sweep (L=1 from scratch -> grow -> L=2 warm-start),
    run COMPLETELY FRESH for this ONE noise point. Nothing here depends on
    any other noise point's trained parameters."""
    phi = build_phi(sigma, rho_corr, mu)
    pc, pn_np, pz, px, py = generate_dataset(N_TRAIN_STATES, gamma, phi)
    pn = jnp.array(pn_np); pc_j = jnp.array(pc)
    tgt = jnp.array(np.stack([pz, px, py], axis=1))

    rc_test, rn_test_np, zt, xt, yt = generate_dataset(N_TEST_STATES, gamma, phi)
    rn_test = jnp.array(rn_test_np); rc_test_j = jnp.array(rc_test)
    zt_j, xt_j, yt_j = jnp.array(zt), jnp.array(xt), jnp.array(yt)

    print(f"  Precomputing standard QEC + Petz benchmarks "
          f"({N_TEST_STATES} test states) ...", end=" ", flush=True)
    t0 = time.time()
    f_std_test = f_std_chunked(rn_test, rc_test_j)
    f_noisy_test = np.real(np.einsum('nij,nji->n', rn_test_np, rc_test))
    f_petz_test = make_petz_fid(gamma, phi)(rn_test, rc_test_j)
    print(f"done in {time.time()-t0:.1f}s  "
          f"F_noisy={f_noisy_test.mean():.5f}  F_std={f_std_test.mean():.5f}  "
          f"F_petz={f_petz_test.mean():.5f}")

    per_layer = {}
    prev_params, prev_nl = None, None
    for nl in layer_sweep:
        comps = comps_cache[nl]
        if prev_params is None:
            res = train_one_layer_config(
                nl, comps, pn, pc_j, tgt, rn_test, rc_test_j, zt_j, xt_j, yt_j,
                f_std_test, f_noisy_test, f_petz_test, rng_seed + nl,
                init_params=None, n_rounds=n_rounds,
                init_epochs=init_epochs, main_epochs=main_epochs)
        else:
            warm = grow_params(prev_params, prev_nl, nl)
            res = train_one_layer_config(
                nl, comps, pn, pc_j, tgt, rn_test, rc_test_j, zt_j, xt_j, yt_j,
                f_std_test, f_noisy_test, f_petz_test, rng_seed + nl,
                init_params=warm, init_epochs=init_epochs, main_epochs=main_epochs)
        per_layer[nl] = res
        prev_nl, prev_params = nl, res['params']
        gc.collect()

    final = per_layer[layer_sweep[-1]]
    final['f_std_test'] = f_std_test
    final['f_petz_test'] = f_petz_test
    final['f_noisy_test'] = f_noisy_test
    return final, per_layer


# ============================================================ noise sources
def noise_at_tau_formula(tau_half_us, T1_us, Tphi_us):
    sig = np.sqrt(max(2.0*tau_half_us/Tphi_us, 0.0)) if Tphi_us > 0 else 0.0
    gam = 1.0 - np.exp(-tau_half_us/T1_us) if T1_us > 0 else 0.0
    return float(sig), float(gam)


def fit_measured_noise(path, tau_list):
    """Fit smooth mu(t), sigma(t), gamma(t) against t_eff (REAL accumulated
    idle incl. pad), evaluate at raw tau (no pad, matches a QAE interval).

    Rows where lam_perp > 1 are PHYSICALLY IMPOSSIBLE (transverse decay
    cannot grow) and are excluded from ALL THREE fits (mu, sigma, gamma),
    not just wherever sigma_deph itself happens to be nan -- a compromised
    measurement point usually contaminates every derived quantity at that
    row, even the ones that superficially look finite.
    """
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        needed = ["t_eff_us", "mu_hw", "sigma_deph", "gamma_hw"]
        missing = [c for c in needed if c not in (r.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path}: missing columns {missing}")
        has_lam = "lam_perp" in (r.fieldnames or [])
        if not has_lam:
            print("  WARNING: no 'lam_perp' column in this CSV -- cannot "
                  "screen for unphysical rows, using all rows as-is")
        for row in r:
            def g(k):
                try:
                    x = float(row[k]); return x if np.isfinite(x) else np.nan
                except (TypeError, ValueError):
                    return np.nan
            lam = g("lam_perp") if has_lam else np.nan
            rows.append((g("t_eff_us"), g("mu_hw"), g("sigma_deph"), g("gamma_hw"), lam))
    rows = np.array(rows, dtype=float)
    t_eff = rows[:, 0]

    bad = np.isfinite(rows[:, 4]) & (rows[:, 4] > 1.0)
    if bad.any():
        bad_t = t_eff[bad]
        print(f"  EXCLUDING {int(bad.sum())} row(s) with lam_perp > 1 "
              f"(physically impossible decay) at t_eff={list(np.round(bad_t, 2))}: "
              f"treated as fully unreliable, dropped from mu/sigma/gamma fits")
    keep = ~bad
    rows = rows[keep]; t_eff = t_eff[keep]
    print(f"  loaded {len(rows) + int(bad.sum())} rows from {path}, "
          f"using {len(rows)} after screening")
    print(f"  fitting vs t_eff (REAL accumulated idle incl. pad), "
          f"evaluating at raw tau/2 (no pad, matches the QAE interval)")

    m_ok = np.isfinite(rows[:, 1])
    A = np.stack([t_eff[m_ok], np.ones(int(m_ok.sum()))], axis=1)
    (slope_mu, icpt_mu), *_ = np.linalg.lstsq(A, rows[m_ok, 1], rcond=None)
    resid = rows[m_ok, 1] - (slope_mu * t_eff[m_ok] + icpt_mu)
    print(f"    mu(t)    = {slope_mu:+.5f}*t {icpt_mu:+.4f}   "
          f"resid std={resid.std():.4f} rad  (n={int(m_ok.sum())})")

    g_ok = np.isfinite(rows[:, 3]) & (rows[:, 3] > -0.05)
    try:
        from scipy.optimize import curve_fit
        gm = lambda t, T1e: 1.0 - np.exp(-t / T1e)
        popt, _ = curve_fit(gm, t_eff[g_ok], rows[g_ok, 3],
                            p0=[300.0], bounds=(1.0, 1e6))
        T1_eff = float(popt[0])
        resid = rows[g_ok, 3] - gm(t_eff[g_ok], T1_eff)
        print(f"    gamma(t) = 1-exp(-t/{T1_eff:.1f})   "
              f"resid std={resid.std():.4f}  (n={int(g_ok.sum())})")
        gamma_fn = lambda t: float(np.clip(1.0 - np.exp(-t / T1_eff), 0.0, 1.0))
    except ImportError:
        A = np.stack([t_eff[g_ok], np.ones(int(g_ok.sum()))], axis=1)
        (sg, ig), *_ = np.linalg.lstsq(A, rows[g_ok, 3], rcond=None)
        print(f"    scipy missing: gamma(t) linear fallback = "
              f"{sg:.6f}*t {ig:+.4f}")
        gamma_fn = lambda t: float(np.clip(sg * t + ig, 0.0, 1.0))

    s_ok = np.isfinite(rows[:, 2]) & (rows[:, 2] >= 0)
    y = rows[s_ok, 2] ** 2
    X = np.stack([t_eff[s_ok], t_eff[s_ok] ** 2], axis=1)
    try:
        from scipy.optimize import nnls
        (a_s, b_s), _ = nnls(X, y)
    except ImportError:
        (a_s, b_s), *_ = np.linalg.lstsq(X, y, rcond=None)
        a_s, b_s = max(a_s, 0.0), max(b_s, 0.0)
    pred = np.sqrt(np.clip(a_s * t_eff[s_ok] + b_s * t_eff[s_ok] ** 2, 0, None))
    resid = rows[s_ok, 2] - pred
    print(f"    sigma(t) = sqrt({a_s:.6f}*t + {b_s:.6f}*t^2)   "
          f"resid std={resid.std():.4f}  (n={int(s_ok.sum())})")
    if resid.std() > 0.15:
        print(f"    WARNING sigma fit residual std is large: device "
              f"dephasing may not be well described by ANY smooth "
              f"2-parameter model here. Consider more repeats.")
    sigma_fn = lambda t: float(np.sqrt(max(a_s * t + b_s * t * t, 0.0)))
    mu_fn = lambda t: float(slope_mu * t + icpt_mu)

    out = []
    for t in tau_list:
        mu_v, sig_v, gam_v = mu_fn(t), sigma_fn(t), gamma_fn(t)
        print(f"    tau/2={t:6.2f} (interval idle, no pad): "
              f"mu={mu_v:+.4f}  sigma={sig_v:.4f}  gamma={gam_v:.4f}")
        out.append((t, mu_v, sig_v, gam_v))
    return out


def build_formula_points(tau_list, T1, Tphi, mu_const, mu_per_us):
    print(f"  no --measured-csv given: using T1/T_phi FORMULA")
    out = []
    for t in tau_list:
        sig, gam = noise_at_tau_formula(t, T1, Tphi)
        mu = mu_per_us * t if mu_per_us is not None else mu_const
        print(f"    tau/2={t:6.2f}: mu={mu:+.4f}  sigma={sig:.4f}  gamma={gam:.4f}")
        out.append((t, mu, sig, gam))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau-list", default="0,2,4,6,8,12,16,24")
    p.add_argument("--measured-csv", default="")
    p.add_argument("--t1", type=float, default=0.0)
    p.add_argument("--tphi", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--mu-per-us", type=float, default=None)
    p.add_argument("--rho-corr", type=float, default=0.05)
    p.add_argument("--L", type=int, default=2,
                   help="TARGET (hardware) layer count; the layer sweep "
                        "runs 1..L, growing into it, at EVERY noise point")
    p.add_argument("--layer-sweep", default="",
                   help="override, e.g. '1,2'; default = 1..L")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="dephasing_multipoint_L2.npz")
    p.add_argument("--n-init-rounds", type=int, default=N_INIT_ROUNDS)
    p.add_argument("--init-epochs", type=int, default=INIT_EPOCHS)
    p.add_argument("--main-epochs", type=int, default=MAIN_EPOCHS)
    a = p.parse_args()

    taus = sorted(float(x) for x in a.tau_list.split(","))
    layer_sweep = ([int(x) for x in a.layer_sweep.split(",")] if a.layer_sweep
                  else list(range(1, a.L + 1)))
    print(f"\n  L(target)={a.L}   layer sweep={layer_sweep}   "
          f"rho_c={a.rho_corr}   {len(taus)} tau/2 points")
    print(f"  EACH noise point runs its OWN FULL layerwise sweep "
          f"{layer_sweep} FROM SCRATCH -- no warm start across points\n")

    if a.measured_csv:
        points = fit_measured_noise(a.measured_csv, taus)
        noise_source = f"measured_csv:{a.measured_csv}"
    else:
        if not (a.t1 and a.tphi):
            raise SystemExit("no --measured-csv given: --t1 and --tphi are "
                             "REQUIRED for formula mode")
        points = build_formula_points(taus, a.t1, a.tphi, a.mu, a.mu_per_us)
        noise_source = f"formula:T1={a.t1},Tphi={a.tphi}"

    print("  JIT-building ansatz components per layer count ...", end=" ", flush=True)
    t0 = time.time()
    comps_cache = {nl: make_layer_components(nl) for nl in layer_sweep}
    print(f"done in {time.time()-t0:.1f}s")

    results, per_layer_all = {}, {}
    t_sweep = time.time()
    for i, (t, mu, sig, gam) in enumerate(points):
        print(f"\n{'='*70}\n  NOISE POINT {i+1}/{len(points)}: tau/2={t} us  "
              f"mu={mu:+.4f}  sigma={sig:.4f}  gamma={gam:.4f}"
              f"\n  (full layerwise sweep {layer_sweep}, FROM SCRATCH)"
              f"\n{'='*70}")
        t0 = time.time()
        final, per_layer = train_point_layerwise(
            a.L, t, mu, sig, gam, a.rho_corr, layer_sweep, comps_cache,
            a.seed + 1000 * i, a.n_init_rounds, a.init_epochs, a.main_epochs)
        results[i] = final
        per_layer_all[i] = per_layer
        print(f"  point {i+1} total time: {time.time()-t0:.0f}s")
        gc.collect()
    print(f"\n  total training time: {time.time()-t_sweep:.0f}s")

    print(f"\n{'='*90}\n  FINAL SUMMARY (all points, target L={a.L})\n{'='*90}")
    print(f"  {'tau/2':>6} {'mu':>8} {'sigma':>7} {'gamma':>7} {'F_noisy':>8} "
          f"{'F_std':>8} {'F_petz':>8} {'F_qae':>8} {'gap%':>6} {'verdict':>14}")
    for i, (t, mu, sig, gam) in enumerate(points):
        r = results[i]
        gap = r['f_std_test'].mean() - r['f_noisy_test'].mean()
        pct = (r['f_qae_test'].mean()-r['f_noisy_test'].mean())/gap*100 if gap > 1e-6 else float("nan")
        verdict = "BEATS std QEC" if r['f_qae_test'].mean() > r['f_std_test'].mean() else "below std QEC"
        print(f"  {t:6.1f} {mu:+8.4f} {sig:7.4f} {gam:7.4f} "
              f"{r['f_noisy_test'].mean():8.5f} {r['f_std_test'].mean():8.5f} "
              f"{r['f_petz_test'].mean():8.5f} {r['f_qae_test'].mean():8.5f} "
              f"{pct:6.1f} {verdict:>14}")

    angles_t_full = fibonacci_sphere(N_TEST_STATES)
    save = dict(tau_half=np.array(taus), rho_corr=a.rho_corr, L=a.L,
                layer_sweep=np.array(layer_sweep), noise_source=noise_source,
                meta_test_thetas=np.array([x for x, _ in angles_t_full]),
                meta_test_phis=np.array([y for _, y in angles_t_full]))
    for i, (t, mu, sig, gam) in enumerate(points):
        r = results[i]
        save[f"P{i}_tau_half"] = t
        save[f"P{i}_mu"] = mu
        save[f"P{i}_sigma"] = sig
        save[f"P{i}_gamma"] = gam
        for k, v in r.items():
            save[f"P{i}_{k}"] = v
        for nl, rl in per_layer_all[i].items():
            save[f"P{i}_L{nl}_params"] = rl['params']
            save[f"P{i}_L{nl}_f_qae_test"] = rl['f_qae_test']
    np.savez(a.out, **save)
    print(f"\n  saved {a.out}  ({len(taus)} points, source: {noise_source})")

    with open(a.out.replace(".npz", ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_half", "mu", "sigma", "gamma", "F_qae", "F_std",
                    "F_petz", "F_noisy", "gap_closed_pct"])
        for i, (t, mu, sig, gam) in enumerate(points):
            r = results[i]
            gap = r['f_std_test'].mean() - r['f_noisy_test'].mean()
            pct = (r['f_qae_test'].mean()-r['f_noisy_test'].mean())/gap*100 if gap > 1e-6 else float("nan")
            w.writerow([t, f"{mu:.5f}", f"{sig:.5f}", f"{gam:.5f}",
                       f"{r['f_qae_test'].mean():.6f}", f"{r['f_std_test'].mean():.6f}",
                       f"{r['f_petz_test'].mean():.6f}", f"{r['f_noisy_test'].mean():.6f}",
                       f"{pct:.2f}"])
    print(f"  saved {a.out.replace('.npz', '.csv')}")


if __name__ == "__main__":
    main()