"""
Spectral Measurement-Free QEC 
===========================================================================
Steane [[7,1,3]] CSS code under single-qubit DEPOLARIZING noise.
Weight-3 local logical representatives (X0X1X2 / Z0Z1Z2 / -Y0Y1Y2).
"""

import warnings; warnings.filterwarnings("ignore")
import os, time, gc, csv, itertools
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"   # reduce fragmentation
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
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce

print(f"  JAX backend : {jax.default_backend().upper()}")
print(f"  JAX devices : {jax.devices()}")

# ================================================================
# 1. Global constants
# ================================================================
N_CODE  = 7
N_ANC   = 1
N_TOTAL = N_CODE + N_ANC          # 8
SYS_DIM = 2 ** N_CODE             # 128
ANC_DIM = 2 ** N_ANC              # 2
TOT_DIM = 2 ** N_TOTAL            # 256

P_NOISE = 0.05

N_TRAIN_STATES = 100
N_TEST_STATES  = 1000
BATCH_SIZE      = 5               
CHUNK_EVAL_SIZE = 50              
STD_CHUNK       = 25              


LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
TOTAL_EPOCHS       = INIT_EPOCHS + MAIN_EPOCHS
WEIGHT_UPDATE_FREQ = 5

# Adaptive weight bounds
W_STAB_HIGH = 5.0;  W_STAB_LOW = 0.5
W_ZL_LOW = 5.0; W_ZL_HIGH = 20.0
W_XL_LOW = 5.0; W_XL_HIGH = 20.0
W_YL_LOW = 5.0; W_YL_HIGH = 20.0
STAB_SAT_LOW  = 0.70
STAB_SAT_HIGH = 0.99
MAX_W_DELTA   = 1.0

# Fixed baseline weights (tracking/plotting only; training uses adaptive)
W_STAB_BASE = 1.0
W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

LAYER_SWEEP = [1, 2, 3, 4, 5]
USE_LAYERWISE = True
FILE_TAG = "layerwise" if USE_LAYERWISE else "multistart"

print(f"""
  Steane [[7,1,3]] — depolarizing p={P_NOISE}
  N_TOTAL={N_TOTAL} (7 code + 1 anc), Layer sweep {LAYER_SWEEP}, layerwise={USE_LAYERWISE}
  Train {N_TRAIN_STATES} / Test {N_TEST_STATES}, batch {BATCH_SIZE}, LR {LR}
  Phase 1 {N_INIT_ROUNDS}x{INIT_EPOCHS} + Phase 2 {MAIN_EPOCHS}  (total {TOTAL_EPOCHS})
""")

# ================================================================
# 2. Pauli primitives + Steane operators (weight-3 logicals)
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops): return reduce(np.kron, ops)

def op_on(qubits, mat, n=N_CODE):
    ops = [I2]*n
    for q in qubits: ops[q] = mat
    return kron_list(ops)

S_mats = [op_on([0,2,4,6], X_), op_on([1,2,5,6], X_), op_on([3,4,5,6], X_),
          op_on([0,2,4,6], Z_), op_on([1,2,5,6], Z_), op_on([3,4,5,6], Z_)]

LOGICAL_SUPPORT = [0, 1, 2]                 # distance-3 rep for Steane
XL_mat = op_on(LOGICAL_SUPPORT, X_)         # X0 X1 X2
ZL_mat = op_on(LOGICAL_SUPPORT, Z_)         # Z0 Z1 Z2
YL_mat = 1j * XL_mat @ ZL_mat               # = -Y0Y1Y2 (Hermitian, ^2 = I)

N_STAB = len(S_mats)

# Pre-stacked operators for O(d^2) einsum expectation values
S_STACK = jnp.array(np.stack(S_mats))                       # (6,128,128)
L_STACK = jnp.array(np.stack([ZL_mat, XL_mat, YL_mat]))     # (3,128,128) -> [zl,xl,yl]

def expect_stack(op_stack, rho):
    """<O_k> = tr(O_k rho) for each stacked O_k, via einsum (no full matmul)."""
    return jnp.real(jnp.einsum('kij,ji->k', op_stack, rho))

# logical basis via projector diagonalization (computed once)
_P = np.eye(SYS_DIM, dtype=complex)
for S in S_mats: _P = _P @ ((np.eye(SYS_DIM) + S)/2)
_ev, _evec = np.linalg.eigh(_P)
_code = _evec[:, np.isclose(_ev, 1.0)]
_ZLsub = _code.conj().T @ ZL_mat @ _code
_e2, _v2 = np.linalg.eigh(_ZLsub)
_k0 = _code @ _v2[:, 0]; _k1 = _code @ _v2[:, 1]
ket_0L, ket_1L = (_k1, _k0) if _e2[0] < _e2[1] else (_k0, _k1)
ket_0L /= np.linalg.norm(ket_0L); ket_1L /= np.linalg.norm(ket_1L)
print("  ✓ Steane logical basis computed (weight-3 logicals, code dim = 2)")

def logical_state(theta, phi):
    return np.cos(theta/2)*ket_0L + np.exp(1j*phi)*np.sin(theta/2)*ket_1L

def state_to_dm(psi):
    psi = psi.reshape(-1, 1); return psi @ psi.conj().T

ANC_0   = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0, 0].set(1.0)
FRESH_0 = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0, 0].set(1.0)

# ================================================================
# 3. Depolarizing channel — JAX batched + chunked (pre-embedded Paulis)
# ================================================================
_PAULI_EMB = [(jnp.array(op_on([q], X_)),
               jnp.array(op_on([q], Y_)),
               jnp.array(op_on([q], Z_))) for q in range(N_CODE)]

@jax.jit
def _depol_single(rho, p):
    res = rho
    for q in range(N_CODE):
        acc = (1.0 - p) * res
        for E in _PAULI_EMB[q]:            # X,Y,Z are Hermitian => E^dag = E
            acc = acc + (p / 3.0) * (E @ res @ E)
        res = acc
    return res

def make_noisy_stack(clean_np, p, chunk=CHUNK_EVAL_SIZE):
    pj = jnp.float64(p)
    f = jax.vmap(lambda r: _depol_single(r, pj))
    n = clean_np.shape[0]; out = np.empty_like(clean_np)
    for k in range(int(np.ceil(n / chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out[sl] = np.array(f(jnp.array(clean_np[sl])))
    return out

# (numpy reference version kept for completeness / debugging)
def depolarizing_channel(rho, p, n=N_CODE):
    result = rho.copy()
    for q in range(n):
        nr = (1-p)*result
        for P_ in [X_, Y_, Z_]:
            ops = [I2]*n; ops[q] = P_
            E = kron_list(ops)
            nr += (p/3)*(E @ result @ E.conj().T)
        result = nr
    return result

# ================================================================
# 4. Standard Steane recovery  (COMPLETE CSS decoder — all 64 syndromes)
# ================================================================
# Steane is CSS: S_mats[0:3] are X-type stabilizers (detect Z errors),
# S_mats[3:6] are Z-type stabilizers (detect X errors). The 6-bit global
# syndrome factorizes into a Z-error part (from the 3 X-stabilizers) and an
# X-error part (from the 3 Z-stabilizers). Decoding each 3-bit part
# independently (8 syndromes -> identity + 7 single-qubit corrections)
# assigns a VALID Pauli correction to EVERY one of the 8x8 = 64 global
# syndromes  (no identity fallback on the 42 "composite" syndromes).
def _commutes(E, S):
    return np.allclose(E @ S - S @ E, 0)

X_STAB_IDX = [0, 1, 2]   # X-type stabilizers -> detect Z errors
Z_STAB_IDX = [3, 4, 5]   # Z-type stabilizers -> detect X errors

# X-error syndrome (measured on Z-type stabilizers) -> correcting X operator
_xsyn_to_Xcorr = {(1, 1, 1): np.eye(SYS_DIM, dtype=complex)}
for q in range(N_CODE):
    E = op_on([q], X_)
    syn = tuple(1 if _commutes(E, S_mats[j]) else -1 for j in Z_STAB_IDX)
    _xsyn_to_Xcorr[syn] = E

# Z-error syndrome (measured on X-type stabilizers) -> correcting Z operator
_zsyn_to_Zcorr = {(1, 1, 1): np.eye(SYS_DIM, dtype=complex)}
for q in range(N_CODE):
    E = op_on([q], Z_)
    syn = tuple(1 if _commutes(E, S_mats[j]) else -1 for j in X_STAB_IDX)
    _zsyn_to_Zcorr[syn] = E

# assert full coverage of both 3-bit sub-syndrome tables
assert len(_xsyn_to_Xcorr) == 8, "X-syndrome table incomplete!"
assert len(_zsyn_to_Zcorr) == 8, "Z-syndrome table incomplete!"

_I128 = np.eye(SYS_DIM, dtype=complex)
_CORR_M = []
for signs in itertools.product([1, -1], repeat=N_STAB):
    Pi = np.eye(SYS_DIM, dtype=complex)
    for s, S in zip(signs, S_mats):
        Pi = Pi @ ((np.eye(SYS_DIM) + s*S)/2)
    # factorize the global syndrome into independent X- and Z-parts
    zsyn = tuple(signs[j] for j in X_STAB_IDX)   # -> Z-correction
    xsyn = tuple(signs[j] for j in Z_STAB_IDX)   # -> X-correction
    R = _xsyn_to_Xcorr[xsyn] @ _zsyn_to_Zcorr[zsyn]
    _CORR_M.append(R @ Pi)
CORR_M = jnp.array(np.stack(_CORR_M))          # (64,128,128) — all valid

@jax.jit
def std_qec_jax(rho):
    def body(acc, M):
        return acc + M @ rho @ jnp.conj(M.T), None
    out, _ = jax.lax.scan(body, jnp.zeros_like(rho), CORR_M)
    return out

@jax.jit
def _std_fid_batch(rn, rc):
    def per(rho, rcl):
        def body(acc, M):
            out = M @ rho @ jnp.conj(M.T)
            return acc + jnp.real(jnp.einsum('ij,ji->', out, rcl)), None
        f, _ = jax.lax.scan(body, jnp.float64(0.0), CORR_M)
        return f
    return jax.vmap(per)(rn, rc)

def f_std_chunked(rn, rc, chunk=STD_CHUNK):
    n = rn.shape[0]; out = np.empty(n)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out[sl] = np.array(_std_fid_batch(rn[sl], rc[sl]))
    return out

# ================================================================
# 5. Encoder ansatz
# ================================================================
def count_params(n_layers, n_total=N_TOTAL):
    return (n_layers+1)*n_total*2 + n_layers*n_total*(n_total-1)*3

def model_b_ansatz(params, n_layers, n_total=N_TOTAL):
    idx = 0
    for q in range(n_total):
        qml.RY(params[idx], wires=q); idx += 1
        qml.RZ(params[idx], wires=q); idx += 1
    for _ in range(n_layers):
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

def grow_params(params_prev, prev_nl, nl):
    extra = count_params(nl) - count_params(prev_nl)
    return np.concatenate([np.asarray(params_prev), np.zeros(extra)])

def partial_trace_system(rho):
    r = rho.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=0, axis2=2)

def partial_trace_ancilla(rho):
    r = rho.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=1, axis2=3)

# ================================================================
# 6. Per-layer factory
# ================================================================
def make_layer_components(n_layers):
    n_p = count_params(n_layers)
    def circuit(params): model_b_ansatz(params, n_layers)
    encoder_fn = qml.matrix(circuit, wire_order=range(N_TOTAL))

    def _fwd(V, Vd, rho_code):
        rho = jnp.kron(rho_code, ANC_0)
        rho = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho = jnp.kron(FRESH_0, rho_a)
        rho = Vd @ rho @ V
        return partial_trace_ancilla(rho)

    _vmapped = jax.vmap(_fwd, in_axes=(None, None, 0))

    @jax.jit
    def qae_batch_chunk(params, rho_chunk):
        V = encoder_fn(params); Vd = jnp.conj(V.T)
        return _vmapped(V, Vd, rho_chunk)

    # ---- BATCHED loss+grad: build V once, vmap forward over mini-batch ----
    @jax.jit
    def batch_loss_and_grad(params, rho_b, tgt_b, ws, wz, wx, wy):
        def _loss(p):
            V = encoder_fn(p); Vd = jnp.conj(V.T)
            outs = _vmapped(V, Vd, rho_b)
            def per(rho, tgt):
                s = expect_stack(S_STACK, rho)
                l = expect_stack(L_STACK, rho)          # [zl, xl, yl]
                return (ws*(-jnp.mean(s))
                        + wz*(tgt[0]-l[0])**2
                        + wx*(tgt[1]-l[1])**2
                        + wy*(tgt[2]-l[2])**2)
            return jnp.mean(jax.vmap(per)(outs, tgt_b))
        return jax.value_and_grad(_loss)(params)

    @jax.jit
    def batch_stab_avg(params, rho_chunk):
        out = qae_batch_chunk(params, rho_chunk)
        per = lambda r: jnp.sum(expect_stack(S_STACK, r)) / N_STAB
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

    # ---- fused probe: ONE forward -> baseline loss + fid + ΔZL/ΔXL/ΔYL ----
    @jax.jit
    def batch_probe_chunk(params, rn, rc, zt, xt, yt):
        out = qae_batch_chunk(params, rn)
        def per(rho, rcl, z, x, y):
            s = expect_stack(S_STACK, rho)
            l = expect_stack(L_STACK, rho)
            bl = (W_STAB_BASE*(-jnp.mean(s)) + W_ZL_BASE*(z-l[0])**2
                  + W_XL_BASE*(x-l[1])**2 + W_YL_BASE*(y-l[2])**2)
            fid = jnp.real(jnp.einsum('ij,ji->', rho, rcl))
            return bl, fid, jnp.abs(z-l[0]), jnp.abs(x-l[1]), jnp.abs(y-l[2])
        return jax.vmap(per)(out, rc, zt, xt, yt)

    return dict(n_params=n_p, qae_batch_chunk=qae_batch_chunk,
                batch_loss_and_grad=batch_loss_and_grad,
                batch_stab_avg=batch_stab_avg, batch_fidelity_arr=batch_fidelity_arr,
                batch_logical_errors_arr=batch_logical_errors_arr,
                batch_probe_chunk=batch_probe_chunk)

# ================================================================
# 7. Datasets
# ================================================================
def fibonacci_sphere(n):
    g = (1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/n), (2*np.pi*i/g) % (2*np.pi)) for i in range(n)]

def generate_dataset(n, p=P_NOISE):
    angles = fibonacci_sphere(n)
    clean  = np.stack([state_to_dm(logical_state(th, ph)) for th, ph in angles])
    noisy  = make_noisy_stack(clean, p)
    data = []
    for i, (th, ph) in enumerate(angles):
        rc = clean[i]
        data.append(dict(theta=th, phi=ph, rho_clean=rc, rho_noisy=noisy[i],
                         zl=float(np.real(np.trace(ZL_mat@rc))),
                         xl=float(np.real(np.trace(XL_mat@rc))),
                         yl=float(np.real(np.trace(YL_mat@rc)))))
    return data

print("  Generating training data ..."); training_data = generate_dataset(N_TRAIN_STATES)
print("  Generating test data ...");     test_data     = generate_dataset(N_TEST_STATES)

rho_noisy_train_stack = jnp.array(np.stack([d['rho_noisy'] for d in training_data]))
rho_clean_train_stack = jnp.array(np.stack([d['rho_clean'] for d in training_data]))
targets_train_stack   = jnp.array([[d['zl'], d['xl'], d['yl']] for d in training_data])

rho_clean_test_np = np.stack([d['rho_clean'] for d in test_data])
rho_noisy_test_np = np.stack([d['rho_noisy'] for d in test_data])
rho_noisy_test_stack = jnp.array(rho_noisy_test_np)
rho_clean_test_stack = jnp.array(rho_clean_test_np)
zl_test = jnp.array([d['zl'] for d in test_data])
xl_test = jnp.array([d['xl'] for d in test_data])
yl_test = jnp.array([d['yl'] for d in test_data])
test_thetas = np.array([d['theta'] for d in test_data])
test_phis   = np.array([d['phi']   for d in test_data])

print(f"  Precomputing standard Steane QEC on {N_TEST_STATES} test states ...")
f_std_test   = f_std_chunked(rho_noisy_test_stack, rho_clean_test_stack)
f_noisy_test = np.real(np.einsum('nij,nji->n', rho_noisy_test_np, rho_clean_test_np))
print(f"  No-correction F = {f_noisy_test.mean():.5f} | Std Steane QEC F = {f_std_test.mean():.5f}")

# ================================================================
# 8. Chunked eval helpers (use the jitted/vmapped factory fns)
# ================================================================
def eval_fidelity_chunked(params, bfa_fn, rn, rc, chunk=CHUNK_EVAL_SIZE):
    n = rn.shape[0]; f = np.empty(n)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        f[sl] = np.array(bfa_fn(params, rn[sl], rc[sl]))
    return f

def eval_logical_errors_chunked(params, bla_fn, rn, zt, xt, yt, chunk=CHUNK_EVAL_SIZE):
    n = rn.shape[0]; dz = np.empty(n); dx = np.empty(n); dy = np.empty(n)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        a, b, c = bla_fn(params, rn[sl], zt[sl], xt[sl], yt[sl])
        dz[sl] = np.array(a); dx[sl] = np.array(b); dy[sl] = np.array(c)
    return dz, dx, dy

# ================================================================
# 9. Adaptive weights + epoch (batched gradient)
# ================================================================
def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab-STAB_SAT_LOW)/(STAB_SAT_HIGH-STAB_SAT_LOW), 0, 1))
    return (W_STAB_HIGH-(W_STAB_HIGH-W_STAB_LOW)*sat,
            W_ZL_LOW+(W_ZL_HIGH-W_ZL_LOW)*sat,
            W_XL_LOW+(W_XL_HIGH-W_XL_LOW)*sat,
            W_YL_LOW+(W_YL_HIGH-W_YL_LOW)*sat)

def step_weight(cur, tgt): return cur + float(np.clip(tgt-cur, -MAX_W_DELTA, MAX_W_DELTA))

def batch_stab_avg_chunked(bsa_fn, params, rn, chunk=CHUNK_EVAL_SIZE):
    # chunked mean stabilizer over full training set (bounds memory)
    n = rn.shape[0]; tot = 0.0; cnt = 0
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        m = float(bsa_fn(params, rn[sl])); w = (sl.stop - sl.start)
        tot += m * w; cnt += w
    return tot / max(cnt, 1)

def run_epoch(params, opt_state, optimizer, ws, wz, wx, wy, rng_key, blg):
    jws, jwz, jwx, jwy = map(jnp.float64, (ws, wz, wx, wy))
    steps = int(np.ceil(N_TRAIN_STATES/BATCH_SIZE))
    rng_key, sk = jax.random.split(rng_key)
    idx = np.array(jax.random.permutation(sk, N_TRAIN_STATES))
    ep_loss = 0.0; nb = 0
    for st in range(steps):
        b = idx[st*BATCH_SIZE:(st+1)*BATCH_SIZE]
        if len(b) == 0: continue
        rho_b = rho_noisy_train_stack[b]
        tgt_b = targets_train_stack[b]
        lv, gv = blg(params, rho_b, tgt_b, jws, jwz, jwx, jwy)
        gv = jnp.where(jnp.isnan(gv), 0.0, gv); lv = jnp.where(jnp.isnan(lv), 0.0, lv)
        upd, opt_state = optimizer.update(gv, opt_state, params)
        params = optax.apply_updates(params, upd)
        ep_loss += float(lv); nb += 1
    return params, opt_state, ep_loss/max(nb, 1), rng_key

# ================================================================
# 10. One training run
# ================================================================
def train_one_config(n_layers, rng_seed=42, init_params=None):
    n_p = count_params(n_layers)
    print(f"\n{'='*68}\n  N_LAYERS={n_layers}  params={n_p}  init="
          f"{'LAYERWISE' if init_params is not None else 'MULTI-START'}\n{'='*68}")
    comps = make_layer_components(n_layers)
    blg = comps['batch_loss_and_grad']; qae = comps['qae_batch_chunk']
    bsa = comps['batch_stab_avg']; bfa = comps['batch_fidelity_arr']
    bla = comps['batch_logical_errors_arr']; bpc = comps['batch_probe_chunk']

    pn, pc = rho_noisy_train_stack, rho_clean_train_stack
    pz, px, py = targets_train_stack[:, 0], targets_train_stack[:, 1], targets_train_stack[:, 2]

    def stab_mean(p):
        return batch_stab_avg_chunked(bsa, p, pn)

    def probe(p):
        n = pn.shape[0]; BL=[]; FD=[]; DZ=[]; DX=[]; DY=[]
        for k in range(int(np.ceil(n/CHUNK_EVAL_SIZE))):
            sl = slice(k*CHUNK_EVAL_SIZE, min((k+1)*CHUNK_EVAL_SIZE, n))
            bl, fd, dz, dx, dy = bpc(p, pn[sl], pc[sl], pz[sl], px[sl], py[sl])
            BL.append(np.array(bl)); FD.append(np.array(fd))
            DZ.append(np.array(dz)); DX.append(np.array(dx)); DY.append(np.array(dy))
        BL=np.concatenate(BL); FD=np.concatenate(FD)
        DZ=np.concatenate(DZ); DX=np.concatenate(DX); DY=np.concatenate(DY)
        return (float(BL.mean()), float(BL.std()), float(FD.mean()), float(FD.std()),
                float(DZ.mean()), float(DZ.std()), float(DX.mean()), float(DX.std()),
                float(DY.mean()), float(DY.std()))

    print("  JIT warm-up ...", end=" ", flush=True); t0 = time.time()
    _p0 = jnp.zeros(n_p)
    blg(_p0, pn[:BATCH_SIZE], targets_train_stack[:BATCH_SIZE],
        jnp.float64(1.0), jnp.float64(1.0), jnp.float64(1.0), jnp.float64(1.0))
    bpc(_p0, pn[:CHUNK_EVAL_SIZE], pc[:CHUNK_EVAL_SIZE],
        pz[:CHUNK_EVAL_SIZE], px[:CHUNK_EVAL_SIZE], py[:CHUNK_EVAL_SIZE])
    print(f"done {time.time()-t0:.1f}s")

    spe = int(np.ceil(N_TRAIN_STATES/BATCH_SIZE))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adam(optax.cosine_decay_schedule(LR, TOTAL_EPOCHS*spe)))
    rng = jax.random.PRNGKey(rng_seed)
    best_p = best_opt = None; best_f = -1.0
    best_w = (W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW); best_rng = rng; best_hist = None

    def phase1_round(tp, rr):
        to = optimizer.init(tp); ws, wz, wx, wy = W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW
        h = []
        for ep in range(INIT_EPOCHS):
            if ep % WEIGHT_UPDATE_FREQ == 0:
                tw = compute_target_weights(stab_mean(tp))
                ws, wz, wx, wy = (step_weight(ws, tw[0]), step_weight(wz, tw[1]),
                                  step_weight(wx, tw[2]), step_weight(wy, tw[3]))
            tp, to, _, rr = run_epoch(tp, to, optimizer, ws, wz, wx, wy, rr, blg)
            h.append((ep, 1, *probe(tp)))
        return tp, to, (ws, wz, wx, wy), rr, h

    if init_params is not None:
        print(f"  Phase 1 : layerwise refinement ({INIT_EPOCHS} ep)")
        rng, rr = jax.random.split(rng)
        tp, to, w, rr, h = phase1_round(jnp.asarray(init_params, jnp.float64), rr)
        best_f = h[-1][4]; best_p, best_opt, best_w, best_rng, best_hist = tp, to, w, rr, h
        print(f"  warm-start F={best_f:.4f}")
    else:
        print(f"  Phase 1 : {N_INIT_ROUNDS} rounds x {INIT_EPOCHS} ep")
        for rd in range(N_INIT_ROUNDS):
            rng, sk = jax.random.split(rng); s = rd % 4
            if   s == 0: tp = jax.random.uniform(sk, (n_p,), minval=-np.pi, maxval=np.pi)
            elif s == 1: tp = jax.random.normal(sk, (n_p,))*0.15
            elif s == 2: tp = jax.random.normal(sk, (n_p,))*1.0
            else:        tp = jax.random.uniform(sk, (n_p,), minval=-0.1, maxval=0.1)
            rng, rr = jax.random.split(rng)
            tp, to, w, rr, h = phase1_round(tp, rr)
            f_end = h[-1][4]; star = ""
            if f_end > best_f:
                best_f, best_p, best_opt, best_w, best_rng, best_hist = f_end, tp, to, w, rr, h
                star = " ★"
            print(f"    rd {rd:<2} F_end={f_end:.4f}{star}")
            if star == "": del to
            gc.collect()
        print(f"  best warm-up F={best_f:.4f}")

    keys = ['epoch','phase','loss','loss_std','fid','fid_std','dzl','dzl_std',
            'dxl','dxl_std','dyl','dyl_std']
    hist = {k: [h[i] for h in best_hist] for i, k in enumerate(keys)}

    print(f"  Phase 2 : {MAIN_EPOCHS} continuation epochs")
    params, opt_state = best_p, best_opt
    ws, wz, wx, wy = best_w; rng_key = best_rng
    for ep in range(MAIN_EPOCHS):
        t0 = time.time()
        if ep % WEIGHT_UPDATE_FREQ == 0:
            tw = compute_target_weights(stab_mean(params))
            ws, wz, wx, wy = (step_weight(ws, tw[0]), step_weight(wz, tw[1]),
                              step_weight(wx, tw[2]), step_weight(wy, tw[3]))
        params, opt_state, _, rng_key = run_epoch(params, opt_state, optimizer,
                                                  ws, wz, wx, wy, rng_key, blg)
        m = probe(params); te = INIT_EPOCHS + ep
        hist['epoch'].append(te); hist['phase'].append(2)
        for i, k in enumerate(keys[2:]): hist[k].append(m[i])
        if ep % 10 == 0 or ep == MAIN_EPOCHS-1:
            print(f"    ep {te:<4} loss={m[0]:8.3f} F={m[2]:.4f} "
                  f"ΔZL={m[4]:.4f} ΔXL={m[6]:.4f} ΔYL={m[8]:.4f} {time.time()-t0:.1f}s")
        if ep % 20 == 0: gc.collect()

    hist = {k: np.array(v) for k, v in hist.items()}

    print("  Test eval ...", end=" ", flush=True); t0 = time.time()
    f_qae = eval_fidelity_chunked(params, bfa, rho_noisy_test_stack, rho_clean_test_stack)
    dzl_t, dxl_t, dyl_t = eval_logical_errors_chunked(
        params, bla, rho_noisy_test_stack, zl_test, xl_test, yl_test)
    print(f"done {time.time()-t0:.1f}s  F̄={f_qae.mean():.5f}")

    return dict(n_layers=n_layers, n_params=n_p, params=np.array(params),
                f_qae_test=f_qae, f_std_test=f_std_test, f_noisy_test=f_noisy_test,
                dzl_test=dzl_t, dxl_test=dxl_t, dyl_test=dyl_t, hist=hist)

# ================================================================
# 11. Sweep
# ================================================================
print("\n" + "="*68 + "\n  LAYER SWEEP\n" + "="*68)
sweep = {}; t_sweep = time.time(); prev_nl = None; prev_params = None
for nl in LAYER_SWEEP:
    if (not USE_LAYERWISE) or (prev_params is None) or (nl <= prev_nl):
        sweep[nl] = train_one_config(nl, rng_seed=42+nl)
    else:
        sweep[nl] = train_one_config(nl, rng_seed=42+nl,
                                     init_params=grow_params(prev_params, prev_nl, nl))
    prev_nl = nl; prev_params = sweep[nl]['params']
    gc.collect()
print(f"\n  Total sweep time : {time.time()-t_sweep:.0f}s")

best_nl = max(LAYER_SWEEP, key=lambda nl: sweep[nl]['f_qae_test'].mean())

# ================================================================
# 12. Save
# ================================================================
sd = {'f_noisy_test': f_noisy_test, 'f_std_test': f_std_test,
      'meta_test_thetas': test_thetas, 'meta_test_phis': test_phis,
      'meta_p_noise': np.array(P_NOISE), 'meta_layers': np.array(LAYER_SWEEP),
      'meta_layerwise': np.array(int(USE_LAYERWISE)),
      'best_n_layers': np.array(best_nl), 'best_params': sweep[best_nl]['params'],
      'best_f_qae': np.array(sweep[best_nl]['f_qae_test'].mean())}
for nl, res in sweep.items():
    sd[f'L{nl}_params']     = res['params']
    sd[f'L{nl}_f_qae_test'] = res['f_qae_test']
    sd[f'L{nl}_dzl_test']   = res['dzl_test']
    sd[f'L{nl}_dxl_test']   = res['dxl_test']
    sd[f'L{nl}_dyl_test']   = res['dyl_test']
    for k, v in res['hist'].items(): sd[f'L{nl}_hist_{k}'] = v
np.savez(f"steane_depolar_results_{FILE_TAG}.npz", **sd)
print(f"  Saved : steane_depolar_results_{FILE_TAG}.npz  (best L={best_nl})")

with open(f"steane_depolar_history_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers","global_epoch","phase","baseline_loss","baseline_loss_std",
                "fidelity","fidelity_std","dzl","dzl_std","dxl","dxl_std","dyl","dyl_std"])
    for nl, res in sweep.items():
        h = res['hist']
        for row in zip(h['epoch'], h['phase'], h['loss'], h['loss_std'], h['fid'],
                       h['fid_std'], h['dzl'], h['dzl_std'], h['dxl'], h['dxl_std'],
                       h['dyl'], h['dyl_std']):
            w.writerow([nl, int(row[0]), int(row[1])] + [f"{v:.6f}" for v in row[2:]])
print(f"  Saved : steane_depolar_history_{FILE_TAG}.csv")

with open(f"steane_depolar_results_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers","n_params","F_noisy_mean","F_stdQEC_mean","F_qae_mean",
                "F_qae_std","dzl_mean","dxl_mean","dyl_mean","gap_closed_pct"])
    for nl, res in sweep.items():
        fn, fs, fq = res['f_noisy_test'], res['f_std_test'], res['f_qae_test']
        gap = fs.mean()-fn.mean(); pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
        w.writerow([nl, res['n_params'], f"{fn.mean():.5f}", f"{fs.mean():.5f}",
                    f"{fq.mean():.5f}", f"{fq.std():.5f}", f"{res['dzl_test'].mean():.5f}",
                    f"{res['dxl_test'].mean():.5f}", f"{res['dyl_test'].mean():.5f}", f"{pct:.1f}"])
print(f"  Saved : steane_depolar_results_{FILE_TAG}.csv")

# ================================================================
# 13. 4-panel plot
# ================================================================
colors = ['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']; ls_ops = ['-','--','-','--','-']
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
ax = axes[0, 0]
for (nl, res), c, ls in zip(sweep.items(), colors, ls_ops):
    h = res['hist']
    ax.plot(h['epoch'], h['fid'], color=c, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(h['epoch'], h['fid']-h['fid_std'], h['fid']+h['fid_std'], color=c, alpha=0.12)
ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_test.mean():.4f}")
ax.axhline(f_noisy_test.mean(), color='gray', ls=':', lw=1.5, label=f"No corr {f_noisy_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel(f"Avg fidelity ({N_TRAIN_STATES} train)")
ax.set_title("Training fidelity vs epoch (±1σ)", fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for (nl, res), c, ls in zip(sweep.items(), colors, ls_ops):
    h = res['hist']
    ax.plot(h['epoch'], h['loss'], color=c, ls=ls, lw=2, label=f"L={nl}")
    ax.fill_between(h['epoch'], h['loss']-h['loss_std'], h['loss']+h['loss_std'], color=c, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel("Fixed-weight baseline spectral loss")
ax.set_title("Baseline loss vs epoch", fontweight='bold'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]; x = np.arange(len(LAYER_SWEEP)); bw = 0.22
fn_m = [sweep[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep[nl]['f_std_test'].mean() for nl in LAYER_SWEEP]
fq_m = [sweep[nl]['f_qae_test'].mean() for nl in LAYER_SWEEP]
fq_s = [sweep[nl]['f_qae_test'].std() for nl in LAYER_SWEEP]
ax.bar(x-bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x,    fs_m, bw, label="Standard QEC", color='#3498db', alpha=0.8)
bars = ax.bar(x+bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
ax.errorbar(x+bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4)
for b, v in zip(bars, fq_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test)")
ax.set_title("Test fidelity vs layer count", fontweight='bold'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

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

plt.suptitle(f"Spectral QAE — Steane [[7,1,3]] depolarizing p={P_NOISE} | JAX {jax.default_backend().upper()} | layerwise={USE_LAYERWISE}",
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(f"steane_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : steane_fidelity_{FILE_TAG}.png")

# ================================================================
# 14. Summary + fidelity vs p sweep
# ================================================================
print("\n" + "="*80 + f"\n  FINAL SUMMARY — {N_TEST_STATES} test | p={P_NOISE}\n" + "="*80)
print(f"  {'L':>2} {'Params':>7} {'F_noisy':>9} {'F_std':>9} {'F_QAE':>8} {'±std':>7} "
      f"{'ΔZL':>7} {'ΔXL':>7} {'ΔYL':>7} {'Gap%':>6}")
for nl in LAYER_SWEEP:
    r = sweep[nl]; fn, fs, fq = r['f_noisy_test'], r['f_std_test'], r['f_qae_test']
    gap = fs.mean()-fn.mean(); pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
    print(f"  {nl:>2} {r['n_params']:>7} {fn.mean():>9.5f} {fs.mean():>9.5f} {fq.mean():>8.5f} "
          f"{fq.std():>7.5f} {r['dzl_test'].mean():>7.4f} {r['dxl_test'].mean():>7.4f} "
          f"{r['dyl_test'].mean():>7.4f} {pct:>5.1f}%")
print(f"\n  Best : L={best_nl}  F̄={sweep[best_nl]['f_qae_test'].mean():.5f}")

print("\n" + "="*68 + "\n  FIDELITY vs ERROR RATE p (best model)\n" + "="*68)
best_comps = make_layer_components(best_nl)
best_bfa = best_comps['batch_fidelity_arr']
best_params = jnp.array(sweep[best_nl]['params'])
P_EVAL = [0.01, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25]
fqae, fqae_std, fnoi, fstd = [], [], [], []
for p in P_EVAL:
    rn_np = make_noisy_stack(rho_clean_test_np, p)   # JAX-batched, chunked
    rn_j = jnp.array(rn_np)
    fq = eval_fidelity_chunked(best_params, best_bfa, rn_j, rho_clean_test_stack)
    fn = np.real(np.einsum('nij,nji->n', rn_np, rho_clean_test_np))
    fsd = f_std_chunked(rn_j, rho_clean_test_stack)
    fqae.append(fq.mean()); fqae_std.append(fq.std()); fnoi.append(fn.mean()); fstd.append(fsd.mean())
    print(f"  p={p:>4.2f}  F_QAE={fq.mean():.4f}  F_std={fsd.mean():.4f}  F_noisy={fn.mean():.4f}")
    del rn_np, rn_j; gc.collect()
P_EVAL = np.array(P_EVAL); fqae = np.array(fqae); fqae_std = np.array(fqae_std)
fnoi = np.array(fnoi); fstd = np.array(fstd)

np.savez(f"steane_depolar_fidelity_vs_p_{FILE_TAG}.npz", best_n_layers=np.array(best_nl),
         best_params=sweep[best_nl]['params'], p_eval=P_EVAL,
         f_qae=fqae, f_qae_std=fqae_std, f_noisy=fnoi, f_std=fstd)
with open(f"steane_depolar_fidelity_vs_p_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["p","F_qae","F_qae_std","F_noisy","F_stdQEC"])
    for i in range(len(P_EVAL)):
        w.writerow([f"{P_EVAL[i]:.3f}", f"{fqae[i]:.6f}", f"{fqae_std[i]:.6f}",
                    f"{fnoi[i]:.6f}", f"{fstd[i]:.6f}"])
fig2, ax = plt.subplots(figsize=(9, 6))
ax.plot(P_EVAL, fqae, 'o-', color='#2ecc71', lw=2, label=f"Spectral QAE (L={best_nl})")
ax.fill_between(P_EVAL, fqae-fqae_std, fqae+fqae_std, color='#2ecc71', alpha=0.15)
ax.plot(P_EVAL, fstd, 's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(P_EVAL, fnoi, '^-', color='#e74c3c', lw=2, label="No correction")
ax.axvline(P_NOISE, color='black', ls=':', alpha=0.6)
ax.text(P_NOISE+0.004, 0.6, f"trained @ p={P_NOISE}", fontsize=8)
ax.set_xlabel("Depolarizing error rate p"); ax.set_ylabel("Mean fidelity")
ax.set_title(f"Fidelity vs error rate — best model (L={best_nl})", fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig2.savefig(f"steane_depolar_fidelity_vs_p_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : steane_depolar_fidelity_vs_p_{FILE_TAG}.png / .csv / .npz")
print("\n  Done ✓")