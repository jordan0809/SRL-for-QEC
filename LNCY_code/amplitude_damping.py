"""
Spectral Measurement-Free QEC 
=============================================================
4-qubit LEUNG–NIELSEN–CHUANG–YAMAMOTO (LNCY) [[4,1,2]] code
under AMPLITUDE-DAMPING noise.
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, gc, csv, itertools
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.40"

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
N_CODE  = 4
N_ANC   = 1
N_TOTAL = N_CODE + N_ANC          # 5
SYS_DIM = 2 ** N_CODE             # 16
ANC_DIM = 2 ** N_ANC              # 2
TOT_DIM = 2 ** N_TOTAL            # 32

GAMMA_DAMP = 0.15                 # training amplitude-damping rate

N_TRAIN_STATES = 100
N_TEST_STATES  = 1000
BATCH_SIZE     = 5
CHUNK_EVAL_SIZE = 200
STD_CHUNK       = 200

LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
TOTAL_EPOCHS       = INIT_EPOCHS + MAIN_EPOCHS
WEIGHT_UPDATE_FREQ = 5

W_STAB_HIGH = 5.0;  W_STAB_LOW = 0.5
W_ZL_LOW = 5.0; W_ZL_HIGH = 20.0
W_XL_LOW = 5.0; W_XL_HIGH = 20.0
W_YL_LOW = 5.0; W_YL_HIGH = 20.0
STAB_SAT_LOW  = 0.70
STAB_SAT_HIGH = 0.99
MAX_W_DELTA   = 1.0
W_STAB_BASE = 1.0
W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

LAYER_SWEEP = [1, 2, 3, 4, 5]
USE_LAYERWISE = True
FILE_TAG = "layerwise" if USE_LAYERWISE else "multistart"
PROBE_N = N_TRAIN_STATES

PETZ_COLOR = '#f39c12'

print(f"""
  LNCY [[4,1,2]] code — amplitude damping (per-qubit Kraus)
  gamma_train = {GAMMA_DAMP}
  Layer sweep {LAYER_SWEEP}, USE_LAYERWISE={USE_LAYERWISE}
  Train {N_TRAIN_STATES} / Test {N_TEST_STATES}, batch {BATCH_SIZE}, LR {LR}
""")

# ================================================================
# 2. Pauli primitives + LNCY [[4,1,2]] operators
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0,1],[1,0]], dtype=complex)
Y_ = np.array([[0,-1j],[1j,0]], dtype=complex)
Z_ = np.array([[1,0],[0,-1]], dtype=complex)
def kron_list(ops): return reduce(np.kron, ops)
def op_on(qubits, mat, n=N_CODE):
    ops = [I2]*n
    for q in qubits: ops[q] = mat
    return kron_list(ops)
def pauli_string(s):
    g = {'I': I2, 'X': X_, 'Y': Y_, 'Z': Z_}
    return kron_list([g[c] for c in s])

# --- LNCY code stabilizers (3 generators) ---
S_mats = [pauli_string('ZZII'),
          pauli_string('IIZZ'),
          pauli_string('XXXX')]
N_STAB = len(S_mats)
for a in range(N_STAB):
    for b in range(N_STAB):
        assert np.allclose(S_mats[a]@S_mats[b], S_mats[b]@S_mats[a])

# --- logical operators ---
XL_mat = pauli_string('IIXX')
ZL_mat = pauli_string('ZIZI')
YL_mat = 1j * XL_mat @ ZL_mat
for S in S_mats:
    assert np.allclose(XL_mat@S, S@XL_mat)
    assert np.allclose(ZL_mat@S, S@ZL_mat)
    assert np.allclose(YL_mat@S, S@YL_mat)
assert np.allclose(XL_mat@ZL_mat, -ZL_mat@XL_mat)
assert np.allclose(YL_mat, YL_mat.conj().T)
print("  ✓ LNCY stabilizers (ZZII, IIZZ, XXXX) + logicals (XL=IIXX, ZL=ZIZI, YL=iXLZL) verified")

S_j  = [jnp.array(m) for m in S_mats]
XLj, YLj, ZLj = jnp.array(XL_mat), jnp.array(YL_mat), jnp.array(ZL_mat)

# --- code projector + logical basis ---
_P = np.eye(SYS_DIM, dtype=complex)
for S in S_mats: _P = _P @ ((np.eye(SYS_DIM)+S)/2)
_ev, _evec = np.linalg.eigh(_P)
_code = _evec[:, np.isclose(_ev, 1.0)]
assert _code.shape[1] == 2
_e2, _v2 = np.linalg.eigh(_code.conj().T @ ZL_mat @ _code)
_k0 = _code @ _v2[:,0]; _k1 = _code @ _v2[:,1]
ket_0L, ket_1L = (_k1, _k0) if _e2[0] < _e2[1] else (_k0, _k1)
ket_0L /= np.linalg.norm(ket_0L); ket_1L /= np.linalg.norm(ket_1L)
print("  ✓ LNCY [[4,1,2]] logical basis computed "
      f"(⟨0L|ZL|0L⟩={np.real(ket_0L.conj()@ZL_mat@ket_0L):+.2f}, "
      f"⟨1L|ZL|1L⟩={np.real(ket_1L.conj()@ZL_mat@ket_1L):+.2f})")

def logical_state(theta, phi):
    return np.cos(theta/2)*ket_0L + np.exp(1j*phi)*np.sin(theta/2)*ket_1L
def state_to_dm(psi):
    psi = psi.reshape(-1,1); return psi @ psi.conj().T

ANC_0     = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0,0].set(1.0)
FRESH_0   = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0,0].set(1.0)

# ================================================================
# 3. Amplitude-damping channel (per-qubit Kraus)
# ================================================================
def ad_single_kraus(gamma):
    K0 = np.array([[1, 0], [0, np.sqrt(1.0 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

def ad_full_kraus(gamma, n=N_CODE):
    singles = ad_single_kraus(gamma)
    kraus = []
    for combo in itertools.product([0, 1], repeat=n):
        kraus.append(kron_list([singles[c] for c in combo]))
    return kraus

KRAUS_TRAIN = ad_full_kraus(GAMMA_DAMP)   # 2^N_CODE Kraus operators

def apply_channel(rho, kraus):
    out = np.zeros_like(rho)
    for k in kraus:
        out += k @ rho @ k.conj().T
    return out

def apply_channel_batch(rho_stack, kraus):
    out = np.zeros_like(rho_stack)
    for k in kraus:
        out += np.einsum('ij,njk,lk->nil', k, rho_stack, k.conj())
    return out

def amplitude_damping_channel(rho, kraus=KRAUS_TRAIN):
    return apply_channel(rho, kraus)

# ================================================================
# 4. Standard LNCY recovery (first-order amplitude-damping QEC)
#    Projective code recovery + single-jump correction, trace-completed.
# ================================================================
def build_lncy_recovery():
    sig_minus = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|
    k0 = ket_0L.reshape(-1, 1); k1 = ket_1L.reshape(-1, 1)
    P_code = k0 @ k0.conj().T + k1 @ k1.conj().T

    R_list = [P_code.copy()]              # no-jump: project back to code space
    P_corr = P_code.copy()
    for q in range(N_CODE):
        Aq = op_on([q], sig_minus)        # jump on qubit q
        v0 = Aq @ ket_0L; v1 = Aq @ ket_1L
        n0 = np.linalg.norm(v0); n1 = np.linalg.norm(v1)
        if n0 < 1e-12 or n1 < 1e-12:
            continue
        v0 = (v0 / n0).reshape(-1, 1)
        v1 = (v1 / n1).reshape(-1, 1)
        R = k0 @ v0.conj().T + k1 @ v1.conj().T
        R_list.append(R)
        P_corr += v0 @ v0.conj().T + v1 @ v1.conj().T

    # completion (uncorrectable subspace -> dump to |0_L>) for trace preservation
    Q = np.eye(SYS_DIM, dtype=complex) - P_corr
    ev, evec = np.linalg.eigh(Q)
    for i in range(len(ev)):
        if ev[i] > 1e-9:
            phi = evec[:, i].reshape(-1, 1)
            R_list.append(k0 @ phi.conj().T)

    R_stack = np.stack(R_list)
    check = sum(R.conj().T @ R for R in R_list)
    assert np.allclose(check, np.eye(SYS_DIM), atol=1e-8), "recovery not trace-preserving"
    return R_stack

CORR_M = jnp.array(build_lncy_recovery())
print(f"  ✓ LNCY recovery built ({CORR_M.shape[0]} Kraus operators, trace-preserving)")

@jax.jit
def std_qec_jax(rho):
    def one(M): return M @ rho @ jnp.conj(M.T)
    return jnp.sum(jax.vmap(one)(CORR_M), axis=0)

@jax.jit
def std_fid_chunk(rn_chunk, rc_chunk):
    ro = jax.vmap(std_qec_jax)(rn_chunk)
    return jax.vmap(lambda a, b: jnp.real(jnp.trace(a @ b)))(ro, rc_chunk)

# ================================================================
# 4b. Petz (transpose) recovery map — general Kraus channel
#     R(rho) = sig^{1/2} N†( N(sig)^{-1/2} rho N(sig)^{-1/2} ) sig^{1/2}
#     with sigma = code-space maximally mixed state, N†(A)=Σ Kᵢ† A Kᵢ.
# ================================================================
_P_CODE = _P
_TR_P   = float(np.real(np.trace(_P_CODE)))     # = 2 (code dim)

def make_petz_fid(kraus, chunk=STD_CHUNK):
    K = [np.asarray(k) for k in kraus]
    sig      = _P_CODE / _TR_P
    sig_half = _P_CODE / np.sqrt(_TR_P)          # (P/2)^{1/2} = P/sqrt(2)
    Nsig = np.zeros_like(sig)
    for k in K: Nsig += k @ sig @ k.conj().T
    Nsig = (Nsig + Nsig.conj().T) / 2.0
    ev, evec = np.linalg.eigh(Nsig)
    tol = 1e-12 * float(ev.max())
    inv_sqrt = np.where(ev > tol, 1.0/np.sqrt(np.where(ev > tol, ev, 1.0)), 0.0)
    Nsig_inv_half = (evec * inv_sqrt) @ evec.conj().T

    Nih = jnp.array(Nsig_inv_half); Sh = jnp.array(sig_half)
    Kj  = jnp.array(np.stack(K))

    @jax.jit
    def _petz_fid_batch(rn, rc):
        def per(rho, rcl):
            inner = Nih @ rho @ Nih
            def adj_term(k): return jnp.conj(k.T) @ inner @ k
            adj = jnp.sum(jax.vmap(adj_term)(Kj), axis=0)
            out = Sh @ adj @ Sh
            out = out / jnp.real(jnp.trace(out))
            return jnp.real(jnp.einsum('ij,ji->', out, rcl))
        return jax.vmap(per)(rn, rc)

    def evalf(rn, rc):
        n = rn.shape[0]; out = np.empty(n)
        for kk in range(int(np.ceil(n/chunk))):
            sl = slice(kk*chunk, min((kk+1)*chunk, n))
            out[sl] = np.array(_petz_fid_batch(rn[sl], rc[sl]))
        return out
    return evalf

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
                    qml.CRX(params[idx], wires=[c,t]); idx += 1
                    qml.CRY(params[idx], wires=[c,t]); idx += 1
                    qml.CRZ(params[idx], wires=[c,t]); idx += 1
        for q in range(n_total):
            qml.RY(params[idx], wires=q); idx += 1
            qml.RZ(params[idx], wires=q); idx += 1
    return idx
def grow_params(params_prev, prev_nl, nl):
    extra = count_params(nl) - count_params(prev_nl)
    return np.concatenate([np.asarray(params_prev), np.zeros(extra)])
def partial_trace_system(rho):
    r = rho.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM); return jnp.trace(r, axis1=0, axis2=2)
def partial_trace_ancilla(rho):
    r = rho.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM); return jnp.trace(r, axis1=1, axis2=3)

# ================================================================
# 6. Per-layer factory  (IDENTICAL logic; N_STAB stabilizers)
# ================================================================
def make_layer_components(n_layers):
    n_p = count_params(n_layers)
    def circuit(params): model_b_ansatz(params, n_layers)
    encoder_fn = qml.matrix(circuit, wire_order=range(N_TOTAL))

    def qae_single(params, rho_code):
        V = encoder_fn(params); Vd = jnp.conj(V.T)
        rho = jnp.kron(rho_code.astype(complex), ANC_0); rho = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho = jnp.kron(FRESH_0, rho_a); rho = Vd @ rho @ V
        return partial_trace_ancilla(rho)

    def _fwd(V, Vd, rho_code):
        rho = jnp.kron(rho_code, ANC_0); rho = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho = jnp.kron(FRESH_0, rho_a); rho = Vd @ rho @ V
        return partial_trace_ancilla(rho)
    _vmapped = jax.vmap(_fwd, in_axes=(None, None, 0))

    @jax.jit
    def qae_batch_chunk(params, rho_chunk):
        V = encoder_fn(params); Vd = jnp.conj(V.T)
        return _vmapped(V, Vd, rho_chunk)

    @jax.jit
    def loss_and_grad(params, rho_noisy, target, ws, wz, wx, wy):
        def _loss(p):
            rho = qae_single(p, rho_noisy)
            s = jnp.mean(jnp.stack([jnp.real(jnp.trace(S @ rho)) for S in S_j]))
            zl = jnp.real(jnp.trace(ZLj@rho)); xl = jnp.real(jnp.trace(XLj@rho)); yl = jnp.real(jnp.trace(YLj@rho))
            return (ws*(-s) + wz*(target[0]-zl)**2 + wx*(target[1]-xl)**2 + wy*(target[2]-yl)**2)
        return jax.value_and_grad(_loss)(params)

    @jax.jit
    def batch_stab_avg(params, rho_chunk):
        out = qae_batch_chunk(params, rho_chunk)
        def per(r): return jnp.mean(jnp.stack([jnp.real(jnp.trace(S@r)) for S in S_j]))
        return jnp.mean(jax.vmap(per)(out))

    @jax.jit
    def batch_fidelity_arr(params, rn, rc):
        out = qae_batch_chunk(params, rn)
        return jax.vmap(lambda ro, r: jnp.real(jnp.trace(ro @ r)))(out, rc)

    @jax.jit
    def batch_logical_errors_arr(params, rn, zt, xt, yt):
        out = qae_batch_chunk(params, rn)
        def per(rho, z, x, y):
            return (jnp.abs(z - jnp.real(jnp.trace(ZLj@rho))),
                    jnp.abs(x - jnp.real(jnp.trace(XLj@rho))),
                    jnp.abs(y - jnp.real(jnp.trace(YLj@rho))))
        return jax.vmap(per)(out, zt, xt, yt)

    @jax.jit
    def batch_baseline_loss_arr(params, rn, zt, xt, yt):
        out = qae_batch_chunk(params, rn)
        def per(rho, z, x, y):
            s = jnp.mean(jnp.stack([jnp.real(jnp.trace(S@rho)) for S in S_j]))
            zl = jnp.real(jnp.trace(ZLj@rho)); xl = jnp.real(jnp.trace(XLj@rho)); yl = jnp.real(jnp.trace(YLj@rho))
            return (W_STAB_BASE*(-s) + W_ZL_BASE*(z-zl)**2 + W_XL_BASE*(x-xl)**2 + W_YL_BASE*(y-yl)**2)
        return jax.vmap(per)(out, zt, xt, yt)

    return dict(n_params=n_p, qae_single=qae_single, qae_batch_chunk=qae_batch_chunk,
                loss_and_grad=loss_and_grad, batch_stab_avg=batch_stab_avg,
                batch_fidelity_arr=batch_fidelity_arr,
                batch_logical_errors_arr=batch_logical_errors_arr,
                batch_baseline_loss_arr=batch_baseline_loss_arr)

# ================================================================
# 7. Datasets
# ================================================================
def fibonacci_sphere(n):
    g = (1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/n), (2*np.pi*i/g) % (2*np.pi)) for i in range(n)]
def generate_dataset(n, kraus=KRAUS_TRAIN):
    data = []
    for th, ph in fibonacci_sphere(n):
        rc = state_to_dm(logical_state(th, ph)); rn = amplitude_damping_channel(rc, kraus)
        data.append(dict(theta=th, phi=ph, rho_clean=rc, rho_noisy=rn,
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

def eval_std_fidelity(rn_stack, rc_stack, chunk=CHUNK_EVAL_SIZE):
    n = rn_stack.shape[0]; out = np.empty(n)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out[sl] = np.array(std_fid_chunk(rn_stack[sl], rc_stack[sl]))
    return out

print(f"  Precomputing standard LNCY QEC on {N_TEST_STATES} test states (vectorized) ...")
f_std_test   = eval_std_fidelity(rho_noisy_test_stack, rho_clean_test_stack)
f_noisy_test = np.real(np.einsum('nij,nji->n', rho_noisy_test_np, rho_clean_test_np))
print(f"  No-correction F = {f_noisy_test.mean():.5f} | Std QEC F = {f_std_test.mean():.5f}")

print(f"  Precomputing Petz recovery on {N_TEST_STATES} test states ...")
petz_fid_fn = make_petz_fid(KRAUS_TRAIN)
f_petz_test = petz_fid_fn(rho_noisy_test_stack, rho_clean_test_stack)
print(f"  Petz map F = {f_petz_test.mean():.5f}")

# ================================================================
# 8. VECTORIZED chunked eval helpers
# ================================================================
def eval_fidelity_batched(bfa, params, rn_stack, rc_stack, chunk=CHUNK_EVAL_SIZE):
    n = rn_stack.shape[0]; out = []
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out.append(np.array(bfa(params, rn_stack[sl], rc_stack[sl])))
    return np.concatenate(out)

def eval_logical_batched(bla, params, rn_stack, zt, xt, yt, chunk=CHUNK_EVAL_SIZE):
    n = rn_stack.shape[0]; DZ=[]; DX=[]; DY=[]
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        dz, dx, dy = bla(params, rn_stack[sl], zt[sl], xt[sl], yt[sl])
        DZ.append(np.array(dz)); DX.append(np.array(dx)); DY.append(np.array(dy))
    return np.concatenate(DZ), np.concatenate(DX), np.concatenate(DY)

# ================================================================
# 9. Adaptive weights + epoch  (IDENTICAL)
# ================================================================
def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab-STAB_SAT_LOW)/(STAB_SAT_HIGH-STAB_SAT_LOW), 0, 1))
    return (W_STAB_HIGH-(W_STAB_HIGH-W_STAB_LOW)*sat,
            W_ZL_LOW+(W_ZL_HIGH-W_ZL_LOW)*sat,
            W_XL_LOW+(W_XL_HIGH-W_XL_LOW)*sat,
            W_YL_LOW+(W_YL_HIGH-W_YL_LOW)*sat)
def step_weight(cur, tgt): return cur + float(np.clip(tgt-cur, -MAX_W_DELTA, MAX_W_DELTA))

def run_epoch(params, opt_state, optimizer, ws, wz, wx, wy, rng_key, lg):
    jws, jwz, jwx, jwy = map(jnp.float64, (ws, wz, wx, wy))
    steps = int(np.ceil(N_TRAIN_STATES/BATCH_SIZE))
    rng_key, sk = jax.random.split(rng_key)
    idx = np.array(jax.random.permutation(sk, N_TRAIN_STATES))
    ep_loss = 0.0; nb = 0
    for st in range(steps):
        b = idx[st*BATCH_SIZE:(st+1)*BATCH_SIZE]
        g = jnp.zeros_like(params); l = 0.0
        for i in b:
            lv, gv = lg(params, rho_noisy_train_stack[i], targets_train_stack[i], jws, jwz, jwx, jwy)
            gv = jnp.where(jnp.isnan(gv), 0.0, gv); lv = jnp.where(jnp.isnan(lv), 0.0, lv)
            g += gv; l += float(lv)
        upd, opt_state = optimizer.update(g/len(b), opt_state, params)
        params = optax.apply_updates(params, upd); ep_loss += l/len(b); nb += 1
    return params, opt_state, ep_loss/max(nb, 1), rng_key

# ================================================================
# 10. One training run  (IDENTICAL)
# ================================================================
def train_one_config(n_layers, rng_seed=42, init_params=None):
    n_p = count_params(n_layers)
    print(f"\n{'='*68}\n  N_LAYERS={n_layers}  params={n_p}  init="
          f"{'LAYERWISE' if init_params is not None else 'MULTI-START'}\n{'='*68}")
    comps = make_layer_components(n_layers)
    lg  = comps['loss_and_grad']; qae = comps['qae_batch_chunk']
    bsa = comps['batch_stab_avg']; bfa = comps['batch_fidelity_arr']
    bla = comps['batch_logical_errors_arr']; bba = comps['batch_baseline_loss_arr']

    pn, pc = rho_noisy_train_stack, rho_clean_train_stack
    pz, px, py = targets_train_stack[:,0], targets_train_stack[:,1], targets_train_stack[:,2]

    def probe(p):
        bl = np.asarray(bba(p, pn, pz, px, py)); fd = np.asarray(bfa(p, pn, pc))
        dz, dx, dy = bla(p, pn, pz, px, py); dz, dx, dy = map(np.asarray, (dz, dx, dy))
        return (float(bl.mean()), float(bl.std()), float(fd.mean()), float(fd.std()),
                float(dz.mean()), float(dz.std()), float(dx.mean()), float(dx.std()),
                float(dy.mean()), float(dy.std()))

    print("  JIT warm-up ...", end=" ", flush=True); t0 = time.time()
    _p0 = jnp.zeros(n_p)
    lg(_p0, pn[0], targets_train_stack[0], jnp.float64(1.0), jnp.float64(1.0),
       jnp.float64(1.0), jnp.float64(1.0)); bba(_p0, pn, pz, px, py)
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
                tw = compute_target_weights(float(bsa(tp, pn)))
                ws, wz, wx, wy = (step_weight(ws, tw[0]), step_weight(wz, tw[1]),
                                  step_weight(wx, tw[2]), step_weight(wy, tw[3]))
            tp, to, _, rr = run_epoch(tp, to, optimizer, ws, wz, wx, wy, rr, lg)
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
            tw = compute_target_weights(float(bsa(params, pn)))
            ws, wz, wx, wy = (step_weight(ws, tw[0]), step_weight(wz, tw[1]),
                              step_weight(wx, tw[2]), step_weight(wy, tw[3]))
        params, opt_state, _, rng_key = run_epoch(params, opt_state, optimizer,
                                                  ws, wz, wx, wy, rng_key, lg)
        m = probe(params); te = INIT_EPOCHS + ep
        hist['epoch'].append(te); hist['phase'].append(2)
        for i, k in enumerate(keys[2:]): hist[k].append(m[i])
        if ep % 10 == 0 or ep == MAIN_EPOCHS-1:
            print(f"    ep {te:<4} loss={m[0]:8.3f} F={m[2]:.4f} "
                  f"ΔZL={m[4]:.4f} ΔXL={m[6]:.4f} ΔYL={m[8]:.4f} {time.time()-t0:.1f}s")
        if ep % 20 == 0: gc.collect()

    hist = {k: np.array(v) for k, v in hist.items()}

    print("  Test eval (vectorized) ...", end=" ", flush=True); t0 = time.time()
    f_qae = eval_fidelity_batched(bfa, params, rho_noisy_test_stack, rho_clean_test_stack)
    dzl_t, dxl_t, dyl_t = eval_logical_batched(
        bla, params, rho_noisy_test_stack, zl_test, xl_test, yl_test)
    print(f"done {time.time()-t0:.1f}s  F̄={f_qae.mean():.5f}")

    return dict(n_layers=n_layers, n_params=n_p, params=np.array(params),
                f_qae_test=f_qae, f_std_test=f_std_test, f_noisy_test=f_noisy_test,
                f_petz_test=f_petz_test,
                dzl_test=dzl_t, dxl_test=dxl_t, dyl_test=dyl_t, hist=hist,
                bfa=bfa, bla=bla)

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
print(f"\n  Total sweep time : {time.time()-t_sweep:.0f}s")
best_nl = max(LAYER_SWEEP, key=lambda nl: sweep[nl]['f_qae_test'].mean())

# ================================================================
# 12. Save
# ================================================================
sd = {'f_noisy_test': f_noisy_test, 'f_std_test': f_std_test,
      'f_petz_test': f_petz_test,
      'meta_test_thetas': test_thetas, 'meta_test_phis': test_phis,
      'meta_gamma': np.array(GAMMA_DAMP), 'meta_layers': np.array(LAYER_SWEEP),
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
np.savez(f"ampdamp4q_results_{FILE_TAG}.npz", **sd)
print(f"  Saved : ampdamp4q_results_{FILE_TAG}.npz  (best L={best_nl})")

with open(f"ampdamp4q_history_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers","global_epoch","phase","baseline_loss","baseline_loss_std",
                "fidelity","fidelity_std","dzl","dzl_std","dxl","dxl_std","dyl","dyl_std"])
    for nl, res in sweep.items():
        h = res['hist']
        for row in zip(h['epoch'], h['phase'], h['loss'], h['loss_std'], h['fid'],
                       h['fid_std'], h['dzl'], h['dzl_std'], h['dxl'], h['dxl_std'],
                       h['dyl'], h['dyl_std']):
            w.writerow([nl, int(row[0]), int(row[1])] + [f"{v:.6f}" for v in row[2:]])
print(f"  Saved : ampdamp4q_history_{FILE_TAG}.csv")

with open(f"ampdamp4q_results_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers","n_params","F_noisy_mean","F_stdQEC_mean","F_petz_mean",
                "F_qae_mean","F_qae_std","dzl_mean","dxl_mean","dyl_mean","gap_closed_pct"])
    for nl, res in sweep.items():
        fn, fs, fq = res['f_noisy_test'], res['f_std_test'], res['f_qae_test']
        fp = res['f_petz_test']
        gap = fs.mean()-fn.mean(); pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
        w.writerow([nl, res['n_params'], f"{fn.mean():.5f}", f"{fs.mean():.5f}",
                    f"{fp.mean():.5f}", f"{fq.mean():.5f}", f"{fq.std():.5f}",
                    f"{res['dzl_test'].mean():.5f}", f"{res['dxl_test'].mean():.5f}",
                    f"{res['dyl_test'].mean():.5f}", f"{pct:.1f}"])
print(f"  Saved : ampdamp4q_results_{FILE_TAG}.csv")

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
ax.axhline(f_petz_test.mean(), color=PETZ_COLOR, ls='-.', lw=1.5, label=f"Petz map {f_petz_test.mean():.4f}")
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

plt.suptitle(f"Spectral QAE — LNCY [[4,1,2]] amplitude damping "
             f"(γ={GAMMA_DAMP}) | JAX {jax.default_backend().upper()}",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(f"ampdamp4q_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : ampdamp4q_fidelity_{FILE_TAG}.png")

# ================================================================
# 14. Fidelity vs damping rate gamma (best model) — vectorized
# ================================================================
print("\n" + "="*68 + "\n  FIDELITY vs DAMPING RATE γ  (best model)\n" + "="*68)
best_comps = make_layer_components(best_nl)
best_bfa = best_comps['batch_fidelity_arr']
best_params = jnp.array(sweep[best_nl]['params'])
GAMMA_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
fqae, fqae_std, fnoi, fstd, fpetz = [], [], [], [], []
for g in GAMMA_SWEEP:
    kraus_g = ad_full_kraus(g)
    rn_np = apply_channel_batch(rho_clean_test_np, kraus_g)
    rn_j  = jnp.array(rn_np)
    fq  = eval_fidelity_batched(best_bfa, best_params, rn_j, rho_clean_test_stack)
    fn  = np.real(np.einsum('nij,nji->n', rn_np, rho_clean_test_np))
    fsd = eval_std_fidelity(rn_j, rho_clean_test_stack)
    fpz = make_petz_fid(kraus_g)(rn_j, rho_clean_test_stack)
    fqae.append(fq.mean()); fqae_std.append(fq.std())
    fnoi.append(fn.mean()); fstd.append(fsd.mean()); fpetz.append(fpz.mean())
    print(f"  γ={g:>4.2f}  F_QAE={fq.mean():.4f}  F_petz={fpz.mean():.4f}  "
          f"F_std={fsd.mean():.4f}  F_noisy={fn.mean():.4f}")
    del rn_np, rn_j; gc.collect()
GAMMA_SWEEP = np.array(GAMMA_SWEEP); fqae = np.array(fqae); fqae_std = np.array(fqae_std)
fnoi = np.array(fnoi); fstd = np.array(fstd); fpetz = np.array(fpetz)

np.savez(f"ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.npz", best_n_layers=np.array(best_nl),
         best_params=sweep[best_nl]['params'], gamma=GAMMA_SWEEP,
         f_qae=fqae, f_qae_std=fqae_std, f_noisy=fnoi, f_std=fstd, f_petz=fpetz)
with open(f"ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["gamma","F_qae","F_qae_std","F_noisy","F_stdQEC","F_petz"])
    for i in range(len(GAMMA_SWEEP)):
        w.writerow([f"{GAMMA_SWEEP[i]:.3f}", f"{fqae[i]:.6f}", f"{fqae_std[i]:.6f}",
                    f"{fnoi[i]:.6f}", f"{fstd[i]:.6f}", f"{fpetz[i]:.6f}"])

fig, ax = plt.subplots(figsize=(9,6))
ax.plot(GAMMA_SWEEP, fqae, 'o-', color='#2ecc71', lw=2, label=f"Spectral QAE (L={best_nl})")
ax.fill_between(GAMMA_SWEEP, fqae-fqae_std, fqae+fqae_std, color='#2ecc71', alpha=0.15)
ax.plot(GAMMA_SWEEP, fstd,  's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(GAMMA_SWEEP, fpetz, 'd-.', color=PETZ_COLOR, lw=2, label="Petz map")
ax.plot(GAMMA_SWEEP, fnoi,  '^-',  color='#e74c3c', lw=2, label="No correction")
ax.axvline(GAMMA_DAMP, color='black', ls=':', alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(GAMMA_DAMP + 0.005, _ylo + 0.05*(_yhi-_ylo),
        f"trained @ γ={GAMMA_DAMP}", fontsize=8, va='bottom')
ax.set_xlabel("Amplitude-damping rate  γ"); ax.set_ylabel("Mean fidelity")
ax.set_title(f"LNCY [[4,1,2]] amplitude damping — best model (L={best_nl}, trained @ γ={GAMMA_DAMP})",
             fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); fig.savefig(f"ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : ampdamp4q_fidelity_vs_gamma_{FILE_TAG}.png")
print("\n  Done ✓")