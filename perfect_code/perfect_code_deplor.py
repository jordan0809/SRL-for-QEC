"""
Layer-sweep experiment for QAE Model B — [[5,1,3]] Perfect Code
================================================================
Optimisations vs. baseline:
  1. RY + RZ single-qubit layers (no RX)  and  CRX + CRY + CRZ entanglers.
  2. jax.vmap over states — V built once, linear algebra vectorised.
  3. Adaptive weights updated every WEIGHT_UPDATE_FREQ epochs.
  4. Final 1000-state test uses chunked vmap (bounded peak memory).
  5. Logical spectrum errors (ΔZL, ΔXL, ΔYL) tracked EVERY epoch
     across BOTH phases and reported in the final per-layer summary.
  6. Multi-start init restricted to wide-range sampling only.
  7. Loss tracking uses a FIXED-weight baseline spectral loss (training
     still uses adaptive weights) for consistent comparison across epochs.
  8. FULL history: Phase 1 (best warm-up trajectory) AND Phase 2 are saved
     with a continuous global epoch index + phase marker, so the entire
     training curve (epoch 0 → final) is reproducible from the .npz.
  9. MONITORING FIX: per-epoch metrics are now computed over ALL 100
     training states (previously the first 10, which were a biased,
     single-pole slice of the Fibonacci sphere). Both MEAN and STD are
     recorded for every metric at every epoch.
 10. LAYERWISE WARM-START: for L>1 the model is grown from the previous
     trained solution by appending identity (zero-angle) blocks, so each
     deeper model starts numerically identical to the shallower one.
 11. Best params (per layer + overall best model) are saved to disk.
 12. Extra study: fidelity vs depolarizing error rate for the best model.
     (F_theory removed — it was a misleading 0/1-error bound.)

Sweep:  N_LAYERS ∈ {1, 2, 3, 4, 5}
Output: perfect_fidelity.png
        perfect_results.npz
        perfect_results.csv
        perfect_history.csv
        perfect_fidelity_vs_p.png / .csv / .npz
"""

# ──────────────────────────────────────────────────────────────
# 0. Imports & JAX config
# ──────────────────────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import os, time, gc, csv
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import reduce

print(f"  JAX backend : {jax.default_backend().upper()}")
print(f"  JAX devices : {jax.devices()}")

# ──────────────────────────────────────────────────────────────
# 1. Global constants
# ──────────────────────────────────────────────────────────────
N_CODE   = 5
N_ANC    = 1
N_TOTAL  = N_CODE + N_ANC   # 6
SYS_DIM  = 2 ** N_CODE      # 32
ANC_DIM  = 2 ** N_ANC       # 2
TOT_DIM  = 2 ** N_TOTAL     # 64

P_NOISE        = 0.05
N_TRAIN_STATES = 100
N_TEST_STATES  = 1_000
BATCH_SIZE     = 5

CHUNK_EVAL_SIZE = 200

#small train states experiment
#N_TRAIN_STATES = 6
#BATCH_SIZE = 1

# Training schedule
LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
TOTAL_EPOCHS       = INIT_EPOCHS + MAIN_EPOCHS   # 200
WEIGHT_UPDATE_FREQ = 5

# Adaptive weight bounds
W_STAB_HIGH   = 5.0;  W_STAB_LOW  = 0.5
W_ZL_LOW      = 5.0;  W_ZL_HIGH    = 20.0  
W_XL_LOW      = 5.0;  W_XL_HIGH   = 20.0
W_YL_LOW      = 5.0;  W_YL_HIGH   = 20.0
STAB_SAT_LOW  = 0.70
STAB_SAT_HIGH = 0.99
MAX_W_DELTA   = 1.0

# Baseline (fixed) spectral-loss weights — used ONLY for consistent
# loss tracking/logging/plotting. Training still uses adaptive weights.
W_STAB_BASE = 1.0
W_ZL_BASE   = 10.0
W_XL_BASE   = 10.0
W_YL_BASE   = 10.0

LAYER_SWEEP = [1, 2, 3, 4, 5]

# Layerwise warm-start: grow L -> L+1 from the previous trained solution
# by appending identity (zero-angle) blocks. Set False to revert to the
# original independent multi-start per layer.
USE_LAYERWISE = True
#USE_LAYERWISE = False

# For file names
if USE_LAYERWISE:
    FILE_TAG = "layerwise"
else:
    FILE_TAG = "multistart"

# Per-epoch monitoring now uses the FULL training set (all 100 states).
PROBE_N = N_TRAIN_STATES

mat_kb = 64 * 64 * 16 / 1024
print(f"\n  Memory budget (complex128, 64×64 = {mat_kb:.0f} KB/matrix):")
for n, lbl in [(PROBE_N, f"probe({PROBE_N})"),
               (N_TRAIN_STATES, f"train({N_TRAIN_STATES})"),
               (CHUNK_EVAL_SIZE, f"chunk({CHUNK_EVAL_SIZE})")]:
    peak_mb = n * mat_kb * 6 / 1024
    print(f"    vmap over {n:>4} states → ~{peak_mb:6.1f} MB peak  [{lbl}]")
print()

# ──────────────────────────────────────────────────────────────
# 2. Pauli / code primitives
# ──────────────────────────────────────────────────────────────
I2  = np.eye(2, dtype=complex)
X_m = np.array([[0,  1 ], [1,   0]], dtype=complex)
Y_m = np.array([[0, -1j], [1j,  0]], dtype=complex)
Z_m = np.array([[1,  0 ], [0,  -1]], dtype=complex)

def kron_list(ops):
    return reduce(np.kron, ops)

def pauli_string(s):
    g = {'I': I2, 'X': X_m, 'Y': Y_m, 'Z': Z_m}
    return kron_list([g[c] for c in s])

S1_mat = pauli_string('XZZXI');  S2_mat = pauli_string('IXZZX')
S3_mat = pauli_string('XIXZZ');  S4_mat = pauli_string('ZXIXZ')
stabilizers_mat = [S1_mat, S2_mat, S3_mat, S4_mat]
XL_mat = pauli_string('XXXXX')
ZL_mat = pauli_string('ZZZZZ')
YL_mat = pauli_string('YYYYY')

S1j = jnp.array(S1_mat); S2j = jnp.array(S2_mat)
S3j = jnp.array(S3_mat); S4j = jnp.array(S4_mat)
ZLj = jnp.array(ZL_mat); XLj = jnp.array(XL_mat); YLj = jnp.array(YL_mat)

ANC_0       = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0, 0].set(1.0)
FRESH_00000 = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0, 0].set(1.0)

# ──────────────────────────────────────────────────────────────
# 3. Logical states
# ──────────────────────────────────────────────────────────────
def compute_logical_states():
    dim  = 2 ** N_CODE
    proj = np.eye(dim, dtype=complex)
    for S in stabilizers_mat:
        proj = proj @ ((np.eye(dim) + S) / 2)
    evals, evecs = np.linalg.eigh(proj)
    basis = [evecs[:, i] for i in range(dim) if abs(evals[i] - 1.0) < 1e-6]
    assert len(basis) == 2
    v0, v1 = basis
    zl_code = np.array([[v0 @ ZL_mat @ v0, v0 @ ZL_mat @ v1],
                         [v1 @ ZL_mat @ v0, v1 @ ZL_mat @ v1]])
    ev, evec = np.linalg.eigh(zl_code)
    if ev[0] > ev[1]:
        k0 = evec[0, 0]*v0 + evec[1, 0]*v1
        k1 = evec[0, 1]*v0 + evec[1, 1]*v1
    else:
        k0 = evec[0, 1]*v0 + evec[1, 1]*v1
        k1 = evec[0, 0]*v0 + evec[1, 0]*v1
    k0 /= np.linalg.norm(k0); k1 /= np.linalg.norm(k1)
    return k0, k1

print("  Computing [[5,1,3]] logical states …")
ket_0L, ket_1L = compute_logical_states()
print("  ✓")

def logical_state(theta, phi):
    return np.cos(theta/2)*ket_0L + np.exp(1j*phi)*np.sin(theta/2)*ket_1L

def state_to_dm(psi):
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

# ──────────────────────────────────────────────────────────────
# 4. Noise & standard QEC
# ──────────────────────────────────────────────────────────────
def depolarizing_channel(rho, p):
    result = rho.copy()
    for q in range(N_CODE):
        nr = (1 - p) * result
        for P in [X_m, Y_m, Z_m]:
            ops = [I2]*N_CODE; ops[q] = P
            E   = kron_list(ops)
            nr += (p/3) * E @ result @ E.conj().T
        result = nr
    return result

def standard_qec_5q(rho_noisy):
    dim  = 2**N_CODE
    corr = {(1, 1, 1, 1): np.eye(dim, dtype=complex)}
    for q in range(N_CODE):
        for _, P in {'X': X_m, 'Y': Y_m, 'Z': Z_m}.items():
            ops = [I2]*N_CODE; ops[q] = P
            E   = kron_list(ops)
            syn = tuple(-1 if np.allclose(S @ E + E @ S, 0) else 1
                        for S in stabilizers_mat)
            corr[syn] = E
    rho_out = np.zeros_like(rho_noisy)
    for syn, c in corr.items():
        proj = np.eye(dim, dtype=complex)
        for j, s in enumerate(syn):
            proj = proj @ ((np.eye(dim) + s*stabilizers_mat[j]) / 2)
        rho_out += c @ proj @ rho_noisy @ proj @ c.conj().T
    return rho_out

# ──────────────────────────────────────────────────────────────
# 5. Dataset  (built once, shared across all layer counts)
# ──────────────────────────────────────────────────────────────
def fibonacci_sphere(n):
    golden = (1 + np.sqrt(5)) / 2
    return [(np.arccos(1 - 2*(i+0.5)/n),
             (2*np.pi*i/golden) % (2*np.pi)) for i in range(n)]

def generate_dataset(n_states, p):
    data = []
    for theta, phi in fibonacci_sphere(n_states):
        psi   = logical_state(theta, phi)
        rho_c = state_to_dm(psi)
        rho_n = depolarizing_channel(rho_c, p)
        data.append(dict(
            theta=theta, phi=phi,
            rho_clean=rho_c, rho_noisy=rho_n,
            zl=float(np.real(np.trace(ZL_mat @ rho_c))),
            xl=float(np.real(np.trace(XL_mat @ rho_c))),
            yl=float(np.real(np.trace(YL_mat @ rho_c))),
        ))
    return data

print(f"  Generating training data ({N_TRAIN_STATES} states) …")
training_data = generate_dataset(N_TRAIN_STATES, P_NOISE)
print(f"  Generating test data ({N_TEST_STATES} states) …")
test_data     = generate_dataset(N_TEST_STATES,  P_NOISE)

# Stacked JAX arrays
rho_noisy_train_stack = jnp.array(
    np.stack([d['rho_noisy'] for d in training_data]), dtype=complex)
rho_clean_train_stack = jnp.array(
    np.stack([d['rho_clean'] for d in training_data]), dtype=complex)
targets_train_stack   = jnp.array(
    [[d['zl'], d['xl'], d['yl']] for d in training_data])

rho_noisy_test_stack = jnp.array(
    np.stack([d['rho_noisy'] for d in test_data]), dtype=complex)
rho_clean_test_stack = jnp.array(
    np.stack([d['rho_clean'] for d in test_data]), dtype=complex)
zl_test_targets = jnp.array([d['zl'] for d in test_data])
xl_test_targets = jnp.array([d['xl'] for d in test_data])
yl_test_targets = jnp.array([d['yl'] for d in test_data])

print(f"  Pre-computing standard QEC on {N_TEST_STATES} test states …")
rho_std_test = np.stack([standard_qec_5q(d['rho_noisy']) for d in test_data])
f_std_test   = np.array([
    float(np.real(np.trace(rho_std_test[i] @ test_data[i]['rho_clean'])))
    for i in range(N_TEST_STATES)
])
f_noisy_test_global = np.array([
    float(np.real(np.trace(
        test_data[i]['rho_noisy'] @ test_data[i]['rho_clean'])))
    for i in range(N_TEST_STATES)
])
print(f"  ✓  Std QEC mean F = {f_std_test.mean():.5f}")

# ──────────────────────────────────────────────────────────────
# 6. Partial-trace helpers
# ──────────────────────────────────────────────────────────────
def partial_trace_system(rho_64):
    r = rho_64.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=0, axis2=2)

def partial_trace_ancilla(rho_64):
    r = rho_64.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=1, axis2=3)

# ──────────────────────────────────────────────────────────────
# 7. Ansatz  —  RY+RZ single-qubit layers (NO RX) / CRX+CRY+CRZ
# ──────────────────────────────────────────────────────────────
def count_params(n_layers):
    rot = (n_layers + 1) * N_TOTAL * 2            # single-qubit RY+RZ layers
    cr  = n_layers * N_TOTAL * (N_TOTAL - 1) * 3  # CRX + CRY + CRZ
    return rot + cr

def make_ansatz(n_layers):
    def ansatz(params):
        idx = 0
        for q in range(N_TOTAL):
            qml.RY(params[idx], wires=q); idx += 1
            qml.RZ(params[idx], wires=q); idx += 1
        for _ in range(n_layers):
            for ctrl in range(N_TOTAL):
                for tgt in range(N_TOTAL):
                    if ctrl != tgt:
                        qml.CRX(params[idx], wires=[ctrl, tgt]); idx += 1
                        qml.CRY(params[idx], wires=[ctrl, tgt]); idx += 1
                        qml.CRZ(params[idx], wires=[ctrl, tgt]); idx += 1
            for q in range(N_TOTAL):
                qml.RY(params[idx], wires=q); idx += 1
                qml.RZ(params[idx], wires=q); idx += 1
    return ansatz

# ──────────────────────────────────────────────────────────────
# 8. Per-layer factory
# ──────────────────────────────────────────────────────────────
def make_layer_components(n_layers):
    n_p        = count_params(n_layers)
    ansatz     = make_ansatz(n_layers)
    encoder_fn = qml.matrix(ansatz, wire_order=range(N_TOTAL))

    # ── single-state forward  (used for gradient) ─────────────
    def qae_single(params, rho_5q):
        V  = encoder_fn(params)
        Vd = jnp.conj(V.T)
        rho  = jnp.kron(rho_5q, ANC_0)
        rho  = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho  = jnp.kron(FRESH_00000, rho_a)
        rho  = Vd @ rho @ V
        return partial_trace_ancilla(rho)

    # ── batched forward: V built ONCE, vmapped linear algebra ─
    def _fwd_with_V(V, Vd, rho_5q):
        rho  = jnp.kron(rho_5q, ANC_0)
        rho  = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho  = jnp.kron(FRESH_00000, rho_a)
        rho  = Vd @ rho @ V
        return partial_trace_ancilla(rho)

    _vmapped = jax.vmap(_fwd_with_V, in_axes=(None, None, 0))

    @jax.jit
    def qae_batch_chunk(params, rho_chunk):
        V  = encoder_fn(params)
        Vd = jnp.conj(V.T)
        return _vmapped(V, Vd, rho_chunk)

    # ── per-state loss + grad (ADAPTIVE weights, used for TRAINING) ──
    @jax.jit
    def loss_and_grad(params, rho_noisy, target, ws, wz, wx, wy):
        def _loss(p):
            rho = qae_single(p, rho_noisy)
            s1  = jnp.real(jnp.trace(S1j @ rho))
            s2  = jnp.real(jnp.trace(S2j @ rho))
            s3  = jnp.real(jnp.trace(S3j @ rho))
            s4  = jnp.real(jnp.trace(S4j @ rho))
            zl  = jnp.real(jnp.trace(ZLj @ rho))
            xl  = jnp.real(jnp.trace(XLj @ rho))
            yl  = jnp.real(jnp.trace(YLj @ rho))
            return (ws*(-(s1+s2+s3+s4))
                    + wz*(target[0]-zl)**2
                    + wx*(target[1]-xl)**2
                    + wy*(target[2]-yl)**2)
        return jax.value_and_grad(_loss)(params)

    # ── batched stabiliser average ────────────────────────────
    @jax.jit
    def batch_stab_avg(params, rho_chunk):
        out = qae_batch_chunk(params, rho_chunk)
        def per(r):
            return (jnp.real(jnp.trace(S1j @ r)) +
                    jnp.real(jnp.trace(S2j @ r)) +
                    jnp.real(jnp.trace(S3j @ r)) +
                    jnp.real(jnp.trace(S4j @ r))) / 4.0
        return jnp.mean(jax.vmap(per)(out))

    # ── batched fidelity (mean) ───────────────────────────────
    @jax.jit
    def batch_fidelity_chunk(params, rho_noisy_chunk, rho_clean_chunk):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(r_out, r_c):
            return jnp.real(jnp.trace(r_out @ r_c))
        return jnp.mean(jax.vmap(per)(out, rho_clean_chunk))

    # ── batched logical spectrum errors  ΔZL / ΔXL / ΔYL (mean) ─
    @jax.jit
    def batch_logical_errors(params, rho_noisy_chunk,
                              zl_targets, xl_targets, yl_targets):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(rho, zl_t, xl_t, yl_t):
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return jnp.abs(zl_t - zl), jnp.abs(xl_t - xl), jnp.abs(yl_t - yl)
        dzl, dxl, dyl = jax.vmap(per)(out, zl_targets, xl_targets, yl_targets)
        return jnp.mean(dzl), jnp.mean(dxl), jnp.mean(dyl)

    # ── batched FIXED-weight baseline spectral loss (mean, tracking only) ──
    @jax.jit
    def batch_baseline_loss(params, rho_noisy_chunk,
                            zl_t, xl_t, yl_t):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(rho, zl_tt, xl_tt, yl_tt):
            s1 = jnp.real(jnp.trace(S1j @ rho))
            s2 = jnp.real(jnp.trace(S2j @ rho))
            s3 = jnp.real(jnp.trace(S3j @ rho))
            s4 = jnp.real(jnp.trace(S4j @ rho))
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return (W_STAB_BASE*(-(s1+s2+s3+s4))
                    + W_ZL_BASE*(zl_tt-zl)**2
                    + W_XL_BASE*(xl_tt-xl)**2
                    + W_YL_BASE*(yl_tt-yl)**2)
        return jnp.mean(jax.vmap(per)(out, zl_t, xl_t, yl_t))

    # ── per-state ARRAY versions (for mean + std logging) ─────
    @jax.jit
    def batch_fidelity_arr(params, rho_noisy_chunk, rho_clean_chunk):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(r_out, r_c):
            return jnp.real(jnp.trace(r_out @ r_c))
        return jax.vmap(per)(out, rho_clean_chunk)

    @jax.jit
    def batch_logical_errors_arr(params, rho_noisy_chunk,
                                 zl_t, xl_t, yl_t):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(rho, zt, xt, yt):
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return jnp.abs(zt-zl), jnp.abs(xt-xl), jnp.abs(yt-yl)
        return jax.vmap(per)(out, zl_t, xl_t, yl_t)

    @jax.jit
    def batch_baseline_loss_arr(params, rho_noisy_chunk,
                                zl_t, xl_t, yl_t):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(rho, zl_tt, xl_tt, yl_tt):
            s1 = jnp.real(jnp.trace(S1j @ rho))
            s2 = jnp.real(jnp.trace(S2j @ rho))
            s3 = jnp.real(jnp.trace(S3j @ rho))
            s4 = jnp.real(jnp.trace(S4j @ rho))
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return (W_STAB_BASE*(-(s1+s2+s3+s4))
                    + W_ZL_BASE*(zl_tt-zl)**2
                    + W_XL_BASE*(xl_tt-xl)**2
                    + W_YL_BASE*(yl_tt-yl)**2)
        return jax.vmap(per)(out, zl_t, xl_t, yl_t)

    return dict(
        n_params                 = n_p,
        loss_and_grad            = loss_and_grad,
        qae_batch_chunk          = qae_batch_chunk,
        batch_stab_avg           = batch_stab_avg,
        batch_fidelity_chunk     = batch_fidelity_chunk,
        batch_logical_errors     = batch_logical_errors,
        batch_baseline_loss      = batch_baseline_loss,
        batch_fidelity_arr       = batch_fidelity_arr,
        batch_logical_errors_arr = batch_logical_errors_arr,
        batch_baseline_loss_arr  = batch_baseline_loss_arr,
    )

# ──────────────────────────────────────────────────────────────
# 9. Chunked full-dataset evaluation helpers
# ──────────────────────────────────────────────────────────────
def eval_fidelity_chunked(params, qae_batch_chunk_fn,
                           rho_noisy_stack, rho_clean_stack,
                           chunk_size=CHUNK_EVAL_SIZE):
    n    = rho_noisy_stack.shape[0]
    fids = np.empty(n, dtype=np.float64)
    for k in range(int(np.ceil(n / chunk_size))):
        sl   = slice(k*chunk_size, min((k+1)*chunk_size, n))
        rout = np.array(qae_batch_chunk_fn(params, rho_noisy_stack[sl]))
        rcln = np.array(rho_clean_stack[sl])
        for i, (ro, rc) in enumerate(zip(rout, rcln)):
            fids[sl.start + i] = float(np.real(np.trace(ro @ rc)))
    return fids

def eval_logical_errors_chunked_v2(params, qae_batch_chunk_fn,
                                    rho_noisy_stack,
                                    zl_tgt, xl_tgt, yl_tgt,
                                    chunk_size=CHUNK_EVAL_SIZE):
    n    = rho_noisy_stack.shape[0]
    dzls = np.empty(n, dtype=np.float64)
    dxls = np.empty(n, dtype=np.float64)
    dyls = np.empty(n, dtype=np.float64)
    for k in range(int(np.ceil(n / chunk_size))):
        sl   = slice(k*chunk_size, min((k+1)*chunk_size, n))
        rout = np.array(qae_batch_chunk_fn(params, rho_noisy_stack[sl]))
        zt   = np.array(zl_tgt[sl])
        xt   = np.array(xl_tgt[sl])
        yt   = np.array(yl_tgt[sl])
        for i, ro in enumerate(rout):
            ro_j = jnp.array(ro)
            dzls[sl.start+i] = abs(zt[i] - float(jnp.real(jnp.trace(ZLj @ ro_j))))
            dxls[sl.start+i] = abs(xt[i] - float(jnp.real(jnp.trace(XLj @ ro_j))))
            dyls[sl.start+i] = abs(yt[i] - float(jnp.real(jnp.trace(YLj @ ro_j))))
    return dzls, dxls, dyls

# ──────────────────────────────────────────────────────────────
# 10. Adaptive weight helpers
# ──────────────────────────────────────────────────────────────
def compute_target_weights(avg_stab):
    sat = float(np.clip(
        (avg_stab - STAB_SAT_LOW) / (STAB_SAT_HIGH - STAB_SAT_LOW), 0, 1))
    ws  = W_STAB_HIGH - (W_STAB_HIGH - W_STAB_LOW) * sat
    wz  = W_ZL_LOW   + (W_ZL_HIGH   - W_ZL_LOW)   * sat
    wx  = W_XL_LOW   + (W_XL_HIGH   - W_XL_LOW)   * sat
    wy  = W_YL_LOW   + (W_YL_HIGH   - W_YL_LOW)   * sat
    return float(ws), float(wz), float(wx), float(wy), float(sat)

def step_weight(cur, tgt):
    return cur + float(np.clip(tgt - cur, -MAX_W_DELTA, MAX_W_DELTA))

# ──────────────────────────────────────────────────────────────
# 10b. Layerwise growth helper
#      params(L) is a PREFIX of params(L+1): new blocks are appended
#      at the tail. Zero angles => CR gates and RY/RZ are identity,
#      so the grown circuit starts numerically identical to the smaller
#      trained solution.
# ──────────────────────────────────────────────────────────────
def grow_params(params_prev, prev_nl, nl):
    extra = count_params(nl) - count_params(prev_nl)
    return np.concatenate([np.asarray(params_prev), np.zeros(extra)])

# ──────────────────────────────────────────────────────────────
# 11. One training epoch
# ──────────────────────────────────────────────────────────────
def run_epoch(params, opt_state, optimizer,
              ws, wz, wx, wy, rng_key, loss_and_grad_fn):
    jws = jnp.float64(ws); jwz = jnp.float64(wz)
    jwx = jnp.float64(wx); jwy = jnp.float64(wy)
    steps   = int(np.ceil(N_TRAIN_STATES / BATCH_SIZE))
    rng_key, sk = jax.random.split(rng_key)
    idx_all = np.array(jax.random.permutation(sk, N_TRAIN_STATES))
    ep_loss = 0.0; nb = 0
    for step in range(steps):
        b     = idx_all[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        g_acc = jnp.zeros_like(params); l_acc = 0.0
        for i in b:
            lv, gv = loss_and_grad_fn(
                params,
                rho_noisy_train_stack[i],
                targets_train_stack[i],
                jws, jwz, jwx, jwy)
            gv     = jnp.where(jnp.isnan(gv), 0.0, gv)
            lv     = jnp.where(jnp.isnan(lv), 0.0, lv)
            g_acc += gv; l_acc += float(lv)
        updates, opt_state = optimizer.update(g_acc / len(b), opt_state, params)
        params   = optax.apply_updates(params, updates)
        ep_loss += l_acc / len(b); nb += 1
    return params, opt_state, ep_loss / max(nb, 1), rng_key

# ──────────────────────────────────────────────────────────────
# 12. Full training run for one layer count
#     FULL history: Phase 1 (best warm-up trajectory) + Phase 2,
#     recorded EVERY epoch on a continuous global epoch index.
#     Monitoring uses ALL 100 training states; MEAN + STD logged.
#
#     If init_params is provided (layerwise warm-start), Phase 1 is a
#     single refinement from those params (NO multi-start). Otherwise
#     the original multi-start warm-up is used (this is what L=1 does,
#     since it has no previous solution to grow from).
# ──────────────────────────────────────────────────────────────
def train_one_config(n_layers, rng_seed=42, init_params=None):
    n_p = count_params(n_layers)
    print(f"\n{'='*68}")
    print(f"  N_LAYERS = {n_layers}   params = {n_p}   "
          f"chunk = {CHUNK_EVAL_SIZE}  "
          f"(peak eval RAM ≈ "
          f"{CHUNK_EVAL_SIZE*64*64*16*6/1024/1024:.0f} MB)")
    print(f"  Ansatz: RY+RZ / CRX+CRY+CRZ  (no RX in single-qubit layers)")
    print(f"  Monitoring on ALL {N_TRAIN_STATES} training states (mean+std)")
    if init_params is not None:
        print(f"  Init: LAYERWISE WARM-START (identity-grown from previous L)")
    else:
        print(f"  Init: MULTI-START warm-up ({N_INIT_ROUNDS} rounds)")
    print(f"{'='*68}")

    comps = make_layer_components(n_layers)
    loss_and_grad            = comps['loss_and_grad']
    qae_batch_chunk          = comps['qae_batch_chunk']
    batch_stab_avg           = comps['batch_stab_avg']
    batch_fidelity_chunk     = comps['batch_fidelity_chunk']
    batch_logical_errors     = comps['batch_logical_errors']
    batch_baseline_loss      = comps['batch_baseline_loss']
    batch_fidelity_arr       = comps['batch_fidelity_arr']
    batch_logical_errors_arr = comps['batch_logical_errors_arr']
    batch_baseline_loss_arr  = comps['batch_baseline_loss_arr']

    # ── FULL training set used for per-epoch monitoring ───────
    probe_rho_noisy = rho_noisy_train_stack           # all 100
    probe_rho_clean = rho_clean_train_stack
    probe_zl        = targets_train_stack[:, 0]
    probe_xl        = targets_train_stack[:, 1]
    probe_yl        = targets_train_stack[:, 2]

    # Convenience: all monitoring metrics at once → (mean, std) pairs.
    def probe_metrics(p):
        bl = np.asarray(batch_baseline_loss_arr(
                p, probe_rho_noisy, probe_zl, probe_xl, probe_yl))
        fd = np.asarray(batch_fidelity_arr(
                p, probe_rho_noisy, probe_rho_clean))
        dz, dx, dy = batch_logical_errors_arr(
                p, probe_rho_noisy, probe_zl, probe_xl, probe_yl)
        dz = np.asarray(dz); dx = np.asarray(dx); dy = np.asarray(dy)
        return (float(bl.mean()), float(bl.std()),
                float(fd.mean()), float(fd.std()),
                float(dz.mean()), float(dz.std()),
                float(dx.mean()), float(dx.std()),
                float(dy.mean()), float(dy.std()))

    # JIT warm-up
    print("  JIT warm-up …", end=" ", flush=True)
    t0  = time.time()
    _p0 = jnp.zeros(n_p)
    loss_and_grad(_p0, rho_noisy_train_stack[0], targets_train_stack[0],
                  jnp.float64(W_STAB_HIGH), jnp.float64(W_ZL_LOW),
                  jnp.float64(W_XL_LOW),    jnp.float64(W_YL_LOW))
    batch_stab_avg(_p0, probe_rho_noisy)
    batch_fidelity_arr(_p0, probe_rho_noisy, probe_rho_clean)
    batch_logical_errors_arr(_p0, probe_rho_noisy, probe_zl, probe_xl, probe_yl)
    batch_baseline_loss_arr(_p0, probe_rho_noisy, probe_zl, probe_xl, probe_yl)
    print(f"done in {time.time()-t0:.1f}s")

    # One cosine LR schedule for the whole run (Phase 1 + Phase 2)
    steps_per_ep = int(np.ceil(N_TRAIN_STATES / BATCH_SIZE))
    optimizer    = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(LR, TOTAL_EPOCHS*steps_per_ep)))

    rng = jax.random.PRNGKey(rng_seed)

    # ── Phase 1 ───────────────────────────────────────────────
    best_p = best_opt = None
    best_f = -1.0
    best_ws = W_STAB_HIGH; best_wz = W_ZL_LOW
    best_wx = W_XL_LOW;    best_wy = W_YL_LOW
    best_rng = rng
    best_round_hist = None   # per-epoch history of the winning trajectory

    if init_params is not None:
        # ── Phase 1 (LAYERWISE WARM-START): single refinement ──
        print(f"\n  Phase 1 : LAYERWISE WARM-START — single {INIT_EPOCHS}-epoch "
              f"refinement from identity-grown params (no multi-start)")
        tp = jnp.asarray(init_params, dtype=jnp.float64)
        to = optimizer.init(tp)
        ws = W_STAB_HIGH; wz = W_ZL_LOW; wx = W_XL_LOW; wy = W_YL_LOW
        rng, rnd_rng = jax.random.split(rng)
        round_hist = []
        for ep in range(INIT_EPOCHS):
            if ep % WEIGHT_UPDATE_FREQ == 0:
                avg_s = float(batch_stab_avg(tp, probe_rho_noisy))
                tw, twz, twx, twy, _ = compute_target_weights(avg_s)
                ws = step_weight(ws, tw); wz = step_weight(wz, twz)
                wx = step_weight(wx, twx); wy = step_weight(wy, twy)
            tp, to, _, rnd_rng = run_epoch(
                tp, to, optimizer, ws, wz, wx, wy, rnd_rng, loss_and_grad)
            m = probe_metrics(tp)
            round_hist.append((ep, 1, *m))
        best_f = round_hist[-1][4]
        best_p = tp; best_opt = to
        best_ws = ws; best_wz = wz; best_wx = wx; best_wy = wy
        best_rng = rnd_rng
        best_round_hist = round_hist
        print(f"  Warm-start refinement F = {best_f:.4f}  "
              f"(appended block = identity → non-destructive growth)")
    else:
        # ── Phase 1: multi-start warm-up (ORIGINAL) ──
        print(f"\n  Phase 1 : {N_INIT_ROUNDS} rounds × {INIT_EPOCHS} epochs "
              f"on all {N_TRAIN_STATES} states  (per-epoch metrics recorded)")
        print(f"  {'Rd':<3} {'Init style':<20} {'F_end':>7} "
              f"{'⟨Sj⟩':>6} {'ΔZL':>6} {'ΔXL':>6} {'ΔYL':>6}")
        print(f"  {'─'*3} {'─'*20} {'─'*7} "
              f"{'─'*6} {'─'*6} {'─'*6} {'─'*6}")

        for rd in range(N_INIT_ROUNDS):
            rng, sk = jax.random.split(rng)
            s = rd % 4
            if   s == 0:
                tp = jax.random.uniform(sk, shape=(n_p,), minval=-np.pi, maxval=np.pi)
                sn = "Uniform[-π,π]"
            elif s == 1:
                tp = jax.random.normal(sk, shape=(n_p,)) * 0.15
                sn = "Normal(0,0.15)"
            elif s == 2:
                tp = jax.random.normal(sk, shape=(n_p,)) * 1.0
                sn = "Normal(0,1.0)"
            else:
                tp = jax.random.uniform(sk, shape=(n_p,), minval=-0.1, maxval=0.1)
                sn = "Unif[-0.1,0.1]"

            to = optimizer.init(tp)
            ws = W_STAB_HIGH; wz = W_ZL_LOW; wx = W_XL_LOW; wy = W_YL_LOW
            rng, rnd_rng = jax.random.split(rng)

            round_hist = []
            for ep in range(INIT_EPOCHS):
                if ep % WEIGHT_UPDATE_FREQ == 0:
                    avg_s = float(batch_stab_avg(tp, probe_rho_noisy))
                    tw, twz, twx, twy, _ = compute_target_weights(avg_s)
                    ws = step_weight(ws, tw); wz = step_weight(wz, twz)
                    wx = step_weight(wx, twx); wy = step_weight(wy, twy)
                tp, to, _, rnd_rng = run_epoch(
                    tp, to, optimizer, ws, wz, wx, wy, rnd_rng, loss_and_grad)
                m = probe_metrics(tp)
                round_hist.append((ep, 1, *m))

            avg_f = round_hist[-1][4]
            avg_s = float(batch_stab_avg(tp, probe_rho_noisy))
            dzl   = round_hist[-1][6]
            dxl   = round_hist[-1][8]
            dyl   = round_hist[-1][10]

            star = ""
            if avg_f > best_f:
                best_f   = avg_f; best_p = tp; best_opt = to
                best_ws  = ws;    best_wz = wz
                best_wx  = wx;    best_wy = wy
                best_rng = rnd_rng
                best_round_hist = round_hist
                star = " ★"

            print(f"  {rd:<3} {sn:<20} {avg_f:>7.4f} "
                  f"{avg_s:>6.3f} {dzl:>6.4f} "
                  f"{dxl:>6.4f} {dyl:>6.4f}{star}")
            if star == "":
                del to
            gc.collect()

        print(f"  Best warm-up F = {best_f:.4f}  "
              f"(Phase 1 trajectory of winning round retained)")

    # ── Unified history seeded with the WINNING Phase-1 trajectory ──
    hist_epoch    = [h[0]  for h in best_round_hist]
    hist_phase    = [h[1]  for h in best_round_hist]
    hist_loss     = [h[2]  for h in best_round_hist]
    hist_loss_std = [h[3]  for h in best_round_hist]
    hist_fid      = [h[4]  for h in best_round_hist]
    hist_fid_std  = [h[5]  for h in best_round_hist]
    hist_dzl      = [h[6]  for h in best_round_hist]
    hist_dzl_std  = [h[7]  for h in best_round_hist]
    hist_dxl      = [h[8]  for h in best_round_hist]
    hist_dxl_std  = [h[9]  for h in best_round_hist]
    hist_dyl      = [h[10] for h in best_round_hist]
    hist_dyl_std  = [h[11] for h in best_round_hist]

    # ── Phase 2: continuation ─────────────────────────────────
    print(f"\n  Phase 2 : {MAIN_EPOCHS} continuation epochs "
          f"(global epochs {INIT_EPOCHS} … {TOTAL_EPOCHS-1})")
    params    = best_p
    opt_state = best_opt
    ws, wz, wx, wy = best_ws, best_wz, best_wx, best_wy
    rng_key   = best_rng

    (hb_m, hb_s, hf_m, hf_s,
     hz_m, hz_s, hx_m, hx_s, hy_m, hy_s) = probe_metrics(params)
    print(f"  Handoff  F={hf_m:.4f}±{hf_s:.4f}  "
          f"BaseLoss={hb_m:.4f}±{hb_s:.4f}  "
          f"ΔZL={hz_m:.4f}±{hz_s:.4f}  ΔXL={hx_m:.4f}  ΔYL={hy_m:.4f}\n")

    print(f"  {'Ep':<5} {'BaseLoss':>9} {'F_tr':>7} {'⟨Sj⟩':>6} "
          f"{'ΔZL':>7} {'ΔXL':>7} {'ΔYL':>7} {'t':>5}")
    print(f"  {'─'*5} {'─'*9} {'─'*7} {'─'*6} "
          f"{'─'*7} {'─'*7} {'─'*7} {'─'*5}")

    for ep in range(MAIN_EPOCHS):
        t0 = time.time()

        if ep % WEIGHT_UPDATE_FREQ == 0:
            avg_s = float(batch_stab_avg(params, probe_rho_noisy))
            tw, twz, twx, twy, _ = compute_target_weights(avg_s)
            ws = step_weight(ws, tw); wz = step_weight(wz, twz)
            wx = step_weight(wx, twx); wy = step_weight(wy, twy)

        params, opt_state, _train_loss, rng_key = run_epoch(
            params, opt_state, optimizer,
            ws, wz, wx, wy, rng_key, loss_and_grad)

        # Record EVERY epoch (mean + std over all 100 training states)
        (base_m, base_s, fid_m, fid_s,
         dzl_m, dzl_s, dxl_m, dxl_s, dyl_m, dyl_s) = probe_metrics(params)
        avg_s = float(batch_stab_avg(params, probe_rho_noisy))

        total_ep = INIT_EPOCHS + ep
        hist_epoch.append(total_ep);   hist_phase.append(2)
        hist_loss.append(base_m);      hist_loss_std.append(base_s)
        hist_fid.append(fid_m);        hist_fid_std.append(fid_s)
        hist_dzl.append(dzl_m);        hist_dzl_std.append(dzl_s)
        hist_dxl.append(dxl_m);        hist_dxl_std.append(dxl_s)
        hist_dyl.append(dyl_m);        hist_dyl_std.append(dyl_s)

        # Console print stays at the readable cadence.
        if ep % 10 == 0 or ep == MAIN_EPOCHS - 1:
            print(f"  {total_ep:<5} {base_m:>9.3f} {fid_m:>7.4f} "
                  f"{avg_s:>6.3f} "
                  f"{dzl_m:>7.4f} {dxl_m:>7.4f} {dyl_m:>7.4f} "
                  f"{time.time()-t0:>4.1f}s")

        if ep % 20 == 0:
            gc.collect()

    # Convert unified history to arrays
    hist_epoch    = np.array(hist_epoch, dtype=np.int64)
    hist_phase    = np.array(hist_phase, dtype=np.int64)
    hist_loss     = np.array(hist_loss,     dtype=np.float64)
    hist_loss_std = np.array(hist_loss_std, dtype=np.float64)
    hist_fid      = np.array(hist_fid,      dtype=np.float64)
    hist_fid_std  = np.array(hist_fid_std,  dtype=np.float64)
    hist_dzl      = np.array(hist_dzl,      dtype=np.float64)
    hist_dzl_std  = np.array(hist_dzl_std,  dtype=np.float64)
    hist_dxl      = np.array(hist_dxl,      dtype=np.float64)
    hist_dxl_std  = np.array(hist_dxl_std,  dtype=np.float64)
    hist_dyl      = np.array(hist_dyl,      dtype=np.float64)
    hist_dyl_std  = np.array(hist_dyl_std,  dtype=np.float64)

    # ── Chunked test evaluation ────────────────────────────────
    print(f"\n  Test eval: {N_TEST_STATES} states in chunks of "
          f"{CHUNK_EVAL_SIZE} …", end=" ", flush=True)
    t0    = time.time()
    f_qae = eval_fidelity_chunked(
        params, qae_batch_chunk,
        rho_noisy_test_stack, rho_clean_test_stack,
        chunk_size=CHUNK_EVAL_SIZE)
    print(f"done in {time.time()-t0:.1f}s  F̄={f_qae.mean():.5f}")

    print(f"  Logical error eval …", end=" ", flush=True)
    t0 = time.time()
    dzl_test, dxl_test, dyl_test = eval_logical_errors_chunked_v2(
        params, qae_batch_chunk,
        rho_noisy_test_stack,
        zl_test_targets, xl_test_targets, yl_test_targets,
        chunk_size=CHUNK_EVAL_SIZE)
    print(f"done in {time.time()-t0:.1f}s  "
          f"ΔZL={dzl_test.mean():.4f}  "
          f"ΔXL={dxl_test.mean():.4f}  "
          f"ΔYL={dyl_test.mean():.4f}")

    return dict(
        n_layers          = n_layers,
        n_params          = n_p,
        params            = np.array(params),
        # ── FULL continuous history (Phase 1 + Phase 2, every epoch) ──
        hist_epoch        = hist_epoch,
        hist_phase        = hist_phase,
        hist_loss         = hist_loss,       # FIXED-weight baseline loss (mean)
        hist_loss_std     = hist_loss_std,
        hist_fid          = hist_fid,
        hist_fid_std      = hist_fid_std,
        hist_dzl          = hist_dzl,
        hist_dzl_std      = hist_dzl_std,
        hist_dxl          = hist_dxl,
        hist_dxl_std      = hist_dxl_std,
        hist_dyl          = hist_dyl,
        hist_dyl_std      = hist_dyl_std,
        # ── test-set results ──
        f_qae_test        = f_qae,
        f_std_test        = f_std_test,
        f_noisy_test      = f_noisy_test_global,
        dzl_test          = dzl_test,
        dxl_test          = dxl_test,
        dyl_test          = dyl_test,
    )

# ──────────────────────────────────────────────────────────────
# 13. Run sweep  (with layerwise warm-start for L>1)
# ──────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("  LAYER SWEEP")
print("="*68)
print(f"  Layers          : {LAYER_SWEEP}")
print(f"  Ansatz          : RY+RZ / CRX+CRY+CRZ  (no RX)")
print(f"  CHUNK_EVAL_SIZE : {CHUNK_EVAL_SIZE}")
print(f"  Weight update   : every {WEIGHT_UPDATE_FREQ} epochs")
print(f"  Monitoring      : ALL {N_TRAIN_STATES} training states, mean+std")
print(f"  Loss tracking   : FIXED baseline weights "
      f"(S={W_STAB_BASE}, ZL={W_ZL_BASE}, XL={W_XL_BASE}, YL={W_YL_BASE})")
print(f"  History         : FULL — Phase 1 + Phase 2, every epoch")
print(f"  Layerwise init  : {USE_LAYERWISE}  (L=1 always uses multi-start)")
print()

sweep_results = {}
t_sweep = time.time()

prev_nl = None; prev_params = None
for nl in LAYER_SWEEP:
    if (not USE_LAYERWISE) or (prev_params is None) or (nl <= prev_nl):
        # First layer (or growth disabled / non-increasing) -> multi-start.
        sweep_results[nl] = train_one_config(nl, rng_seed=42 + nl)
    else:
        # Grow from previous trained solution (identity-appended block).
        warm = grow_params(prev_params, prev_nl, nl)
        sweep_results[nl] = train_one_config(nl, rng_seed=42 + nl,
                                             init_params=warm)
    prev_nl     = nl
    prev_params = sweep_results[nl]['params']

print(f"\n  Total sweep time : {time.time()-t_sweep:.0f}s "
      f"({(time.time()-t_sweep)/60:.1f} min)")

# ──────────────────────────────────────────────────────────────
# 14. Save outputs
#     Aligned, equal-length per-layer arrays covering ALL epochs.
#     Now ALSO saves the trained parameters (per layer + overall best).
# ──────────────────────────────────────────────────────────────
save_dict = {}
for nl, res in sweep_results.items():
    # Trained parameters for this layer count
    save_dict[f'L{nl}_params']        = res['params']
    # Continuous per-epoch history (Phase 1 + Phase 2)
    save_dict[f'L{nl}_epoch']         = res['hist_epoch']  # global epoch index
    save_dict[f'L{nl}_phase']         = res['hist_phase']  # 1 or 2
    save_dict[f'L{nl}_loss_hist']     = res['hist_loss']   # baseline loss (mean)
    save_dict[f'L{nl}_loss_hist_std'] = res['hist_loss_std']
    save_dict[f'L{nl}_fid_hist']      = res['hist_fid']
    save_dict[f'L{nl}_fid_hist_std']  = res['hist_fid_std']
    save_dict[f'L{nl}_dzl_hist']      = res['hist_dzl']
    save_dict[f'L{nl}_dzl_hist_std']  = res['hist_dzl_std']
    save_dict[f'L{nl}_dxl_hist']      = res['hist_dxl']
    save_dict[f'L{nl}_dxl_hist_std']  = res['hist_dxl_std']
    save_dict[f'L{nl}_dyl_hist']      = res['hist_dyl']
    save_dict[f'L{nl}_dyl_hist_std']  = res['hist_dyl_std']
    # Test-set results
    save_dict[f'L{nl}_f_qae_test']    = res['f_qae_test']
    save_dict[f'L{nl}_f_std_test']    = res['f_std_test']
    save_dict[f'L{nl}_f_noisy_test']  = res['f_noisy_test']
    save_dict[f'L{nl}_dzl_test']      = res['dzl_test']
    save_dict[f'L{nl}_dxl_test']      = res['dxl_test']
    save_dict[f'L{nl}_dyl_test']      = res['dyl_test']

# Overall best model (by mean test fidelity)
best_nl_save = max(LAYER_SWEEP,
                   key=lambda nl: sweep_results[nl]['f_qae_test'].mean())
save_dict['best_n_layers'] = np.array(best_nl_save)
save_dict['best_params']   = sweep_results[best_nl_save]['params']
save_dict['best_f_qae']    = np.array(
    sweep_results[best_nl_save]['f_qae_test'].mean())

# Metadata so external readers understand the schema
save_dict['meta_INIT_EPOCHS']  = np.array(INIT_EPOCHS)
save_dict['meta_MAIN_EPOCHS']  = np.array(MAIN_EPOCHS)
save_dict['meta_TOTAL_EPOCHS'] = np.array(TOTAL_EPOCHS)
save_dict['meta_layers']       = np.array(LAYER_SWEEP)
save_dict['meta_baseline_w']   = np.array(
    [W_STAB_BASE, W_ZL_BASE, W_XL_BASE, W_YL_BASE])
save_dict['meta_probe_n']      = np.array(PROBE_N)  # now = full training set
save_dict['meta_layerwise']    = np.array(int(USE_LAYERWISE))

np.savez(f"perfect_results_{FILE_TAG}.npz", **save_dict)
print(f"  Saved : perfect_results_{FILE_TAG}.npz")
print(f"    Per-layer keys: L{best_nl_save}_params, L{best_nl_save}_epoch, L{best_nl_save}_phase, "
      f"L{best_nl_save}_loss_hist(_std), L{best_nl_save}_fid_hist(_std), "
      f"L{best_nl_save}_dzl_hist(_std), L{best_nl_save}_dxl_hist(_std), L{best_nl_save}_dyl_hist(_std)")
print(f"    Best model    : best_n_layers=L{best_nl_save}, best_params, best_f_qae")

# ── Long-format per-epoch CSV (fully reproducible training curves) ──
with open(f"perfect_history_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "global_epoch", "phase",
                "baseline_loss", "baseline_loss_std",
                "fidelity", "fidelity_std",
                "dzl", "dzl_std", "dxl", "dxl_std", "dyl", "dyl_std"])
    for nl, res in sweep_results.items():
        for (e, ph, lo, lo_s, fi, fi_s, dz, dz_s,
             dx, dx_s, dy, dy_s) in zip(
                res['hist_epoch'], res['hist_phase'],
                res['hist_loss'],  res['hist_loss_std'],
                res['hist_fid'],   res['hist_fid_std'],
                res['hist_dzl'],   res['hist_dzl_std'],
                res['hist_dxl'],   res['hist_dxl_std'],
                res['hist_dyl'],   res['hist_dyl_std']):
            w.writerow([nl, int(e), int(ph),
                        f"{lo:.6f}", f"{lo_s:.6f}",
                        f"{fi:.6f}", f"{fi_s:.6f}",
                        f"{dz:.6f}", f"{dz_s:.6f}",
                        f"{dx:.6f}", f"{dx_s:.6f}",
                        f"{dy:.6f}", f"{dy_s:.6f}"])
print(f"  Saved : perfect_history_{FILE_TAG}.csv  (long format, all epochs, mean+std)")

# ── Summary CSV (test-set results) — F_theory removed ──
with open(f"perfect_results_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "n_params",
                "F_noisy_mean",  "F_noisy_std",
                "F_stdQEC_mean", "F_stdQEC_std",
                "F_qae_mean",    "F_qae_std",
                "dzl_mean", "dxl_mean", "dyl_mean",
                "gap_closed_pct"])
    for nl, res in sweep_results.items():
        fn  = res['f_noisy_test']; fs = res['f_std_test']; fq = res['f_qae_test']
        gap = fs.mean() - fn.mean()
        pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
        w.writerow([nl, res['n_params'],
                    f"{fn.mean():.5f}", f"{fn.std():.5f}",
                    f"{fs.mean():.5f}", f"{fs.std():.5f}",
                    f"{fq.mean():.5f}", f"{fq.std():.5f}",
                    f"{res['dzl_test'].mean():.5f}",
                    f"{res['dxl_test'].mean():.5f}",
                    f"{res['dyl_test'].mean():.5f}",
                    f"{pct:.1f}"])
print(f"  Saved : perfect_results_{FILE_TAG}.csv")

# ──────────────────────────────────────────────────────────────
# 15. Plot — full training curve from GLOBAL epoch 0 (Phase 1) → final
# ──────────────────────────────────────────────────────────────
colors  = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
markers = ['o', 's', '^', 'D', 'v']
ls_ops  = ['-', '--', '-', '--', '-']

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ── (0,0)  Fidelity vs epoch (full curve, both phases) + std band ──
ax = axes[0, 0]
for (nl, res), col, mk, ls in zip(sweep_results.items(), colors, markers, ls_ops):
    ax.plot(res['hist_epoch'], res['hist_fid'],
            color=col, linestyle=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p)  "
                  f"F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(res['hist_epoch'],
                    res['hist_fid'] - res['hist_fid_std'],
                    res['hist_fid'] + res['hist_fid_std'],
                    color=col, alpha=0.12)

ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5,
           label=f"Std QEC  {f_std_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(INIT_EPOCHS+1, 0.625, "Phase 2 →", fontsize=8, color='gray')
ax.text(1, 0.625, "← Phase 1", fontsize=8, color='gray')
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)", fontsize=12)
ax.set_ylabel(f"Avg fidelity (all {N_TRAIN_STATES} train states)", fontsize=12)
ax.set_title("Training fidelity vs epoch (full curve, both phases, ±1σ band)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim([0.60, 1.02]); ax.grid(True, alpha=0.3)

# ── (0,1)  Baseline loss vs epoch (full curve, both phases) ───
ax = axes[0, 1]
for (nl, res), col, ls in zip(sweep_results.items(), colors, ls_ops):
    ax.plot(res['hist_epoch'], res['hist_loss'],
            color=col, linestyle=ls, lw=2, label=f"L={nl}")
    ax.fill_between(res['hist_epoch'],
                    res['hist_loss'] - res['hist_loss_std'],
                    res['hist_loss'] + res['hist_loss_std'],
                    color=col, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(INIT_EPOCHS+1, ax.get_ylim()[1]*0.9, "Phase 2 →",
        fontsize=8, color='gray')
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)", fontsize=12)
ax.set_ylabel("Fixed-weight baseline spectral loss", fontsize=12)
ax.set_title(f"Baseline loss vs epoch (full curve, ±1σ band)\n"
             f"weights S={W_STAB_BASE}, ZL={W_ZL_BASE}, "
             f"XL={W_XL_BASE}, YL={W_YL_BASE}",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)

# ── (1,0)  Test fidelity bar chart ───────────────────────────
ax = axes[1, 0]
x    = np.arange(len(LAYER_SWEEP)); bw = 0.22
fn_m = [sweep_results[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep_results[nl]['f_std_test'].mean()   for nl in LAYER_SWEEP]
fq_m = [sweep_results[nl]['f_qae_test'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep_results[nl]['f_qae_test'].std()    for nl in LAYER_SWEEP]

ax.bar(x-bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x,    fs_m, bw, label="Standard QEC",  color='#3498db', alpha=0.8)
bars = ax.bar(x+bw, fq_m, bw, label="Spectral QEC",   color='#2ecc71', alpha=0.8)
ax.errorbar(x+bw, fq_m, yerr=fq_s,
            fmt='none', color='black', capsize=4, lw=1.5)
for bar, val in zip(bars, fq_m):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.003,
            f"{val:.4f}", ha='center', va='bottom',
            fontsize=8, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP], fontsize=11)
ax.set_ylabel("Mean fidelity (1 000 test states)", fontsize=12)
ax.set_title("Test fidelity vs layer count", fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim([0.60, 1.02])
ax.grid(True, alpha=0.3, axis='y')

# ── (1,1)  Logical errors bar chart ──────────────────────────
ax = axes[1, 1]
dzl_m = [sweep_results[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dxl_m = [sweep_results[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dyl_m = [sweep_results[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
bw2   = 0.25

ax.bar(x-bw2, dzl_m, bw2, label="ΔZL", color='#e74c3c', alpha=0.85)
ax.bar(x,     dxl_m, bw2, label="ΔXL", color='#3498db', alpha=0.85)
ax.bar(x+bw2, dyl_m, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
for xi, (dz, dx, dy) in enumerate(zip(dzl_m, dxl_m, dyl_m)):
    ax.text(xi-bw2, dz+0.002, f"{dz:.3f}", ha='center',
            va='bottom', fontsize=7)
    ax.text(xi,     dx+0.002, f"{dx:.3f}", ha='center',
            va='bottom', fontsize=7)
    ax.text(xi+bw2, dy+0.002, f"{dy:.3f}", ha='center',
            va='bottom', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP], fontsize=11)
ax.set_ylabel("Mean |target − recovered| (1 000 test states)", fontsize=12)
ax.set_title("Logical spectrum error vs layer count\n"
             "(lower = better logical info restoration)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f"Spectral loss + QAE — [[5,1,3]] Layer Sweep  |  p={P_NOISE}  "
    f"|  JAX {jax.default_backend().upper()}\n"
    f"Ansatz: RY+RZ / CRX+CRY+CRZ  (no RX)  |  "
    f"vmap chunk={CHUNK_EVAL_SIZE}  |  "
    f"weight update every {WEIGHT_UPDATE_FREQ} epochs  |  "
    f"monitoring on all {N_TRAIN_STATES} states (mean±std)  |  "
    f"layerwise warm-start={USE_LAYERWISE}",
    fontsize=11, fontweight='bold', y=1.02)

plt.tight_layout()
fig.savefig(f"perfect_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches="tight")
print(f"  Saved : perfect_fidelity_{FILE_TAG}.png")
plt.show()

# ──────────────────────────────────────────────────────────────
# 16. Final summary table  (F_theory removed)
# ──────────────────────────────────────────────────────────────
print("\n" + "="*80)
print(f"  FINAL SUMMARY — {N_TEST_STATES} test states  |  p={P_NOISE}  "
      f"|  RY+RZ / CRX+CRY+CRZ ansatz")
print("="*80)
hdr = (f"  {'L':>2}  {'Params':>7}  "
       f"{'F_noisy':>9}  {'F_stdQEC':>9}  "
       f"{'F_QAE':>8}  {'±std':>7}  "
       f"{'ΔZL':>7}  {'ΔXL':>7}  {'ΔYL':>7}  "
       f"{'Gap%':>6}")
print(hdr)
print("  " + "─"*(len(hdr)-2))

for nl in LAYER_SWEEP:
    res  = sweep_results[nl]
    fn   = res['f_noisy_test']
    fs   = res['f_std_test']
    fq   = res['f_qae_test']
    gap  = fs.mean() - fn.mean()
    pct  = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
    dzl  = res['dzl_test'].mean()
    dxl  = res['dxl_test'].mean()
    dyl  = res['dyl_test'].mean()
    print(f"  {nl:>2}  {res['n_params']:>7}  "
          f"{fn.mean():>9.5f}  {fs.mean():>9.5f}  "
          f"{fq.mean():>8.5f}  {fq.std():>7.5f}  "
          f"{dzl:>7.4f}  {dxl:>7.4f}  {dyl:>7.4f}  "
          f"{pct:>5.1f}%")

best_nl = max(LAYER_SWEEP,
              key=lambda nl: sweep_results[nl]['f_qae_test'].mean())
print(f"\n  Best layer count : L={best_nl}  "
      f"F̄={sweep_results[best_nl]['f_qae_test'].mean():.5f}  "
      f"ΔZL={sweep_results[best_nl]['dzl_test'].mean():.4f}  "
      f"ΔXL={sweep_results[best_nl]['dxl_test'].mean():.4f}  "
      f"ΔYL={sweep_results[best_nl]['dyl_test'].mean():.4f}")

# ──────────────────────────────────────────────────────────────
# 17. Fidelity vs error rate — BEST trained model
#     (best params + best layer count; model trained at p=P_NOISE,
#      evaluated across a range of depolarizing rates p < 0.5)
# ──────────────────────────────────────────────────────────────
print("\n" + "="*68)
print("  FIDELITY vs ERROR RATE  (best model)")
print("="*68)

best_nl_p   = max(LAYER_SWEEP,
                  key=lambda nl: sweep_results[nl]['f_qae_test'].mean())
best_params = jnp.array(sweep_results[best_nl_p]['params'])
best_comps  = make_layer_components(best_nl_p)
best_qae    = best_comps['qae_batch_chunk']
print(f"  Best model: L={best_nl_p}  "
      f"({sweep_results[best_nl_p]['n_params']} params)  trained at p={P_NOISE}")

P_EVAL_SWEEP = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
N_P_EVAL     = 1000

fqae_vs_p = []; fqae_std_vs_p = []
fnoisy_vs_p = []; fstd_vs_p = []

for p in P_EVAL_SWEEP:
    ds    = generate_dataset(N_P_EVAL, p)
    rho_n = jnp.array(np.stack([d['rho_noisy'] for d in ds]), dtype=complex)
    rho_c = jnp.array(np.stack([d['rho_clean'] for d in ds]), dtype=complex)

    fq = eval_fidelity_chunked(best_params, best_qae, rho_n, rho_c,
                               chunk_size=CHUNK_EVAL_SIZE)
    fn = np.array([float(np.real(np.trace(ds[i]['rho_noisy']
                                          @ ds[i]['rho_clean'])))
                   for i in range(N_P_EVAL)])
    fstd = np.array([
        float(np.real(np.trace(standard_qec_5q(ds[i]['rho_noisy'])
                               @ ds[i]['rho_clean'])))
        for i in range(N_P_EVAL)])

    fqae_vs_p.append(fq.mean());   fqae_std_vs_p.append(fq.std())
    fnoisy_vs_p.append(fn.mean()); fstd_vs_p.append(fstd.mean())
    print(f"  p={p:>4.2f}  F_QAE={fq.mean():.4f}  F_std={fstd.mean():.4f}  "
          f"F_noisy={fn.mean():.4f}")

P_EVAL_SWEEP  = np.array(P_EVAL_SWEEP)
fqae_vs_p     = np.array(fqae_vs_p);   fqae_std_vs_p = np.array(fqae_std_vs_p)
fnoisy_vs_p   = np.array(fnoisy_vs_p); fstd_vs_p     = np.array(fstd_vs_p)

np.savez(f"perfect_fidelity_vs_p_{FILE_TAG}.npz",
         best_n_layers=np.array(best_nl_p),
         best_params=np.array(sweep_results[best_nl_p]['params']),
         p_eval=P_EVAL_SWEEP,
         f_qae=fqae_vs_p, f_qae_std=fqae_std_vs_p,
         f_noisy=fnoisy_vs_p, f_std=fstd_vs_p)
with open(f"perfect_fidelity_vs_p_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p", "F_qae", "F_qae_std", "F_noisy", "F_stdQEC"])
    for i in range(len(P_EVAL_SWEEP)):
        w.writerow([f"{P_EVAL_SWEEP[i]:.3f}",
                    f"{fqae_vs_p[i]:.6f}", f"{fqae_std_vs_p[i]:.6f}",
                    f"{fnoisy_vs_p[i]:.6f}", f"{fstd_vs_p[i]:.6f}"])
print(f"  Saved : perfect_fidelity_vs_p_{FILE_TAG}.npz , perfect_fidelity_vs_p_{FILE_TAG}.csv")

fig2, ax = plt.subplots(figsize=(9, 6))
ax.plot(P_EVAL_SWEEP, fqae_vs_p, 'o-', color='#2ecc71', lw=2,
        label=f"Spectral QAE (L={best_nl_p})")
ax.fill_between(P_EVAL_SWEEP, fqae_vs_p-fqae_std_vs_p,
                fqae_vs_p+fqae_std_vs_p, color='#2ecc71', alpha=0.15)
ax.plot(P_EVAL_SWEEP, fstd_vs_p,   's--', color='#3498db', lw=2,
        label="Standard QEC")
ax.plot(P_EVAL_SWEEP, fnoisy_vs_p, '^-',  color='#e74c3c', lw=2,
        label="No correction")
ax.axvline(P_NOISE, color='black', ls=':', lw=1.2, alpha=0.6)
ax.text(P_NOISE+0.005, 0.60, f"trained @ p={P_NOISE}", fontsize=8, color='black')
ax.set_xlabel("Depolarizing error rate  p", fontsize=12)
ax.set_ylabel("Mean fidelity", fontsize=12)
ax.set_title(f"Fidelity vs error rate — best model "
             f"(L={best_nl_p}, trained at p={P_NOISE})",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_ylim([0.55, 1.02])
plt.tight_layout()
fig2.savefig(f"perfect_fidelity_vs_p_{FILE_TAG}.png", dpi=150, bbox_inches="tight")
print(f"  Saved : perfect_fidelity_vs_p_{FILE_TAG}.png")
plt.show()

print(f"\n  Saved files:")
print(f"    ./perfect_fidelity_{FILE_TAG}.png")
print(f"    ./perfect_results_{FILE_TAG}.npz        (params + per-epoch history + test results)")
print(f"    ./perfect_results_{FILE_TAG}.csv        (per-layer test summary)")
print(f"    ./perfect_history_{FILE_TAG}.csv        (long-format, every epoch, both phases)")
print(f"    ./perfect_fidelity_vs_p_{FILE_TAG}.png  (best model fidelity vs error rate)")
print(f"    ./perfect_fidelity_vs_p_{FILE_TAG}.csv  ./perfect_fidelity_vs_p_{FILE_TAG}.npz")
print("\n  Done ✓")