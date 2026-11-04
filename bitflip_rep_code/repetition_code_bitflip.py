"""
Spectral Measurement-Free QEC 
=========================================================================
LAYER-SWEEP VERSION:  N_LAYERS ∈ {1, 2, 3, 4, 5}  on the [[3,1,3]] bit-flip code.

QAE Protocol:
  1. ρ_noisy(3q) ⊗ |0⟩(anc) → V(θ) · ρ · V†(θ)
  2. Trace out SYSTEM → ρ_anc (2×2, holds logical info)
  3. |000⟩(fresh) ⊗ ρ_anc → V†(θ) · ρ · V(θ)
  4. Trace out ANCILLA → ρ_corrected (8×8)

Ansatz: [Ry+Rz] → { CRX+CRY+CRZ (all-to-all) → [Ry+Rz] } × N_LAYERS

Output: bitflip_fidelity.png
        bitflip_results.npz
        bitflip_results.csv
        bitflip_history.csv
        bitflip_fidelity_vs_p.png / .csv / .npz
"""

import warnings; warnings.filterwarnings("ignore")
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

# ============================================================
# JAX CONFIGURATION
# ============================================================
import jax
jax.config.update("jax_enable_x64", True)

import logging
logging.getLogger("jax").setLevel(logging.WARNING)
try:
    jax.config.update("jax_platform_name", "gpu")
except Exception:
    jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import optax

print(f"  JAX backend: {jax.default_backend().upper()}")
print(f"  JAX devices: {jax.devices()}")

import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from functools import reduce
from itertools import product as iter_product
import time
import gc
import csv

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================
N_CODE = 3
N_ANC = 1
N_TOTAL = N_CODE + N_ANC  # 4 qubits

P_NOISE = 0.1
N_TRAIN_STATES = 100
N_TEST_STATES = 1000
BATCH_SIZE = 5            # Per-state JIT: accumulate this many before update
LR = 0.08

# ── Two-phase schedule (Phase 1 warm-up + Phase 2 continuation) ──
N_INIT_ROUNDS = 12
INIT_EPOCHS   = 20
MAIN_EPOCHS   = 180
TOTAL_EPOCHS  = INIT_EPOCHS + MAIN_EPOCHS   # 200

# ── SWEEP over layer counts ──
LAYER_SWEEP = [1, 2, 3, 4, 5]

# Layerwise warm-start: grow L -> L+1 from previous trained solution by
# appending identity (zero-angle) blocks. Set False for independent inits.
USE_LAYERWISE = True

# For file names
if USE_LAYERWISE:
    FILE_TAG = "layerwise"
else:
    FILE_TAG = "multistart"

W_STAB = 1.0
W_LOGICAL = 15.0
W_YL = 15.0

# ============================================================
# PRINT CONFIGURATION
# ============================================================
print("=" * 70)
print("  SPECTRAL MEASUREMENT-FREE QEC — QAE MODEL B (EFFICIENT, LAYER SWEEP)")
print("  Pure JAX matrix ops + per-state JIT | 3q + 1 ancilla")
print("=" * 70)
print(f"""
  Architecture:
    Code qubits: {N_CODE}, Ancilla: {N_ANC}, Total: {N_TOTAL}
    Ansatz: Model B (all-to-all CRX+CRY+CRZ + Ry/Rz)
    Layer sweep: {LAYER_SWEEP}   layerwise warm-start: {USE_LAYERWISE}

  Training:
    Noise p: {P_NOISE}, States: {N_TRAIN_STATES}, Test: {N_TEST_STATES}
    Phase 1: {N_INIT_ROUNDS} rounds x {INIT_EPOCHS} ep | Phase 2: {MAIN_EPOCHS} ep
    Total epochs: {TOTAL_EPOCHS}, LR: {LR}, Batch: {BATCH_SIZE}
    Loss weights: W_stab={W_STAB}, W_log={W_LOGICAL}, W_yl={W_YL}
""")

# ============================================================
# QUANTUM PRIMITIVES
# ============================================================
I2 = np.eye(2, dtype=complex)
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops):
    return reduce(np.kron, ops)

ket_000 = np.zeros(8, dtype=complex); ket_000[0] = 1.0
ket_111 = np.zeros(8, dtype=complex); ket_111[7] = 1.0

S1_mat = kron_list([Z_mat, Z_mat, I2])
S2_mat = kron_list([I2, Z_mat, Z_mat])
ZL_mat = kron_list([Z_mat, Z_mat, Z_mat])
XL_mat = kron_list([X_mat, X_mat, X_mat])
YL_mat = kron_list([Y_mat, Y_mat, Y_mat])

S1_jax = jnp.array(S1_mat)
S2_jax = jnp.array(S2_mat)
ZL_jax = jnp.array(ZL_mat)
XL_jax = jnp.array(XL_mat)
YL_jax = jnp.array(YL_mat)

ANC_0 = jnp.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
FRESH_000 = jnp.zeros((8, 8), dtype=complex).at[0, 0].set(1.0)

def logical_state(theta, phi):
    return np.cos(theta/2) * ket_000 + np.exp(1j*phi) * np.sin(theta/2) * ket_111

def state_to_dm(psi):
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

def bit_flip_channel(rho, p):
    result = np.zeros_like(rho)
    for i, j, k in iter_product([0, 1], repeat=3):
        prob = (p if i else 1-p) * (p if j else 1-p) * (p if k else 1-p)
        E = kron_list([X_mat if b else I2 for b in [i, j, k]])
        result += prob * E @ rho @ E.conj().T
    return result

def single_bitflip_only(rho, p):
    p_none = (1-p)**3
    p_single = p * (1-p)**2
    total = p_none + 3 * p_single
    result = (p_none/total) * rho
    for q in range(3):
        ops = [X_mat if i == q else I2 for i in range(3)]
        E = kron_list(ops)
        result += (p_single/total) * E @ rho @ E.conj().T
    return result

def standard_qec_recovery(rho_noisy):
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    Pi_00 = kron_list([P0, P0, P0]) + kron_list([P1, P1, P1])
    Pi_10 = kron_list([P1, P0, P0]) + kron_list([P0, P1, P1])
    Pi_01 = kron_list([P0, P0, P1]) + kron_list([P1, P1, P0])
    Pi_11 = kron_list([P0, P1, P0]) + kron_list([P1, P0, P1])
    X0 = kron_list([X_mat, I2, I2])
    X1 = kron_list([I2, X_mat, I2])
    X2 = kron_list([I2, I2, X_mat])
    I8 = np.eye(8, dtype=complex)
    corrections = [(Pi_00, I8), (Pi_10, X0), (Pi_11, X1), (Pi_01, X2)]
    rho_out = np.zeros_like(rho_noisy)
    for Pi, R in corrections:
        rho_out += R @ Pi @ rho_noisy @ Pi @ R
    return rho_out

# ============================================================
# JAX STANDARD-QEC RECOVERY (constants + kernel) — for batched eval
# ============================================================
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_Pi_00 = kron_list([_P0, _P0, _P0]) + kron_list([_P1, _P1, _P1])
_Pi_10 = kron_list([_P1, _P0, _P0]) + kron_list([_P0, _P1, _P1])
_Pi_01 = kron_list([_P0, _P0, _P1]) + kron_list([_P1, _P1, _P0])
_Pi_11 = kron_list([_P0, _P1, _P0]) + kron_list([_P1, _P0, _P1])
_X0 = kron_list([X_mat, I2, I2])
_X1 = kron_list([I2, X_mat, I2])
_X2 = kron_list([I2, I2, X_mat])
_I8 = np.eye(8, dtype=complex)

CORR_PI = jnp.array(np.stack([_Pi_00, _Pi_10, _Pi_11, _Pi_01]))
CORR_R  = jnp.array(np.stack([_I8,    _X0,    _X1,    _X2]))

def std_qec_jax(rho):
    """Standard [[3,1,3]] recovery, pure JAX (single 8×8 state)."""
    def one(Pi, R):
        return R @ Pi @ rho @ Pi @ R
    return jnp.sum(jax.vmap(one)(CORR_PI, CORR_R), axis=0)

# ============================================================
# ENCODER ANSATZ (layer-parametrised)
# ============================================================
def count_model_b_params(n_layers):
    n_rot_params = (n_layers + 1) * N_TOTAL * 2
    n_cr_params = n_layers * N_TOTAL * (N_TOTAL - 1) * 3
    return n_rot_params + n_cr_params

def model_b_ansatz(params, n_layers):
    """
    Model B: [Ry+Rz] → { CRX+CRY+CRZ (all-to-all) → [Ry+Rz] } × n_layers
    """
    idx = 0
    for q in range(N_TOTAL):
        qml.RY(params[idx], wires=q); idx += 1
        qml.RZ(params[idx], wires=q); idx += 1
    for layer in range(n_layers):
        for control in range(N_TOTAL):
            for target in range(N_TOTAL):
                if control != target:
                    qml.CRX(params[idx], wires=[control, target]); idx += 1
                    qml.CRY(params[idx], wires=[control, target]); idx += 1
                    qml.CRZ(params[idx], wires=[control, target]); idx += 1
        for q in range(N_TOTAL):
            qml.RY(params[idx], wires=q); idx += 1
            qml.RZ(params[idx], wires=q); idx += 1
    return idx

# ============================================================
# LOSS: EXPECTATIONS FROM DENSITY MATRIX (pure JAX, layer-independent)
# ============================================================
def expectations_from_dm(rho_3q):
    """Compute expectations directly from 8×8 density matrix."""
    s1 = jnp.real(jnp.trace(S1_jax @ rho_3q))
    s2 = jnp.real(jnp.trace(S2_jax @ rho_3q))
    zl = jnp.real(jnp.trace(ZL_jax @ rho_3q))
    xl = jnp.real(jnp.trace(XL_jax @ rho_3q))
    yl = jnp.real(jnp.trace(YL_jax @ rho_3q))
    return s1, s2, zl, xl, yl

# ============================================================
# PARTIAL-TRACE HELPERS (layer-independent)
# ============================================================
def partial_trace_system(rho_16):
    """Trace out system qubits [0,1,2] (dim 8). Keep ancilla [3] (dim 2)."""
    rho_r = rho_16.reshape(8, 2, 8, 2)
    return jnp.trace(rho_r, axis1=0, axis2=2)

def partial_trace_ancilla(rho_16):
    """Trace out ancilla [3] (dim 2). Keep system [0,1,2] (dim 8)."""
    rho_r = rho_16.reshape(8, 2, 8, 2)
    return jnp.trace(rho_r, axis1=1, axis2=3)

# ============================================================
# PER-LAYER FACTORY
# ============================================================
def make_layer_components(n_layers):
    n_params = count_model_b_params(n_layers)

    def encoder_circuit(params):
        model_b_ansatz(params, n_layers)

    encoder_matrix_fn = qml.matrix(encoder_circuit, wire_order=range(N_TOTAL))

    # ── Pure-JAX QAE forward pass (no QNodes) ──
    def qae_forward(params, rho_noisy_3q):
        V = encoder_matrix_fn(params)          # 16×16
        V_dag = jnp.conj(V.T)                  # 16×16
        rho_in = jnp.kron(rho_noisy_3q, ANC_0)          # 16×16
        rho_encoded = V @ rho_in @ V_dag                 # 16×16
        rho_anc = partial_trace_system(rho_encoded)      # 2×2
        rho_decode_in = jnp.kron(FRESH_000, rho_anc)     # 16×16
        rho_decoded = V_dag @ rho_decode_in @ V          # 16×16
        rho_out = partial_trace_ancilla(rho_decoded)     # 8×8
        return rho_out

    # ── Per-state JIT loss+grad ──
    @jax.jit
    def single_state_loss_and_grad(params, rho_noisy, target):
        def loss_fn(p):
            rho_out = qae_forward(p, rho_noisy)
            s1, s2, zl, xl, yl = expectations_from_dm(rho_out)
            loss_stab = W_STAB * (-(s1 + s2))
            loss_log = W_LOGICAL * ((target[0] - zl)**2 + (target[1] - xl)**2)
            loss_yl = W_YL * (target[2] - yl)**2
            return loss_stab + loss_log + loss_yl
        return jax.value_and_grad(loss_fn)(params)

    # ── JIT + vmap batch eval (fidelity + stabilizers) ──
    @jax.jit
    def batch_eval(params, rho_noisy_stack, rho_clean_stack):
        def single(rho_noisy, rho_clean):
            rho_out = qae_forward(params, rho_noisy)
            fid = jnp.real(jnp.trace(rho_out @ rho_clean))
            s1 = jnp.real(jnp.trace(S1_jax @ rho_out))
            s2 = jnp.real(jnp.trace(S2_jax @ rho_out))
            return fid, s1, s2
        return jax.vmap(single)(rho_noisy_stack, rho_clean_stack)

    # ── Per-epoch monitoring: loss + fidelity + stabilizers (arrays) ──
    @jax.jit
    def batch_metrics(params, rho_noisy_stack, rho_clean_stack, targets_stack):
        def single(rho_noisy, rho_clean, target):
            rho_out = qae_forward(params, rho_noisy)
            s1, s2, zl, xl, yl = expectations_from_dm(rho_out)
            fid = jnp.real(jnp.trace(rho_out @ rho_clean))
            loss = (W_STAB * (-(s1 + s2))
                    + W_LOGICAL * ((target[0]-zl)**2 + (target[1]-xl)**2)
                    + W_YL * (target[2]-yl)**2)
            return loss, fid, s1, s2
        return jax.vmap(single)(rho_noisy_stack, rho_clean_stack, targets_stack)

    # ── Full batched eval: noisy / std-QEC / QAE + stabilizers ──
    @jax.jit
    def batch_eval_full(params, rho_noisy_stack, rho_clean_stack):
        def single(rho_noisy, rho_clean):
            rho_out = qae_forward(params, rho_noisy)
            f_qae   = jnp.real(jnp.trace(rho_out @ rho_clean))
            s1      = jnp.real(jnp.trace(S1_jax @ rho_out))
            s2      = jnp.real(jnp.trace(S2_jax @ rho_out))
            f_noisy = jnp.real(jnp.trace(rho_noisy @ rho_clean))
            f_std   = jnp.real(jnp.trace(std_qec_jax(rho_noisy) @ rho_clean))
            return f_noisy, f_std, f_qae, s1, s2
        return jax.vmap(single)(rho_noisy_stack, rho_clean_stack)

    return dict(
        n_params                   = n_params,
        encoder_matrix_fn          = encoder_matrix_fn,
        qae_forward                = qae_forward,
        single_state_loss_and_grad = single_state_loss_and_grad,
        batch_eval                 = batch_eval,
        batch_metrics              = batch_metrics,
        batch_eval_full            = batch_eval_full,
    )

# ============================================================
# LAYERWISE GROWTH HELPER
#   params(L) is a PREFIX of params(L+1): appended zero-angle blocks are
#   identity, so the grown circuit starts numerically identical.
# ============================================================
def grow_params(params_prev, prev_nl, nl):
    extra = count_model_b_params(nl) - count_model_b_params(prev_nl)
    return np.concatenate([np.asarray(params_prev), np.zeros(extra)])

# ============================================================
# TRAINING DATA (layer-independent → built once, shared)
# ============================================================
def generate_fibonacci_sphere_states(n_states):
    states = []
    golden = (1 + np.sqrt(5)) / 2
    for i in range(n_states):
        theta = np.arccos(1 - 2*(i + 0.5)/n_states)
        phi = (2 * np.pi * i / golden) % (2*np.pi)
        states.append((theta, phi))
    return states

def generate_training_data(n_states, p_noise):
    data = []
    points = generate_fibonacci_sphere_states(n_states)
    for theta, phi in points:
        psi = logical_state(theta, phi)
        rho_clean = state_to_dm(psi)
        rho_noisy = bit_flip_channel(rho_clean, p_noise)
        data.append({
            'theta': theta, 'phi': phi,
            'rho_clean': rho_clean, 'rho_noisy': rho_noisy,
            'zl': np.real(np.trace(ZL_mat @ rho_clean)),
            'xl': np.real(np.trace(XL_mat @ rho_clean)),
            'yl': np.real(np.trace(YL_mat @ rho_clean)),
        })
    return data

print("\n  Generating training data...")
training_data = generate_training_data(N_TRAIN_STATES, P_NOISE)

rho_noisy_list = [jnp.array(d['rho_noisy'], dtype=complex) for d in training_data]
targets_list = [jnp.array([d['zl'], d['xl'], d['yl']]) for d in training_data]

rho_noisy_stack = jnp.stack(rho_noisy_list)                                  # (N,8,8)
rho_clean_stack = jnp.stack(
    [jnp.array(d['rho_clean'], dtype=complex) for d in training_data])       # (N,8,8)
targets_stack = jnp.stack(targets_list)                                      # (N,3)
print(f"  ✓ {N_TRAIN_STATES} training states ready")

# ── Test-set stacks (built once, layer-independent) ──
print(f"  Generating test data ({N_TEST_STATES} states, full + single channels)...")
test_points = generate_fibonacci_sphere_states(N_TEST_STATES)
rho_clean_test_np   = np.empty((N_TEST_STATES, 8, 8), dtype=complex)
rho_noisy_full_np   = np.empty((N_TEST_STATES, 8, 8), dtype=complex)
rho_noisy_single_np = np.empty((N_TEST_STATES, 8, 8), dtype=complex)
for i, (theta, phi) in enumerate(test_points):
    rc = state_to_dm(logical_state(theta, phi))
    rho_clean_test_np[i]   = rc
    rho_noisy_full_np[i]   = bit_flip_channel(rc, P_NOISE)
    rho_noisy_single_np[i] = single_bitflip_only(rc, P_NOISE)

rho_clean_test_j   = jnp.array(rho_clean_test_np)
rho_noisy_full_j   = jnp.array(rho_noisy_full_np)
rho_noisy_single_j = jnp.array(rho_noisy_single_np)
test_thetas = np.array([t for t, _ in test_points])
test_phis   = np.array([p for _, p in test_points])
print(f"  ✓ {N_TEST_STATES} test states ready")

# ============================================================
# ONE FULL TRAINING EPOCH (shared by Phase 1 and Phase 2)
# ============================================================
def run_epoch(params, opt_state, optimizer, rng_key, single_state_loss_and_grad):
    steps = int(np.ceil(N_TRAIN_STATES / BATCH_SIZE))
    rng_key, sk = jax.random.split(rng_key)
    idx_all = np.array(jax.random.permutation(sk, N_TRAIN_STATES))
    ep_loss = 0.0; nb = 0
    for step in range(steps):
        b = idx_all[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        g_acc = jnp.zeros_like(params); l_acc = 0.0
        for i in b:
            lv, gv = single_state_loss_and_grad(
                params, rho_noisy_list[i], targets_list[i])
            gv = jnp.where(jnp.isnan(gv), 0.0, gv)
            lv = jnp.where(jnp.isnan(lv), 0.0, lv)
            g_acc += gv; l_acc += float(lv)
        updates, opt_state = optimizer.update(g_acc / len(b), opt_state, params)
        params = optax.apply_updates(params, updates)
        ep_loss += l_acc / len(b); nb += 1
    return params, opt_state, ep_loss / max(nb, 1), rng_key

# ============================================================
# ONE FULL TRAINING RUN FOR A GIVEN LAYER COUNT
#   FULL history: Phase 1 (winning warm-up trajectory) + Phase 2, on a
#   continuous global epoch index + phase marker.
#   init_params given -> layerwise warm-start (single Phase-1 refinement);
#   otherwise -> multi-start Phase-1 (L=1).
# ============================================================
def train_one_config(n_layers, rng_seed=42, init_params=None):
    comps = make_layer_components(n_layers)
    n_params                   = comps['n_params']
    single_state_loss_and_grad = comps['single_state_loss_and_grad']
    batch_metrics              = comps['batch_metrics']
    batch_eval_full            = comps['batch_eval_full']

    print("\n" + "=" * 70)
    print(f"  N_LAYERS = {n_layers}   params = {n_params}")
    if init_params is not None:
        print(f"  Init: LAYERWISE WARM-START (identity-grown from previous L)")
    else:
        print(f"  Init: MULTI-START warm-up ({N_INIT_ROUNDS} rounds)")
    print("=" * 70)

    def probe_metrics(p):
        lo, fd, s1, s2 = batch_metrics(
            p, rho_noisy_stack, rho_clean_stack, targets_stack)
        return (float(jnp.mean(lo)), float(jnp.mean(fd)),
                float(jnp.mean(s1)), float(jnp.mean(s2)))

    # ── JIT warm-up ──
    print("  JIT warm-up ...", end=" ", flush=True)
    t0 = time.time()
    _p0 = jnp.zeros(n_params)
    single_state_loss_and_grad(_p0, rho_noisy_list[0], targets_list[0])
    batch_metrics(_p0, rho_noisy_stack, rho_clean_stack, targets_stack)
    batch_eval_full(_p0, rho_noisy_full_j, rho_clean_test_j)[0].block_until_ready()
    print(f"done in {time.time()-t0:.1f}s")

    STEPS_PER_EPOCH = int(np.ceil(N_TRAIN_STATES / BATCH_SIZE))
    # One cosine LR schedule for the whole run (Phase 1 + Phase 2)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(LR, TOTAL_EPOCHS*STEPS_PER_EPOCH)))

    rng = jax.random.PRNGKey(rng_seed)

    # ── Phase 1 ──
    best_p = best_opt = None
    best_f = -1.0
    best_rng = rng
    best_round_hist = None

    if init_params is not None:
        print(f"\n  Phase 1 : LAYERWISE WARM-START — single {INIT_EPOCHS}-epoch "
              f"refinement (no multi-start)")
        tp = jnp.asarray(init_params, dtype=jnp.float64)
        to = optimizer.init(tp)
        rng, rnd_rng = jax.random.split(rng)
        round_hist = []
        for ep in range(INIT_EPOCHS):
            tp, to, _, rnd_rng = run_epoch(
                tp, to, optimizer, rnd_rng, single_state_loss_and_grad)
            round_hist.append((ep, 1, *probe_metrics(tp)))
        best_f = round_hist[-1][3]      # fidelity is index 3 (ep, phase, loss, fid, s1, s2)
        best_p = tp; best_opt = to; best_rng = rnd_rng
        best_round_hist = round_hist
        print(f"  Warm-start refinement F = {best_f:.4f}  "
              f"(appended block = identity → non-destructive growth)")
    else:
        print(f"\n  Phase 1 : {N_INIT_ROUNDS} rounds × {INIT_EPOCHS} epochs")
        print(f"  {'Rd':<3} {'Init style':<20} {'F_end':>7} {'⟨S1⟩':>6} {'⟨S2⟩':>6}")
        print(f"  {'─'*3} {'─'*20} {'─'*7} {'─'*6} {'─'*6}")
        for rd in range(N_INIT_ROUNDS):
            rng, sk = jax.random.split(rng)
            s = rd % 4
            if s == 0:
                tp = jax.random.uniform(sk, shape=(n_params,), minval=-np.pi, maxval=np.pi)
                sn = "Uniform[-π,π]"
            elif s == 1:
                tp = jax.random.normal(sk, shape=(n_params,)) * 0.15
                sn = "Normal(0,0.15)"
            elif s == 2:
                tp = jax.random.normal(sk, shape=(n_params,)) * 1.0
                sn = "Normal(0,1.0)"
            else:
                tp = jax.random.uniform(sk, shape=(n_params,), minval=-0.1, maxval=0.1)
                sn = "Unif[-0.1,0.1]"

            to = optimizer.init(tp)
            rng, rnd_rng = jax.random.split(rng)
            round_hist = []
            for ep in range(INIT_EPOCHS):
                tp, to, _, rnd_rng = run_epoch(
                    tp, to, optimizer, rnd_rng, single_state_loss_and_grad)
                round_hist.append((ep, 1, *probe_metrics(tp)))

            f_end = round_hist[-1][3]
            s1_end = round_hist[-1][4]; s2_end = round_hist[-1][5]
            star = ""
            if f_end > best_f:
                best_f = f_end; best_p = tp; best_opt = to
                best_rng = rnd_rng; best_round_hist = round_hist
                star = " ★"
            print(f"  {rd:<3} {sn:<20} {f_end:>7.4f} {s1_end:>6.3f} {s2_end:>6.3f}{star}")
            if star == "":
                del to
            gc.collect()
        print(f"  Best warm-up F = {best_f:.4f}  (winning Phase-1 trajectory retained)")

    # ── Unified history seeded with winning Phase-1 trajectory ──
    hist_epoch = [h[0] for h in best_round_hist]
    hist_phase = [h[1] for h in best_round_hist]
    hist_loss  = [h[2] for h in best_round_hist]
    hist_fid   = [h[3] for h in best_round_hist]
    hist_s1    = [h[4] for h in best_round_hist]
    hist_s2    = [h[5] for h in best_round_hist]

    # ── Phase 2: continuation ──
    print(f"\n  Phase 2 : {MAIN_EPOCHS} continuation epochs "
          f"(global epochs {INIT_EPOCHS} … {TOTAL_EPOCHS-1})")
    params = best_p; opt_state = best_opt; rng_key = best_rng

    print(f"\n  {'Epoch':<6} {'Loss':>9} {'F_avg':>7} {'⟨S1⟩':>6} {'⟨S2⟩':>6} {'Time':>6}")
    print(f"  {'-'*6} {'-'*9} {'-'*7} {'-'*6} {'-'*6} {'-'*6}")

    t_total_start = time.time()
    for ep in range(MAIN_EPOCHS):
        t0 = time.time()
        params, opt_state, tr_loss, rng_key = run_epoch(
            params, opt_state, optimizer, rng_key, single_state_loss_and_grad)
        lo_m, fid_m, s1_m, s2_m = probe_metrics(params)
        total_ep = INIT_EPOCHS + ep
        hist_epoch.append(total_ep); hist_phase.append(2)
        hist_loss.append(lo_m); hist_fid.append(fid_m)
        hist_s1.append(s1_m); hist_s2.append(s2_m)

        if ep % 10 == 0 or ep == MAIN_EPOCHS - 1:
            print(f"  {total_ep:<6} {lo_m:>9.4f} {fid_m:>7.4f} "
                  f"{s1_m:>6.3f} {s2_m:>6.3f} {time.time()-t0:>5.2f}s")
        if ep % 20 == 0:
            gc.collect()

    t_total = time.time() - t_total_start
    print(f"\n  Phase-2 time: {t_total:.1f}s ({t_total/60:.1f}min)")

    hist_epoch = np.array(hist_epoch, dtype=np.int64)
    hist_phase = np.array(hist_phase, dtype=np.int64)
    hist_loss  = np.array(hist_loss,  dtype=np.float64)
    hist_fid   = np.array(hist_fid,   dtype=np.float64)
    hist_s1    = np.array(hist_s1,    dtype=np.float64)
    hist_s2    = np.array(hist_s2,    dtype=np.float64)

    # ── Final test evaluation (full + single channel) ──
    print("  Test eval (full + single bit-flip) ...", end=" ", flush=True)
    t0 = time.time()
    fn_f, fs_f, fq_f, s1_f, s2_f = batch_eval_full(params, rho_noisy_full_j, rho_clean_test_j)
    fn_f.block_until_ready()
    fn_s, fs_s, fq_s, s1_s, s2_s = batch_eval_full(params, rho_noisy_single_j, rho_clean_test_j)
    fn_s.block_until_ready()
    print(f"done in {time.time()-t0:.1f}s  "
          f"F_qae(full)={float(jnp.mean(fq_f)):.5f}  "
          f"F_qae(single)={float(jnp.mean(fq_s)):.5f}")

    return dict(
        n_layers        = n_layers,
        n_params        = n_params,
        params          = np.array(params),
        train_time_s    = t_total,
        # FULL continuous history (Phase 1 + Phase 2)
        hist_epoch      = hist_epoch,
        hist_phase      = hist_phase,
        loss_hist       = hist_loss,
        fid_hist        = hist_fid,
        s1_hist         = hist_s1,
        s2_hist         = hist_s2,
        # test-set (full channel)
        f_noisy_full    = np.asarray(fn_f),
        f_std_full      = np.asarray(fs_f),
        f_qae_full      = np.asarray(fq_f),
        s1_full         = np.asarray(s1_f),
        s2_full         = np.asarray(s2_f),
        # test-set (single flip)
        f_noisy_single  = np.asarray(fn_s),
        f_std_single    = np.asarray(fs_s),
        f_qae_single    = np.asarray(fq_s),
    )

# ============================================================
# RUN THE LAYER SWEEP  (with layerwise warm-start for L>1)
# ============================================================
print("\n" + "=" * 70)
print("  LAYER SWEEP")
print("=" * 70)
print(f"  Layers          : {LAYER_SWEEP}")
print(f"  Ansatz          : Model B  [Ry+Rz] / CRX+CRY+CRZ (all-to-all)")
print(f"  Noise           : bit-flip, p={P_NOISE}")
print(f"  History         : FULL — Phase 1 + Phase 2, every epoch")
print(f"  Layerwise init  : {USE_LAYERWISE}  (L=1 always uses multi-start)")

sweep_results = {}
t_sweep = time.time()

prev_nl = None; prev_params = None
for nl in LAYER_SWEEP:
    if (not USE_LAYERWISE) or (prev_params is None) or (nl <= prev_nl):
        sweep_results[nl] = train_one_config(nl, rng_seed=42 + nl)
    else:
        warm = grow_params(prev_params, prev_nl, nl)
        sweep_results[nl] = train_one_config(nl, rng_seed=42 + nl, init_params=warm)
    prev_nl = nl
    prev_params = sweep_results[nl]['params']

print(f"\n  Total sweep time : {time.time()-t_sweep:.0f}s "
      f"({(time.time()-t_sweep)/60:.1f} min)")

# ============================================================
# SAVE : bitflip_results.npz
# ============================================================
save_dict = {}
for nl, res in sweep_results.items():
    save_dict[f'L{nl}_params']          = res['params']
    save_dict[f'L{nl}_epoch']           = res['hist_epoch']
    save_dict[f'L{nl}_phase']           = res['hist_phase']
    save_dict[f'L{nl}_loss_hist']       = res['loss_hist']
    save_dict[f'L{nl}_fid_hist']        = res['fid_hist']
    save_dict[f'L{nl}_s1_hist']         = res['s1_hist']
    save_dict[f'L{nl}_s2_hist']         = res['s2_hist']
    save_dict[f'L{nl}_f_noisy_full']    = res['f_noisy_full']
    save_dict[f'L{nl}_f_std_full']      = res['f_std_full']
    save_dict[f'L{nl}_f_qae_full']      = res['f_qae_full']
    save_dict[f'L{nl}_s1_full']         = res['s1_full']
    save_dict[f'L{nl}_s2_full']         = res['s2_full']
    save_dict[f'L{nl}_f_noisy_single']  = res['f_noisy_single']
    save_dict[f'L{nl}_f_std_single']    = res['f_std_single']
    save_dict[f'L{nl}_f_qae_single']    = res['f_qae_single']

# Overall best model (by mean full-channel test fidelity)
best_nl_save = max(LAYER_SWEEP, key=lambda nl: sweep_results[nl]['f_qae_full'].mean())
save_dict['best_n_layers'] = np.array(best_nl_save)
save_dict['best_params']   = sweep_results[best_nl_save]['params']
save_dict['best_f_qae']    = np.array(sweep_results[best_nl_save]['f_qae_full'].mean())

save_dict['meta_layers']        = np.array(LAYER_SWEEP)
save_dict['meta_INIT_EPOCHS']   = np.array(INIT_EPOCHS)
save_dict['meta_MAIN_EPOCHS']   = np.array(MAIN_EPOCHS)
save_dict['meta_TOTAL_EPOCHS']  = np.array(TOTAL_EPOCHS)
save_dict['meta_p_noise']       = np.array(P_NOISE)
save_dict['meta_n_train']       = np.array(N_TRAIN_STATES)
save_dict['meta_n_test']        = np.array(N_TEST_STATES)
save_dict['meta_layerwise']     = np.array(int(USE_LAYERWISE))
save_dict['meta_test_thetas']   = test_thetas
save_dict['meta_test_phis']     = test_phis

np.savez(f"bitflip_results_{FILE_TAG}.npz", **save_dict)
print(f"\n  Saved : bitflip_results_{FILE_TAG}.npz")
print(f"    Best model : best_n_layers=L{best_nl_save}, best_params, best_f_qae")

# ============================================================
# SAVE : bitflip_history.csv  (long format, every epoch, both phases)
# ============================================================
with open(f"bitflip_history_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "global_epoch", "phase",
                "loss", "avg_fidelity", "s1_mean", "s2_mean"])
    for nl, res in sweep_results.items():
        for e, ph, lo, fi, s1v, s2v in zip(
                res['hist_epoch'], res['hist_phase'], res['loss_hist'],
                res['fid_hist'], res['s1_hist'], res['s2_hist']):
            w.writerow([nl, int(e), int(ph), f"{lo:.6f}", f"{fi:.6f}",
                        f"{s1v:.6f}", f"{s2v:.6f}"])
print(f"  Saved : bitflip_history_{FILE_TAG}.csv")

# ============================================================
# SAVE : bitflip_results.csv  (per-layer test summary) — F_theory removed
# ============================================================
with open(f"bitflip_results_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "n_params",
                "F_noisy_mean",  "F_noisy_std",
                "F_stdQEC_mean", "F_stdQEC_std",
                "F_qae_mean",    "F_qae_std",
                "F_qae_single_mean", "F_qae_single_min",
                "gap_closed_pct", "train_time_s"])
    for nl, res in sweep_results.items():
        fn = res['f_noisy_full']; fs = res['f_std_full']; fq = res['f_qae_full']
        gap = fs.mean() - fn.mean()
        pct = (fq.mean() - fn.mean()) / gap * 100 if gap > 1e-6 else 0.0
        w.writerow([nl, res['n_params'],
                    f"{fn.mean():.5f}", f"{fn.std():.5f}",
                    f"{fs.mean():.5f}", f"{fs.std():.5f}",
                    f"{fq.mean():.5f}", f"{fq.std():.5f}",
                    f"{res['f_qae_single'].mean():.5f}",
                    f"{res['f_qae_single'].min():.5f}",
                    f"{pct:.1f}", f"{res['train_time_s']:.1f}"])
print(f"  Saved : bitflip_results_{FILE_TAG}.csv")

# ============================================================
# PLOT : bitflip_fidelity.png  (4-panel layer comparison)
# ============================================================
matplotlib.use("Agg")
colors  = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
ls_ops  = ['-', '--', '-', '--', '-']

# Empirical reference lines (fixed test set → identical across layers)
f_std_ref   = sweep_results[LAYER_SWEEP[0]]['f_std_full'].mean()
f_noisy_ref = sweep_results[LAYER_SWEEP[0]]['f_noisy_full'].mean()

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# ── (0,0) Training fidelity vs global epoch (both phases) ──
ax = axes[0, 0]
for (nl, res), col, ls in zip(sweep_results.items(), colors, ls_ops):
    ax.plot(res['hist_epoch'], res['fid_hist'], color=col, linestyle=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p)  F̄={res['f_qae_full'].mean():.4f}")
ax.axhline(f_std_ref, color='navy', ls='--', lw=1.5, label=f"Std QEC  {f_std_ref:.4f}")
ax.axhline(f_noisy_ref, color='gray', ls=':', lw=1.5, label=f"No correction {f_noisy_ref:.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.text(INIT_EPOCHS+1, 0.53, "Phase 2 →", fontsize=8, color='gray')
ax.text(1, 0.53, "← Phase 1", fontsize=8, color='gray')
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)", fontsize=12)
ax.set_ylabel(f"Avg fidelity (all {N_TRAIN_STATES} train states)", fontsize=12)
ax.set_title("Training fidelity vs epoch (full curve, both phases)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim([0.50, 1.02]); ax.grid(True, alpha=0.3)

# ── (0,1) Training loss vs global epoch ──
ax = axes[0, 1]
for (nl, res), col, ls in zip(sweep_results.items(), colors, ls_ops):
    ax.plot(res['hist_epoch'], res['loss_hist'], color=col, linestyle=ls, lw=2, label=f"L={nl}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)", fontsize=12)
ax.set_ylabel("Training loss", fontsize=12)
ax.set_title(f"Training loss vs epoch\n"
             f"weights S={W_STAB}, log={W_LOGICAL}, YL={W_YL}",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.3)

# ── (1,0) Test fidelity bar chart (full channel) ──
ax = axes[1, 0]
x = np.arange(len(LAYER_SWEEP)); bw = 0.22
fn_m = [sweep_results[nl]['f_noisy_full'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep_results[nl]['f_std_full'].mean()   for nl in LAYER_SWEEP]
fq_m = [sweep_results[nl]['f_qae_full'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep_results[nl]['f_qae_full'].std()    for nl in LAYER_SWEEP]
ax.bar(x-bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x,    fs_m, bw, label="Standard QEC",  color='#3498db', alpha=0.8)
bars = ax.bar(x+bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
ax.errorbar(x+bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4, lw=1.5)
for bar, val in zip(bars, fq_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP], fontsize=11)
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test states)", fontsize=12)
ax.set_title("Test fidelity vs layer count (full bit-flip channel)",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim([0.50, 1.02]); ax.grid(True, alpha=0.3, axis='y')

# ── (1,1) Single bit-flip QAE fidelity vs layer count ──
ax = axes[1, 1]
fqs_m = [sweep_results[nl]['f_qae_single'].mean() for nl in LAYER_SWEEP]
fqs_s = [sweep_results[nl]['f_qae_single'].std()  for nl in LAYER_SWEEP]
bars = ax.bar(x, fqs_m, 0.5, color='#2ecc71', alpha=0.85, label="QAE (1-flip)")
ax.errorbar(x, fqs_m, yerr=fqs_s, fmt='none', color='black', capsize=4, lw=1.5)
for bar, val in zip(bars, fqs_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
            f"{val:.4f}", ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.axhline(1.0, color='#3498db', ls='--', lw=1.5, label="Std QEC (1.0)")
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP], fontsize=11)
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test states)", fontsize=12)
ax.set_title("Single bit-flip fidelity vs layer count\n(correctable errors)",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.set_ylim([0.80, 1.02]); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(
    f"Spectral loss + QAE — [[3,1,3]] Bit-Flip Layer Sweep  |  p={P_NOISE}  "
    f"|  JAX {jax.default_backend().upper()}\n"
    f"Ansatz: [Ry+Rz] / CRX+CRY+CRZ (all-to-all)  |  "
    f"{N_TRAIN_STATES} train / {N_TEST_STATES} test states  |  "
    f"Phase 1 ({INIT_EPOCHS}) + Phase 2 ({MAIN_EPOCHS})  |  "
    f"layerwise warm-start={USE_LAYERWISE}",
    fontsize=11, fontweight='bold', y=1.02)

plt.tight_layout()
fig.savefig(f"bitflip_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches="tight")
print(f"  Saved : bitflip_fidelity_{FILE_TAG}.png")

# ============================================================
# FINAL SUMMARY TABLE 
# ============================================================
print("\n" + "=" * 80)
print(f"  FINAL SUMMARY — {N_TEST_STATES} test states  |  p={P_NOISE}  |  Spectral loss + QAE")
print("=" * 80)
hdr = (f"  {'L':>2}  {'Params':>7}  "
       f"{'F_noisy':>9}  {'F_stdQEC':>9}  {'F_QAE':>8}  {'±std':>7}  "
       f"{'F_QAE_1flip':>11}  {'Gap%':>6}")
print(hdr)
print("  " + "─" * (len(hdr) - 2))
for nl in LAYER_SWEEP:
    res = sweep_results[nl]
    fn = res['f_noisy_full']; fs = res['f_std_full']; fq = res['f_qae_full']
    gap = fs.mean() - fn.mean()
    pct = (fq.mean() - fn.mean()) / gap * 100 if gap > 1e-6 else 0.0
    print(f"  {nl:>2}  {res['n_params']:>7}  "
          f"{fn.mean():>9.5f}  {fs.mean():>9.5f}  "
          f"{fq.mean():>8.5f}  {fq.std():>7.5f}  "
          f"{res['f_qae_single'].mean():>11.5f}  {pct:>5.1f}%")

best_nl = max(LAYER_SWEEP, key=lambda nl: sweep_results[nl]['f_qae_full'].mean())
print(f"\n  Best layer count : L={best_nl}  "
      f"F̄={sweep_results[best_nl]['f_qae_full'].mean():.5f}  "
      f"F̄_1flip={sweep_results[best_nl]['f_qae_single'].mean():.5f}")

# ============================================================
# FIDELITY vs ERROR RATE — BEST trained model
#   (best params + best layer count; trained at p=P_NOISE, evaluated
#    across a range of bit-flip rates)
# ============================================================
print("\n" + "=" * 68)
print("  FIDELITY vs ERROR RATE  (best model)")
print("=" * 68)

best_nl_p   = max(LAYER_SWEEP, key=lambda nl: sweep_results[nl]['f_qae_full'].mean())
best_params = jnp.array(sweep_results[best_nl_p]['params'])
best_comps  = make_layer_components(best_nl_p)
best_full   = best_comps['batch_eval_full']
print(f"  Best model: L={best_nl_p}  "
      f"({sweep_results[best_nl_p]['n_params']} params)  trained at p={P_NOISE}")

P_EVAL_SWEEP = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

fqae_vs_p = []; fqae_std_vs_p = []
fnoisy_vs_p = []; fstd_vs_p = []

for p in P_EVAL_SWEEP:
    rho_n_np = np.stack([bit_flip_channel(rho_clean_test_np[i], p)
                         for i in range(N_TEST_STATES)])
    rho_n_j = jnp.array(rho_n_np)
    fn, fs, fq, _, _ = best_full(best_params, rho_n_j, rho_clean_test_j)
    fn = np.asarray(fn); fs = np.asarray(fs); fq = np.asarray(fq)
    fqae_vs_p.append(fq.mean());   fqae_std_vs_p.append(fq.std())
    fnoisy_vs_p.append(fn.mean()); fstd_vs_p.append(fs.mean())
    print(f"  p={p:>4.2f}  F_QAE={fq.mean():.4f}  F_std={fs.mean():.4f}  "
          f"F_noisy={fn.mean():.4f}")

P_EVAL_SWEEP  = np.array(P_EVAL_SWEEP)
fqae_vs_p     = np.array(fqae_vs_p);   fqae_std_vs_p = np.array(fqae_std_vs_p)
fnoisy_vs_p   = np.array(fnoisy_vs_p); fstd_vs_p     = np.array(fstd_vs_p)

np.savez(f"bitflip_fidelity_vs_p_{FILE_TAG}.npz",
         best_n_layers=np.array(best_nl_p),
         best_params=np.array(sweep_results[best_nl_p]['params']),
         p_eval=P_EVAL_SWEEP,
         f_qae=fqae_vs_p, f_qae_std=fqae_std_vs_p,
         f_noisy=fnoisy_vs_p, f_std=fstd_vs_p)
with open(f"bitflip_fidelity_vs_p_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["p", "F_qae", "F_qae_std", "F_noisy", "F_stdQEC"])
    for i in range(len(P_EVAL_SWEEP)):
        w.writerow([f"{P_EVAL_SWEEP[i]:.3f}",
                    f"{fqae_vs_p[i]:.6f}", f"{fqae_std_vs_p[i]:.6f}",
                    f"{fnoisy_vs_p[i]:.6f}", f"{fstd_vs_p[i]:.6f}"])
print(f"  Saved : bitflip_fidelity_vs_p_{FILE_TAG}.npz , bitflip_fidelity_vs_p_{FILE_TAG}.csv")

fig2, ax = plt.subplots(figsize=(9, 6))
ax.plot(P_EVAL_SWEEP, fqae_vs_p, 'o-', color='#2ecc71', lw=2,
        label=f"spectral QAE (L={best_nl_p})")
ax.fill_between(P_EVAL_SWEEP, fqae_vs_p-fqae_std_vs_p,
                fqae_vs_p+fqae_std_vs_p, color='#2ecc71', alpha=0.15)
ax.plot(P_EVAL_SWEEP, fstd_vs_p,   's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(P_EVAL_SWEEP, fnoisy_vs_p, '^-',  color='#e74c3c', lw=2, label="No correction")
ax.axvline(P_NOISE, color='black', ls=':', lw=1.2, alpha=0.6)
ax.text(P_NOISE+0.005, 0.55, f"trained @ p={P_NOISE}", fontsize=8, color='black')
ax.set_xlabel("Bit-flip error rate  p", fontsize=12)
ax.set_ylabel("Mean fidelity", fontsize=12)
ax.set_title(f"Fidelity vs error rate — best model "
             f"(L={best_nl_p}, trained at p={P_NOISE})",
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_ylim([0.45, 1.02])
plt.tight_layout()
fig2.savefig(f"bitflip_fidelity_vs_p_{FILE_TAG}.png", dpi=150, bbox_inches="tight")
print(f"  Saved : bitflip_fidelity_vs_p_{FILE_TAG}.png")

print(f"\n  Saved files:")
print(f"    ./bitflip_fidelity_{FILE_TAG}.png       (4-panel layer comparison)")
print(f"    ./bitflip_results_{FILE_TAG}.npz        (params + per-epoch history + test results)")
print(f"    ./bitflip_results_{FILE_TAG}.csv        (per-layer test summary)")
print(f"    ./bitflip_history_{FILE_TAG}.csv        (long-format, every epoch, both phases)")
print(f"    ./bitflip_fidelity_vs_p_{FILE_TAG}.png  (best model fidelity vs error rate)")
print(f"    ./bitflip_fidelity_vs_p_{FILE_TAG}.csv  ./bitflip_fidelity_vs_p_{FILE_TAG}.npz")
print("\n  Done ✓")