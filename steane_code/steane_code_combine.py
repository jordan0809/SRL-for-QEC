"""
Spectral Measurement-Free QEC 
=============================================================
Steane [[7,1,3]] under COMBINED noise:
      N = AmplitudeDamping(gamma) ∘ Depolarizing(p) ∘ CorrelatedGaussianDephasing(sigma, rho, mu)

Benchmarks
  * No correction                     (saved, plotted in Fig.1 / Fig.3)
  * Standard Steane CSS decoder       (saved, plotted in Fig.1 / Fig.3)
  * Petz (transpose) map for the FULL composed channel   (all figures)
  * SRL
"""

import warnings; warnings.filterwarnings("ignore")
import os, time, gc, csv, itertools
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]  = "false"
os.environ["TF_GPU_ALLOCATOR"]               = "cuda_malloc_async"
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
from matplotlib.colors import TwoSlopeNorm
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

# ---- combined-noise training point (same values as the single-noise scripts) ----
P_NOISE     = 0.05                # depolarizing
SIGMA_THETA = 0.5                 # correlated Gaussian dephasing std
RHO_CORR    = 0.3                 # dephasing spatial correlation
MU_COHERENT = 0.3                 # coherent (non-zero mean) dephasing angle
GAMMA_DAMP  = 0.15                # amplitude damping
NOISE_ORDER_STR = "AD ∘ Depol ∘ Dephase"

N_TRAIN_STATES = 100
N_TEST_STATES  = 1000

BATCH_SIZE      = 3
CHUNK_EVAL_SIZE = 25
STD_CHUNK       = 10

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

# Fixed baseline weights (tracking/plotting only)
W_STAB_BASE = 1.0
W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

LAYER_SWEEP = [1, 2, 3, 4, 5]
USE_LAYERWISE = True
FILE_TAG = "layerwise" if USE_LAYERWISE else "multistart"
PREFIX   = "steane_combined"

# ---- parameter grid for Fig.2 (heat maps) ----
P_GRID     = [0.00, 0.02, 0.05, 0.08, 0.12]
SIGMA_GRID = [0.00, 0.20, 0.50, 0.80, 1.20]
GAMMA_GRID = [0.05, 0.15, 0.30]           # 3 rows
SWEEP_N_STATES = 200                       # subsample of the test set for the grid

PETZ_COLOR = '#f39c12'
QAE_COLOR  = '#2ecc71'
STD_COLOR  = '#3498db'
NOI_COLOR  = '#e74c3c'

print(f"""
  Steane [[7,1,3]] — COMBINED noise  ({NOISE_ORDER_STR})
  p={P_NOISE}, sigma={SIGMA_THETA}, rho_corr={RHO_CORR}, mu={MU_COHERENT}, gamma={GAMMA_DAMP}
  N_TOTAL={N_TOTAL} (7 code + 1 anc), Layer sweep {LAYER_SWEEP}, layerwise={USE_LAYERWISE}
  Train {N_TRAIN_STATES} / Test {N_TEST_STATES}, batch {BATCH_SIZE}, LR {LR}
  Phase 1 {N_INIT_ROUNDS}x{INIT_EPOCHS} + Phase 2 {MAIN_EPOCHS}  (total {TOTAL_EPOCHS})
  Grid: {len(P_GRID)}x{len(SIGMA_GRID)}x{len(GAMMA_GRID)} = {len(P_GRID)*len(SIGMA_GRID)*len(GAMMA_GRID)} points on {SWEEP_N_STATES} states
""")

# ================================================================
# 2. Pauli primitives + Steane operators (weight-3 logicals)
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)
P1_ = np.array([[0, 0], [0, 1]], dtype=complex)      # |1><1|
SM_ = np.array([[0, 1], [0, 0]], dtype=complex)      # |0><1| = sigma_minus

def kron_list(ops): return reduce(np.kron, ops)

def op_on(qubits, mat, n=N_CODE):
    ops = [I2]*n
    for q in qubits: ops[q] = mat
    return kron_list(ops)

S_mats = [op_on([0,2,4,6], X_), op_on([1,2,5,6], X_), op_on([3,4,5,6], X_),
          op_on([0,2,4,6], Z_), op_on([1,2,5,6], Z_), op_on([3,4,5,6], Z_)]

LOGICAL_SUPPORT = [0, 1, 2]
XL_mat = op_on(LOGICAL_SUPPORT, X_)
ZL_mat = op_on(LOGICAL_SUPPORT, Z_)
YL_mat = 1j * XL_mat @ ZL_mat

N_STAB = len(S_mats)
S_STACK = jnp.array(np.stack(S_mats))
L_STACK = jnp.array(np.stack([ZL_mat, XL_mat, YL_mat]))   # [zl, xl, yl]

def expect_stack(op_stack, rho):
    return jnp.real(jnp.einsum('kij,ji->k', op_stack, rho))

# logical basis
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
# 3. COMBINED noise channel  (dephase -> depol -> amp.damping)
# ================================================================
# 3a. correlated coherent Gaussian dephasing (exact, elementwise PHI)
_zvecs = np.array([[1 - 2*((b >> (N_CODE-1-i)) & 1) for i in range(N_CODE)]
                   for b in range(SYS_DIM)], dtype=float)

def build_phi(sigma, rho_corr=RHO_CORR, mu=MU_COHERENT):
    M = np.eye(N_CODE)*(1-rho_corr) + rho_corr*np.ones((N_CODE, N_CODE))
    Sigma = (sigma**2)*M
    mu_vec = np.full(N_CODE, mu)
    C = (_zvecs[:, None, :] - _zvecs[None, :, :]) / 2.0
    lin  = C @ mu_vec
    quad = np.einsum('abi,ij,abj->ab', C, Sigma, C)
    return np.exp(-1j*lin - 0.5*quad)

PHI_TRAIN = build_phi(SIGMA_THETA)
print(f"  max|Im(PHI)| = {float(np.max(np.abs(PHI_TRAIN.imag))):.4e} "
      f"({'coherent / non-diag PTM' if np.max(np.abs(PHI_TRAIN.imag))>1e-12 else 'diagonal'})")

# 3b. pre-embedded operators
_PAULI_EMB = [(jnp.array(op_on([q], X_)),
               jnp.array(op_on([q], Y_)),
               jnp.array(op_on([q], Z_))) for q in range(N_CODE)]
_P1_EMB = [jnp.array(op_on([q], P1_)) for q in range(N_CODE)]
_SM_EMB = [jnp.array(op_on([q], SM_)) for q in range(N_CODE)]
_I_SYS  = jnp.eye(SYS_DIM, dtype=complex)

@jax.jit
def combined_channel(rho, p, gamma, phi):
    """N(rho) = AD_gamma( Depol_p( Deph_phi(rho) ) )   (fixed order)"""
    # (1) correlated coherent dephasing — elementwise
    rho = phi * rho
    # (2) single-qubit depolarizing on every code qubit
    for q in range(N_CODE):
        acc = (1.0 - p) * rho
        for E in _PAULI_EMB[q]:
            acc = acc + (p / 3.0) * (E @ rho @ E)
        rho = acc
    # (3) single-qubit amplitude damping on every code qubit
    s = jnp.sqrt(1.0 - gamma); sg = jnp.sqrt(gamma)
    for q in range(N_CODE):
        K0 = _I_SYS - (1.0 - s) * _P1_EMB[q]
        K1 = sg * _SM_EMB[q]
        rho = K0 @ rho @ jnp.conj(K0.T) + K1 @ rho @ jnp.conj(K1.T)
    return rho

@jax.jit
def combined_adjoint(A, p, gamma, phi_conj):
    """N^dagger(A) = Deph^dag( Depol^dag( AD^dag(A) ) )  (reverse order)"""
    s = jnp.sqrt(1.0 - gamma); sg = jnp.sqrt(gamma)
    for q in range(N_CODE):
        K0 = _I_SYS - (1.0 - s) * _P1_EMB[q]
        K1 = sg * _SM_EMB[q]
        A = jnp.conj(K0.T) @ A @ K0 + jnp.conj(K1.T) @ A @ K1
    for q in range(N_CODE):                      # depolarizing is self-adjoint
        acc = (1.0 - p) * A
        for E in _PAULI_EMB[q]:
            acc = acc + (p / 3.0) * (E @ A @ E)
        A = acc
    return phi_conj * A                          # Deph^dag: conj(PHI) ⊙ A

@jax.jit
def _chan_batch(rho_b, p, gamma, phi):
    return jax.vmap(lambda r: combined_channel(r, p, gamma, phi))(rho_b)

def make_noisy_stack(clean_np, p=P_NOISE, gamma=GAMMA_DAMP, phi=PHI_TRAIN,
                     chunk=CHUNK_EVAL_SIZE):
    pj, gj, phij = jnp.float64(p), jnp.float64(gamma), jnp.array(phi)
    n = clean_np.shape[0]; out = np.empty_like(clean_np)
    for k in range(int(np.ceil(n / chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        out[sl] = np.array(_chan_batch(jnp.array(clean_np[sl]), pj, gj, phij))
    return out

# sanity: trace preservation
_rt = state_to_dm(logical_state(0.7, 1.1))
_tr = float(np.real(np.trace(np.array(combined_channel(
        jnp.array(_rt), jnp.float64(P_NOISE), jnp.float64(GAMMA_DAMP), jnp.array(PHI_TRAIN))))))
print(f"  ✓ combined channel trace check: tr N(rho) = {_tr:.12f}")

# ================================================================
# 4. Standard Steane recovery (complete CSS look-up decoder, 64 syndromes)
# ================================================================
def _commutes(E, S): return np.allclose(E @ S - S @ E, 0)

X_STAB_IDX = [0, 1, 2]
Z_STAB_IDX = [3, 4, 5]

_xsyn_to_Xcorr = {(1, 1, 1): np.eye(SYS_DIM, dtype=complex)}
for q in range(N_CODE):
    E = op_on([q], X_)
    _xsyn_to_Xcorr[tuple(1 if _commutes(E, S_mats[j]) else -1 for j in Z_STAB_IDX)] = E
_zsyn_to_Zcorr = {(1, 1, 1): np.eye(SYS_DIM, dtype=complex)}
for q in range(N_CODE):
    E = op_on([q], Z_)
    _zsyn_to_Zcorr[tuple(1 if _commutes(E, S_mats[j]) else -1 for j in X_STAB_IDX)] = E
assert len(_xsyn_to_Xcorr) == 8 and len(_zsyn_to_Zcorr) == 8

_CORR_M = []
for signs in itertools.product([1, -1], repeat=N_STAB):
    Pi = np.eye(SYS_DIM, dtype=complex)
    for s, S in zip(signs, S_mats):
        Pi = Pi @ ((np.eye(SYS_DIM) + s*S)/2)
    zsyn = tuple(signs[j] for j in X_STAB_IDX)
    xsyn = tuple(signs[j] for j in Z_STAB_IDX)
    _CORR_M.append(_xsyn_to_Xcorr[xsyn] @ _zsyn_to_Zcorr[zsyn] @ Pi)
CORR_M = jnp.array(np.stack(_CORR_M))
print(f"  ✓ Steane CSS decoder built ({CORR_M.shape[0]} correction terms)")

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
# 4b. Petz (transpose) map for the FULL composed channel
#     R(rho) = sig^{1/2} N^dag( N(sig)^{-1/2} rho N(sig)^{-1/2} ) sig^{1/2}
#     sigma = maximally mixed state on the 2-dim code space
# ================================================================
_P_CODE = _P
_TR_P   = float(np.real(np.trace(_P_CODE)))          # = 2

def make_petz_fid(p, gamma, phi, chunk=STD_CHUNK):
    sig      = _P_CODE / _TR_P
    sig_half = _P_CODE / np.sqrt(_TR_P)              # (P/2)^{1/2}
    Nsig = np.array(combined_channel(jnp.array(sig), jnp.float64(p),
                                     jnp.float64(gamma), jnp.array(phi)))
    Nsig = (Nsig + Nsig.conj().T) / 2.0
    ev, evec = np.linalg.eigh(Nsig)
    tol = 1e-12 * float(ev.max())
    inv_sqrt = np.where(ev > tol, 1.0/np.sqrt(np.where(ev > tol, ev, 1.0)), 0.0)
    Nih = jnp.array((evec * inv_sqrt) @ evec.conj().T)
    Sh  = jnp.array(sig_half)
    pj, gj, pcj = jnp.float64(p), jnp.float64(gamma), jnp.array(np.conj(phi))

    @jax.jit
    def _petz_out(rho):
        inner = Nih @ rho @ Nih
        out   = Sh @ combined_adjoint(inner, pj, gj, pcj) @ Sh
        return out / jnp.real(jnp.trace(out))        # renormalize (non-unital N)

    @jax.jit
    def _fid_batch(rn, rc):
        def per(rho, rcl):
            return jnp.real(jnp.einsum('ij,ji->', _petz_out(rho), rcl))
        return jax.vmap(per)(rn, rc)

    @jax.jit
    def _spec_batch(rn, zt, xt, yt):
        def per(rho, z, x, y):
            l = expect_stack(L_STACK, _petz_out(rho))
            return jnp.abs(z-l[0]), jnp.abs(x-l[1]), jnp.abs(y-l[2])
        return jax.vmap(per)(rn, zt, xt, yt)

    def fid(rn, rc):
        n = rn.shape[0]; out = np.empty(n)
        for k in range(int(np.ceil(n/chunk))):
            sl = slice(k*chunk, min((k+1)*chunk, n))
            out[sl] = np.array(_fid_batch(rn[sl], rc[sl]))
        return out

    def spec(rn, zt, xt, yt):
        n = rn.shape[0]; DZ=np.empty(n); DX=np.empty(n); DY=np.empty(n)
        for k in range(int(np.ceil(n/chunk))):
            sl = slice(k*chunk, min((k+1)*chunk, n))
            a, b, c = _spec_batch(rn[sl], zt[sl], xt[sl], yt[sl])
            DZ[sl]=np.array(a); DX[sl]=np.array(b); DY[sl]=np.array(c)
        return DZ, DX, DY

    return fid, spec

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

    @jax.jit
    def batch_loss_and_grad(params, rho_b, tgt_b, ws, wz, wx, wy):
        def _loss(p):
            V = encoder_fn(p); Vd = jnp.conj(V.T)
            outs = _vmapped(V, Vd, rho_b)
            def per(rho, tgt):
                s = expect_stack(S_STACK, rho)
                l = expect_stack(L_STACK, rho)
                return (ws*(-jnp.sum(s))
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

    @jax.jit
    def batch_probe_chunk(params, rn, rc, zt, xt, yt):
        out = qae_batch_chunk(params, rn)
        def per(rho, rcl, z, x, y):
            s = expect_stack(S_STACK, rho)
            l = expect_stack(L_STACK, rho)
            bl = (W_STAB_BASE*(-jnp.sum(s)) + W_ZL_BASE*(z-l[0])**2
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

def generate_dataset(n):
    angles = fibonacci_sphere(n)
    clean  = np.stack([state_to_dm(logical_state(th, ph)) for th, ph in angles])
    noisy  = make_noisy_stack(clean)
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

print(f"  Precomputing Petz recovery on {N_TEST_STATES} test states ...")
petz_fid_train_pt, petz_spec_train_pt = make_petz_fid(P_NOISE, GAMMA_DAMP, PHI_TRAIN)
f_petz_test = petz_fid_train_pt(rho_noisy_test_stack, rho_clean_test_stack)
pz_dzl, pz_dxl, pz_dyl = petz_spec_train_pt(rho_noisy_test_stack, zl_test, xl_test, yl_test)
print(f"  Petz map F = {f_petz_test.mean():.5f}")

# ================================================================
# 8. Chunked eval helpers
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
# 9. Adaptive weights + epoch
# ================================================================
def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab-STAB_SAT_LOW)/(STAB_SAT_HIGH-STAB_SAT_LOW), 0, 1))
    return (W_STAB_HIGH-(W_STAB_HIGH-W_STAB_LOW)*sat,
            W_ZL_LOW+(W_ZL_HIGH-W_ZL_LOW)*sat,
            W_XL_LOW+(W_XL_HIGH-W_XL_LOW)*sat,
            W_YL_LOW+(W_YL_HIGH-W_YL_LOW)*sat)

def step_weight(cur, tgt): return cur + float(np.clip(tgt-cur, -MAX_W_DELTA, MAX_W_DELTA))

def batch_stab_avg_chunked(bsa_fn, params, rn, chunk=CHUNK_EVAL_SIZE):
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
        lv, gv = blg(params, rho_noisy_train_stack[b], targets_train_stack[b],
                     jws, jwz, jwx, jwy)
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
    blg = comps['batch_loss_and_grad']
    bsa = comps['batch_stab_avg']; bfa = comps['batch_fidelity_arr']
    bla = comps['batch_logical_errors_arr']; bpc = comps['batch_probe_chunk']

    pn, pc = rho_noisy_train_stack, rho_clean_train_stack
    pz, px, py = targets_train_stack[:, 0], targets_train_stack[:, 1], targets_train_stack[:, 2]

    def stab_mean(p): return batch_stab_avg_chunked(bsa, p, pn)

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
                f_petz_test=f_petz_test,
                dzl_test=dzl_t, dxl_test=dxl_t, dyl_test=dyl_t, hist=hist)

# ================================================================
# 11. Layer sweep
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
# 12. Save (training point)
# ================================================================
sd = {'f_noisy_test': f_noisy_test, 'f_std_test': f_std_test, 'f_petz_test': f_petz_test,
      'petz_dzl_test': pz_dzl, 'petz_dxl_test': pz_dxl, 'petz_dyl_test': pz_dyl,
      'meta_test_thetas': test_thetas, 'meta_test_phis': test_phis,
      'meta_p': np.array(P_NOISE), 'meta_sigma': np.array(SIGMA_THETA),
      'meta_rho_corr': np.array(RHO_CORR), 'meta_mu': np.array(MU_COHERENT),
      'meta_gamma': np.array(GAMMA_DAMP), 'meta_layers': np.array(LAYER_SWEEP),
      'meta_layerwise': np.array(int(USE_LAYERWISE)),
      'meta_noise_order': np.array(NOISE_ORDER_STR),
      'best_n_layers': np.array(best_nl), 'best_params': sweep[best_nl]['params'],
      'best_f_qae': np.array(sweep[best_nl]['f_qae_test'].mean())}
for nl, res in sweep.items():
    sd[f'L{nl}_params']     = res['params']
    sd[f'L{nl}_f_qae_test'] = res['f_qae_test']
    sd[f'L{nl}_dzl_test']   = res['dzl_test']
    sd[f'L{nl}_dxl_test']   = res['dxl_test']
    sd[f'L{nl}_dyl_test']   = res['dyl_test']
    for k, v in res['hist'].items(): sd[f'L{nl}_hist_{k}'] = v
np.savez(f"{PREFIX}_results_{FILE_TAG}.npz", **sd)
print(f"  Saved : {PREFIX}_results_{FILE_TAG}.npz  (best L={best_nl})")

with open(f"{PREFIX}_history_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers","global_epoch","phase","baseline_loss","baseline_loss_std",
                "fidelity","fidelity_std","dzl","dzl_std","dxl","dxl_std","dyl","dyl_std"])
    for nl, res in sweep.items():
        h = res['hist']
        for row in zip(h['epoch'], h['phase'], h['loss'], h['loss_std'], h['fid'],
                       h['fid_std'], h['dzl'], h['dzl_std'], h['dxl'], h['dxl_std'],
                       h['dyl'], h['dyl_std']):
            w.writerow([nl, int(row[0]), int(row[1])] + [f"{v:.6f}" for v in row[2:]])
print(f"  Saved : {PREFIX}_history_{FILE_TAG}.csv")

with open(f"{PREFIX}_results_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers","n_params","F_noisy_mean","F_stdQEC_mean","F_petz_mean",
                "F_qae_mean","F_qae_std","dzl_mean","dxl_mean","dyl_mean",
                "petz_dzl_mean","petz_dxl_mean","petz_dyl_mean",
                "gap_vs_petz","gap_closed_pct_vs_std"])
    for nl, res in sweep.items():
        fn, fs, fq, fp = (res['f_noisy_test'], res['f_std_test'],
                          res['f_qae_test'], res['f_petz_test'])
        gap = fs.mean()-fn.mean(); pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
        w.writerow([nl, res['n_params'], f"{fn.mean():.5f}", f"{fs.mean():.5f}",
                    f"{fp.mean():.5f}", f"{fq.mean():.5f}", f"{fq.std():.5f}",
                    f"{res['dzl_test'].mean():.5f}", f"{res['dxl_test'].mean():.5f}",
                    f"{res['dyl_test'].mean():.5f}",
                    f"{pz_dzl.mean():.5f}", f"{pz_dxl.mean():.5f}", f"{pz_dyl.mean():.5f}",
                    f"{fq.mean()-fp.mean():+.5f}", f"{pct:.1f}"])
print(f"  Saved : {PREFIX}_results_{FILE_TAG}.csv")

# ================================================================
# 13. FIGURE 1 — 4-panel layer sweep at the training noise point
# ================================================================
colors = ['#e41a1c','#ff7f00','#4daf4a','#377eb8','#984ea3']; ls_ops = ['-','--','-','--','-']
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax = axes[0, 0]
for (nl, res), c, ls in zip(sweep.items(), colors, ls_ops):
    h = res['hist']
    ax.plot(h['epoch'], h['fid'], color=c, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(h['epoch'], h['fid']-h['fid_std'], h['fid']+h['fid_std'], color=c, alpha=0.12)
ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5, label=f"Std Steane QEC {f_std_test.mean():.4f}")
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
fs_m = [sweep[nl]['f_std_test'].mean()   for nl in LAYER_SWEEP]
fp_m = [sweep[nl]['f_petz_test'].mean()  for nl in LAYER_SWEEP]
fq_m = [sweep[nl]['f_qae_test'].mean()   for nl in LAYER_SWEEP]
fq_s = [sweep[nl]['f_qae_test'].std()    for nl in LAYER_SWEEP]
ax.bar(x-1.5*bw, fn_m, bw, label="No correction", color=NOI_COLOR, alpha=0.8)
ax.bar(x-0.5*bw, fs_m, bw, label="Standard Steane QEC", color=STD_COLOR, alpha=0.8)
ax.bar(x+0.5*bw, fp_m, bw, label="Petz map (full channel)", color=PETZ_COLOR, alpha=0.8)
bars = ax.bar(x+1.5*bw, fq_m, bw, label="Spectral QAE", color=QAE_COLOR, alpha=0.8)
ax.errorbar(x+1.5*bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4)
for b, v in zip(bars, fq_m):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003, f"{v:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test)")
ax.set_title("Test fidelity vs layer count", fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]; bw2 = 0.25
dz = [sweep[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dx = [sweep[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dy = [sweep[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
ax.bar(x-bw2, dz, bw2, label="ΔZL", color=NOI_COLOR, alpha=0.85)
ax.bar(x,     dx, bw2, label="ΔXL", color=STD_COLOR, alpha=0.85)
ax.bar(x+bw2, dy, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
ax.axhline(pz_dzl.mean(), color=PETZ_COLOR, ls='-.', lw=1.2, label="Petz ΔZL")
ax.axhline(pz_dxl.mean(), color=PETZ_COLOR, ls='--', lw=1.0, label="Petz ΔXL")
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel("Mean |target − recovered|")
ax.set_title("Logical spectrum error vs layer count", fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Spectral QAE — Steane [[7,1,3]] COMBINED noise  ({NOISE_ORDER_STR}) | "
             f"p={P_NOISE}, σ={SIGMA_THETA}, ρ={RHO_CORR}, μ={MU_COHERENT}, γ={GAMMA_DAMP} | "
             f"JAX {jax.default_backend().upper()} | layerwise={USE_LAYERWISE}",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout(); fig.savefig(f"{PREFIX}_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved : {PREFIX}_fidelity_{FILE_TAG}.png")

# ================================================================
# 14. Summary table
# ================================================================
print("\n" + "="*92 + f"\n  FINAL SUMMARY — {N_TEST_STATES} test | p={P_NOISE}, σ={SIGMA_THETA}, "
      f"ρ={RHO_CORR}, μ={MU_COHERENT}, γ={GAMMA_DAMP}\n" + "="*92)
print(f"  {'L':>2} {'Params':>7} {'F_noisy':>9} {'F_std':>9} {'F_petz':>9} {'F_QAE':>8} "
      f"{'±std':>7} {'ΔZL':>7} {'ΔXL':>7} {'ΔYL':>7} {'QAE-Petz':>9}")
for nl in LAYER_SWEEP:
    r = sweep[nl]; fn, fs, fq, fp = (r['f_noisy_test'], r['f_std_test'],
                                     r['f_qae_test'], r['f_petz_test'])
    print(f"  {nl:>2} {r['n_params']:>7} {fn.mean():>9.5f} {fs.mean():>9.5f} {fp.mean():>9.5f} "
          f"{fq.mean():>8.5f} {fq.std():>7.5f} {r['dzl_test'].mean():>7.4f} "
          f"{r['dxl_test'].mean():>7.4f} {r['dyl_test'].mean():>7.4f} "
          f"{fq.mean()-fp.mean():>+9.5f}")
print(f"\n  Best : L={best_nl}  F̄={sweep[best_nl]['f_qae_test'].mean():.5f}")

# ================================================================
# 15. 3-PARAMETER GRID SWEEP (best model)
# ================================================================
print("\n" + "="*68 + "\n  3-PARAMETER GRID SWEEP  (p × σ × γ)\n" + "="*68)
best_comps  = make_layer_components(best_nl)
best_bfa    = best_comps['batch_fidelity_arr']
best_bla    = best_comps['batch_logical_errors_arr']
best_params = jnp.array(sweep[best_nl]['params'])

sub_idx = np.linspace(0, N_TEST_STATES-1, SWEEP_N_STATES).astype(int)
rc_sub_np = rho_clean_test_np[sub_idx]
rc_sub    = jnp.array(rc_sub_np)
zl_sub, xl_sub, yl_sub = zl_test[sub_idx], xl_test[sub_idx], yl_test[sub_idx]

nG, nS, nP = len(GAMMA_GRID), len(SIGMA_GRID), len(P_GRID)
G_qae   = np.zeros((nG, nS, nP)); G_qae_s = np.zeros((nG, nS, nP))
G_petz  = np.zeros((nG, nS, nP)); G_std   = np.zeros((nG, nS, nP))
G_noisy = np.zeros((nG, nS, nP))
G_dzl   = np.zeros((nG, nS, nP)); G_dxl = np.zeros((nG, nS, nP)); G_dyl = np.zeros((nG, nS, nP))

t0 = time.time()
for ig, gam in enumerate(GAMMA_GRID):
    for isg, sig in enumerate(SIGMA_GRID):
        phi_s = build_phi(sig)
        for ip, pp in enumerate(P_GRID):
            rn_np = make_noisy_stack(rc_sub_np, p=pp, gamma=gam, phi=phi_s)
            rn_j  = jnp.array(rn_np)
            fq  = eval_fidelity_chunked(best_params, best_bfa, rn_j, rc_sub)
            dz, dx, dy = eval_logical_errors_chunked(best_params, best_bla, rn_j,
                                                     zl_sub, xl_sub, yl_sub)
            fpz_fn, _ = make_petz_fid(pp, gam, phi_s)
            fpz = fpz_fn(rn_j, rc_sub)
            fsd = f_std_chunked(rn_j, rc_sub)
            fn  = np.real(np.einsum('nij,nji->n', rn_np, rc_sub_np))
            G_qae[ig, isg, ip]   = fq.mean();  G_qae_s[ig, isg, ip] = fq.std()
            G_petz[ig, isg, ip]  = fpz.mean(); G_std[ig, isg, ip]   = fsd.mean()
            G_noisy[ig, isg, ip] = fn.mean()
            G_dzl[ig, isg, ip]   = dz.mean(); G_dxl[ig, isg, ip] = dx.mean()
            G_dyl[ig, isg, ip]   = dy.mean()
            print(f"  γ={gam:4.2f} σ={sig:4.2f} p={pp:4.2f} | "
                  f"QAE={fq.mean():.4f}  Petz={fpz.mean():.4f}  "
                  f"std={fsd.mean():.4f}  none={fn.mean():.4f}")
            del rn_np, rn_j; gc.collect()
print(f"  grid sweep time : {time.time()-t0:.0f}s")

np.savez(f"{PREFIX}_grid_{FILE_TAG}.npz",
         best_n_layers=np.array(best_nl), best_params=sweep[best_nl]['params'],
         p_grid=np.array(P_GRID), sigma_grid=np.array(SIGMA_GRID),
         gamma_grid=np.array(GAMMA_GRID), rho_corr=np.array(RHO_CORR),
         mu=np.array(MU_COHERENT), n_states=np.array(SWEEP_N_STATES),
         noise_order=np.array(NOISE_ORDER_STR),
         F_qae=G_qae, F_qae_std=G_qae_s, F_petz=G_petz, F_std=G_std, F_noisy=G_noisy,
         dzl_qae=G_dzl, dxl_qae=G_dxl, dyl_qae=G_dyl)
with open(f"{PREFIX}_grid_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gamma","sigma","p","F_qae","F_qae_std","F_petz","F_stdQEC","F_noisy",
                "dzl_qae","dxl_qae","dyl_qae","F_qae_minus_F_petz"])
    for ig, gam in enumerate(GAMMA_GRID):
        for isg, sig in enumerate(SIGMA_GRID):
            for ip, pp in enumerate(P_GRID):
                w.writerow([f"{gam:.3f}", f"{sig:.3f}", f"{pp:.3f}",
                            f"{G_qae[ig,isg,ip]:.6f}", f"{G_qae_s[ig,isg,ip]:.6f}",
                            f"{G_petz[ig,isg,ip]:.6f}", f"{G_std[ig,isg,ip]:.6f}",
                            f"{G_noisy[ig,isg,ip]:.6f}", f"{G_dzl[ig,isg,ip]:.6f}",
                            f"{G_dxl[ig,isg,ip]:.6f}", f"{G_dyl[ig,isg,ip]:.6f}",
                            f"{G_qae[ig,isg,ip]-G_petz[ig,isg,ip]:+.6f}"])
print(f"  Saved : {PREFIX}_grid_{FILE_TAG}.npz / .csv")

# ---------------- FIGURE 2 — 3x3 heat-map grid ----------------
vmin = float(min(G_qae.min(), G_petz.min())); vmax = float(max(G_qae.max(), G_petz.max()))
dmax = float(np.abs(G_qae - G_petz).max()); dmax = max(dmax, 1e-6)
xt = np.arange(nP); yt = np.arange(nS)

fig, axes = plt.subplots(nG, 3, figsize=(16, 4.4*nG))
if nG == 1: axes = axes[None, :]
for ig, gam in enumerate(GAMMA_GRID):
    panels = [(G_qae[ig],  f"Spectral QAE (L={best_nl})", 'viridis', vmin, vmax, None),
              (G_petz[ig], "Petz map (full channel)",     'viridis', vmin, vmax, None),
              (G_qae[ig]-G_petz[ig], "F(QAE) − F(Petz)",  'RdBu_r',  -dmax, dmax,
               TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax))]
    for ic, (M, ttl, cmap, lo, hi, norm) in enumerate(panels):
        ax = axes[ig, ic]
        if norm is None:
            im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, vmin=lo, vmax=hi)
        else:
            im = ax.imshow(M, origin='lower', aspect='auto', cmap=cmap, norm=norm)
        ax.set_xticks(xt); ax.set_xticklabels([f"{v:g}" for v in P_GRID])
        ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in SIGMA_GRID])
        ax.set_xlabel("depolarizing p"); ax.set_ylabel("dephasing σ")
        ax.set_title(f"γ={gam:g} — {ttl}", fontsize=10, fontweight='bold')
        for a in range(nS):
            for b in range(nP):
                val = M[a, b]
                rel = (val-lo)/(hi-lo+1e-12)
                col = 'white' if (norm is None and rel < 0.5) else 'black'
                ax.text(b, a, f"{val:+.3f}" if ic == 2 else f"{val:.3f}",
                        ha='center', va='center', fontsize=7, color=col)
        # mark the training point if it lies on the grid
        if np.any(np.isclose(P_GRID, P_NOISE)) and np.any(np.isclose(SIGMA_GRID, SIGMA_THETA)) \
           and np.isclose(gam, GAMMA_DAMP):
            bx = int(np.argmin(np.abs(np.array(P_GRID)-P_NOISE)))
            by = int(np.argmin(np.abs(np.array(SIGMA_GRID)-SIGMA_THETA)))
            ax.plot(bx, by, marker='*', ms=14, mfc='none', mec='red', mew=2)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                     label="mean fidelity" if ic < 2 else "ΔF")
plt.suptitle(f"Steane [[7,1,3]] combined noise ({NOISE_ORDER_STR}) — generalization grid "
             f"| best L={best_nl} trained @ (p={P_NOISE}, σ={SIGMA_THETA}, γ={GAMMA_DAMP}) "
             f"| ρ={RHO_CORR}, μ={MU_COHERENT} | {SWEEP_N_STATES} states  (★ = training point)",
             fontsize=11, fontweight='bold', y=1.005)
plt.tight_layout()
fig.savefig(f"{PREFIX}_grid_heatmaps_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved : {PREFIX}_grid_heatmaps_{FILE_TAG}.png")

# ================================================================
# 16. FIGURE 3 — 1D cuts through the training point
# ================================================================
print("\n" + "="*68 + "\n  1D CUTS THROUGH THE TRAINING POINT\n" + "="*68)
CUT_P     = [0.0, 0.01, 0.02, 0.05, 0.08, 0.12, 0.16, 0.20]
CUT_SIGMA = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3]
CUT_GAMMA = [0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 0.50]

def cut_eval(kind, values):
    Q=[]; QS=[]; PZ=[]; SD=[]; NO=[]
    for v in values:
        pp  = v if kind == 'p'     else P_NOISE
        sig = v if kind == 'sigma' else SIGMA_THETA
        gam = v if kind == 'gamma' else GAMMA_DAMP
        phi_s = build_phi(sig)
        rn_np = make_noisy_stack(rc_sub_np, p=pp, gamma=gam, phi=phi_s)
        rn_j  = jnp.array(rn_np)
        fq  = eval_fidelity_chunked(best_params, best_bfa, rn_j, rc_sub)
        fpz = make_petz_fid(pp, gam, phi_s)[0](rn_j, rc_sub)
        fsd = f_std_chunked(rn_j, rc_sub)
        fn  = np.real(np.einsum('nij,nji->n', rn_np, rc_sub_np))
        Q.append(fq.mean()); QS.append(fq.std()); PZ.append(fpz.mean())
        SD.append(fsd.mean()); NO.append(fn.mean())
        print(f"  cut {kind}={v:5.3f} | QAE={fq.mean():.4f} Petz={fpz.mean():.4f} "
              f"std={fsd.mean():.4f} none={fn.mean():.4f}")
        del rn_np, rn_j; gc.collect()
    return (np.array(Q), np.array(QS), np.array(PZ), np.array(SD), np.array(NO))

cuts = {'p': (CUT_P, cut_eval('p', CUT_P)),
        'sigma': (CUT_SIGMA, cut_eval('sigma', CUT_SIGMA)),
        'gamma': (CUT_GAMMA, cut_eval('gamma', CUT_GAMMA))}

np.savez(f"{PREFIX}_cuts_{FILE_TAG}.npz",
         **{f"{k}_values": np.array(v[0]) for k, v in cuts.items()},
         **{f"{k}_{n}": v[1][i] for k, v in cuts.items()
            for i, n in enumerate(['F_qae','F_qae_std','F_petz','F_std','F_noisy'])},
         best_n_layers=np.array(best_nl))
with open(f"{PREFIX}_cuts_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cut_axis","value","F_qae","F_qae_std","F_petz","F_stdQEC","F_noisy"])
    for k, (vals, (Q, QS, PZ, SD, NO)) in cuts.items():
        for i, v in enumerate(vals):
            w.writerow([k, f"{v:.4f}", f"{Q[i]:.6f}", f"{QS[i]:.6f}",
                        f"{PZ[i]:.6f}", f"{SD[i]:.6f}", f"{NO[i]:.6f}"])
print(f"  Saved : {PREFIX}_cuts_{FILE_TAG}.npz / .csv")

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
labels = {'p': ("depolarizing p", P_NOISE),
          'sigma': ("dephasing σ", SIGMA_THETA),
          'gamma': ("amplitude damping γ", GAMMA_DAMP)}
for ax, (k, (vals, (Q, QS, PZ, SD, NO))) in zip(axes, cuts.items()):
    ax.plot(vals, Q, 'o-', color=QAE_COLOR, lw=2, label=f"Spectral QAE (L={best_nl})")
    ax.fill_between(vals, Q-QS, Q+QS, color=QAE_COLOR, alpha=0.15)
    ax.plot(vals, PZ, 'd-.', color=PETZ_COLOR, lw=2, label="Petz map")
    ax.plot(vals, SD, 's--', color=STD_COLOR, lw=1.6, alpha=0.9, label="Standard Steane QEC")
    ax.plot(vals, NO, '^:',  color=NOI_COLOR, lw=1.4, alpha=0.8, label="No correction")
    xl, tp = labels[k]
    ax.axvline(tp, color='black', ls=':', alpha=0.6)
    lo, hi = ax.get_ylim(); ax.text(tp, lo+0.03*(hi-lo), f"  trained @ {tp}", fontsize=8)
    ax.set_xlabel(xl); ax.set_ylabel("Mean fidelity")
    ax.set_title(f"vary {xl}  (others fixed at training point)", fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
plt.suptitle(f"Steane [[7,1,3]] combined noise ({NOISE_ORDER_STR}) — 1D cuts, best model L={best_nl}"
             f" | ρ={RHO_CORR}, μ={MU_COHERENT} | {SWEEP_N_STATES} states",
             fontsize=11, fontweight='bold', y=1.03)
plt.tight_layout()
fig.savefig(f"{PREFIX}_cuts_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved : {PREFIX}_cuts_{FILE_TAG}.png")

print("\n  Done ✓")