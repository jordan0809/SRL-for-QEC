"""
Spectral Measurement-Free QEC 
=============================================================
Phase-flip [[3,1,3]] repetition code under CORRELATED GAUSSIAN dephasing.

Noise (EXACT closed form, no quadrature):
  Each qubit gets U_i = exp(-i theta_i/2 Z_i), theta ~ N(mu, Sigma),
  Sigma_ii = sigma^2,  Sigma_ij = rho_corr * sigma^2  (i != j).
  Averaging Gaussian random Z-rotations => computational-basis (de)phasing:
      rho_noisy[a,b] = Phi[a,b] * rho[a,b],
      Phi[a,b] = exp(-i mu (sum c) - 1/2 c^T Sigma c),  c_i = (z^a_i - z^b_i)/2, z_i = ±1.
  mu = 0  => real Phi  => diagonal (pure dephasing) PTM.
  mu != 0 => complex Phi => NON-diagonal (coherent) PTM with X<->Y mixing.

Phase-flip code (Hadamard-dual of bit-flip):
  S1 = X1X2,  S2 = X2X3 ;  XL = XXX, ZL = ZZZ, YL = YYY ;
  |0L> = |+++>,  |1L> = |--->.
  Standard recovery: syndrome (s1,s2) -> {I, Z1, Z2, Z3}.

Petz map addition:
  Theoretical Petz (transpose) recovery R_{sigma,N} benchmark, evaluated on the
  test set and across the sigma sweep; added to all CSV/npz/plots.

Outputs:
  dephasing_results_<tag>.npz / .csv
  dephasing_history_<tag>.csv
  dephasing_ptm_<tag>.png                    (single-qubit PTM mu=0 vs mu)
  dephasing_fidelity_<tag>.png
  dephasing_fidelity_vs_sigma_<tag>.png / .csv / .npz
"""

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

# ================================================================
# 1. Global constants
# ================================================================
N_CODE  = 3
N_ANC   = 1
N_TOTAL = N_CODE + N_ANC          # 4
SYS_DIM = 2 ** N_CODE             # 8
ANC_DIM = 2 ** N_ANC             # 2
TOT_DIM = 2 ** N_TOTAL            # 16

SIGMA_THETA = 0.5                 # per-qubit std of the random Z-rotation angle
RHO_CORR    = 0.3                 # correlation across the 3 qubits (0=indep, 1=collective)
MU_COHERENT = 0.3                 # mean rotation angle

N_TRAIN_STATES = 100
N_TEST_STATES  = 1000
BATCH_SIZE     = 5
CHUNK_EVAL_SIZE = 200

LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
TOTAL_EPOCHS       = INIT_EPOCHS + MAIN_EPOCHS   # 200
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

PROBE_N = N_TRAIN_STATES

# ---- plotting colors ----
PETZ_COLOR = '#f39c12'            # <-- Petz-map curve/bar/line color

r_equiv = (1 - np.exp(-SIGMA_THETA**2 / 2)) / 2
print(f"""
  Phase-flip [[3,1,3]] code — correlated COHERENT dephasing (exact closed form)
  sigma={SIGMA_THETA}, rho_corr={RHO_CORR}, mu={MU_COHERENT}  (equiv single-qubit r = {r_equiv:.4f})
  PTM type: {"NON-diagonal (coherent)" if MU_COHERENT != 0 else "diagonal (dephasing)"}
  Layer sweep {LAYER_SWEEP}, USE_LAYERWISE={USE_LAYERWISE}
  Train {N_TRAIN_STATES} / Test {N_TEST_STATES}, batch {BATCH_SIZE}, LR {LR}
  Phase 1: {N_INIT_ROUNDS}x{INIT_EPOCHS}  Phase 2: {MAIN_EPOCHS}  (total {TOTAL_EPOCHS})
""")

# ================================================================
# 2. Pauli primitives + phase-flip code operators
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops): return reduce(np.kron, ops)

S1_mat = kron_list([X_, X_, I2]); S2_mat = kron_list([I2, X_, X_])
XL_mat = kron_list([X_, X_, X_])
ZL_mat = kron_list([Z_, Z_, Z_])
YL_mat = kron_list([Y_, Y_, Y_])

S1j = jnp.array(S1_mat); S2j = jnp.array(S2_mat)
ZLj = jnp.array(ZL_mat); XLj = jnp.array(XL_mat); YLj = jnp.array(YL_mat)

ket_p = np.array([1, 1],  dtype=complex) / np.sqrt(2)
ket_m = np.array([1, -1], dtype=complex) / np.sqrt(2)
ket_0L = kron_list([ket_p, ket_p, ket_p]).flatten()
ket_1L = kron_list([ket_m, ket_m, ket_m]).flatten()

def logical_state(theta, phi):
    return np.cos(theta/2)*ket_0L + np.exp(1j*phi)*np.sin(theta/2)*ket_1L

def state_to_dm(psi):
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T

ANC_0     = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0, 0].set(1.0)
FRESH_000 = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0, 0].set(1.0)

# code projector P = (I+S1)/2 (I+S2)/2 ;  tr(P)=2 (2-dim code space)
I8 = np.eye(SYS_DIM, dtype=complex)
P_CODE = ((I8 + S1_mat)/2) @ ((I8 + S2_mat)/2)
TR_P   = float(np.real(np.trace(P_CODE)))          # = 2

# ================================================================
# 3. EXACT correlated coherent-dephasing channel  (elementwise Phi)
# ================================================================
_zvecs = np.array([[1 - 2*((b >> (N_CODE-1-i)) & 1) for i in range(N_CODE)]
                   for b in range(SYS_DIM)], dtype=float)     # (8,3), entries ±1

def build_phi(sigma, rho_corr, mu=0.0):
    """Exact damping matrix. mu != 0 => non-diagonal (coherent) PTM."""
    M = np.eye(N_CODE)*(1-rho_corr) + rho_corr*np.ones((N_CODE, N_CODE))
    Sigma = (sigma**2)*M
    mu_vec = np.full(N_CODE, mu)
    Phi = np.empty((SYS_DIM, SYS_DIM), dtype=complex)
    for a in range(SYS_DIM):
        for b in range(SYS_DIM):
            c = (_zvecs[a] - _zvecs[b]) / 2.0
            Phi[a, b] = np.exp(-1j*(mu_vec @ c) - 0.5*(c @ Sigma @ c))
    return Phi

PHI = build_phi(SIGMA_THETA, RHO_CORR, mu=MU_COHERENT)

_maxim = float(np.max(np.abs(PHI.imag)))
print(f"  max|Im(PHI)| = {_maxim:.4e}  "
      f"({'>0 => NON-diagonal / coherent' if _maxim > 1e-12 else '0 => diagonal / dephasing'})")

def dephasing_channel(rho, phi=PHI):
    """Exact coherent-dephasing channel: elementwise damping."""
    return phi * rho

# --- single-qubit PTM visualisation: shows X<->Y mixing for mu!=0 ---
def single_qubit_ptm(sigma, mu):
    c = np.exp(-0.5*sigma**2)*np.cos(mu)
    s = np.exp(-0.5*sigma**2)*np.sin(mu)
    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s,  c, 0],
                     [0, 0,  0, 1]])

fig0, axp = plt.subplots(1, 2, figsize=(10, 4))
for a_, m_, ttl in [(axp[0], 0.0, "μ=0  (diagonal)"),
                    (axp[1], MU_COHERENT, f"μ={MU_COHERENT}  (non-diagonal)")]:
    K = single_qubit_ptm(SIGMA_THETA, m_)
    im = a_.imshow(K, cmap="RdBu", vmin=-1, vmax=1)
    a_.set_xticks(range(4)); a_.set_xticklabels(["I", "X", "Y", "Z"])
    a_.set_yticks(range(4)); a_.set_yticklabels(["I", "X", "Y", "Z"])
    a_.set_title(ttl, fontweight="bold")
    for i in range(4):
        for j in range(4):
            a_.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig0.colorbar(im, ax=axp, fraction=0.046)
fig0.suptitle(f"Single-qubit PTM (σ={SIGMA_THETA}): X↔Y mixing appears only for μ≠0",
              fontweight="bold")
fig0.savefig(f"dephasing_ptm_{FILE_TAG}.png", dpi=150, bbox_inches="tight")
plt.close(fig0)
print(f"  Saved : dephasing_ptm_{FILE_TAG}.png")

# ================================================================
# 4. Standard QEC for the phase-flip code (numpy + JAX)
# ================================================================
Z0 = kron_list([Z_, I2, I2]); Z1 = kron_list([I2, Z_, I2]); Z2 = kron_list([I2, I2, Z_])

def _proj(s1, s2):
    return ((I8 + s1*S1_mat)/2) @ ((I8 + s2*S2_mat)/2)

# syndrome -> correction:  (+,+):I  (-,+):Z1  (-,-):Z2  (+,-):Z3
_STD_CORR = [(_proj(+1, +1), I8),
             (_proj(-1, +1), Z0),
             (_proj(-1, -1), Z1),
             (_proj(+1, -1), Z2)]

def standard_qec_phaseflip(rho):
    out = np.zeros_like(rho)
    for Pi, R in _STD_CORR:
        out += R @ Pi @ rho @ Pi @ R
    return out

CORR_PI = jnp.array(np.stack([Pi for Pi, _ in _STD_CORR]))
CORR_R  = jnp.array(np.stack([R  for _, R in _STD_CORR]))

def std_qec_jax(rho):
    def one(Pi, R):
        return R @ Pi @ rho @ Pi @ R
    return jnp.sum(jax.vmap(one)(CORR_PI, CORR_R), axis=0)

# ================================================================
# 4b. Petz (transpose) recovery map — theoretical benchmark
# ================================================================
# Reference sigma = maximally mixed state on the 2-dim code space = P/2.
# N(rho) = PHI ⊙ rho  (coherent dephasing);  N†(A) = conj(PHI) ⊙ A.
# R(rho) = sigma^{1/2} N†( N(sigma)^{-1/2} rho N(sigma)^{-1/2} ) sigma^{1/2}
def make_petz_fid(phi, chunk=CHUNK_EVAL_SIZE):
    """Build the Petz recovery map for noise channel N(rho)=phi⊙rho and
    return a chunked fidelity evaluator F = tr(R(N(rho_clean)) rho_clean)."""
    phi = np.asarray(phi)
    sig      = P_CODE / TR_P                       # sigma = P/2
    sig_half = P_CODE / np.sqrt(TR_P)              # (P/2)^{1/2} = P/sqrt(2)
    Nsig = phi * sig
    Nsig = (Nsig + Nsig.conj().T) / 2.0            # hermitize
    ev, evec = np.linalg.eigh(Nsig)
    tol = 1e-12 * float(ev.max())
    inv_sqrt = np.where(ev > tol, 1.0/np.sqrt(np.where(ev > tol, ev, 1.0)), 0.0)
    Nsig_inv_half = (evec * inv_sqrt) @ evec.conj().T
    phi_conj = np.conj(phi)

    Nih = jnp.array(Nsig_inv_half); Sh = jnp.array(sig_half); Pc = jnp.array(phi_conj)

    @jax.jit
    def _petz_fid_batch(rn, rc):
        def per(rho, rcl):
            inner = Nih @ rho @ Nih
            out   = Sh @ (Pc * inner) @ Sh
            out   = out / jnp.real(jnp.trace(out))          # renormalize (CPTP recovery)
            return jnp.real(jnp.einsum('ij,ji->', out, rcl))
        return jax.vmap(per)(rn, rc)

    def evalf(rn, rc):
        n = rn.shape[0]; out = np.empty(n)
        for k in range(int(np.ceil(n/chunk))):
            sl = slice(k*chunk, min((k+1)*chunk, n))
            out[sl] = np.array(_petz_fid_batch(rn[sl], rc[sl]))
        return out
    return evalf

# ================================================================
# 5. Model B ansatz  (identical to the reference scripts)
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

# ================================================================
# 6. Partial-trace helpers
# ================================================================
def partial_trace_system(rho_16):
    r = rho_16.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=0, axis2=2)

def partial_trace_ancilla(rho_16):
    r = rho_16.reshape(SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=1, axis2=3)

# ================================================================
# 7. Per-layer factory
# ================================================================
def make_layer_components(n_layers):
    n_p = count_params(n_layers)
    def circuit(params): model_b_ansatz(params, n_layers)
    encoder_fn = qml.matrix(circuit, wire_order=range(N_TOTAL))

    def qae_single(params, rho_code):
        V  = encoder_fn(params); Vd = jnp.conj(V.T)
        rho = jnp.kron(rho_code.astype(complex), ANC_0)
        rho = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho = jnp.kron(FRESH_000, rho_a)
        rho = Vd @ rho @ V
        return partial_trace_ancilla(rho)

    def _fwd_with_V(V, Vd, rho_code):
        rho = jnp.kron(rho_code, ANC_0)
        rho = V @ rho @ Vd
        rho_a = partial_trace_system(rho)
        rho = jnp.kron(FRESH_000, rho_a)
        rho = Vd @ rho @ V
        return partial_trace_ancilla(rho)

    _vmapped = jax.vmap(_fwd_with_V, in_axes=(None, None, 0))

    @jax.jit
    def qae_batch_chunk(params, rho_chunk):
        V  = encoder_fn(params); Vd = jnp.conj(V.T)
        return _vmapped(V, Vd, rho_chunk)

    # ── adaptive-weight per-state loss+grad (TRAINING) ──
    @jax.jit
    def loss_and_grad(params, rho_noisy, target, ws, wz, wx, wy):
        def _loss(p):
            rho = qae_single(p, rho_noisy)
            s1 = jnp.real(jnp.trace(S1j @ rho)); s2 = jnp.real(jnp.trace(S2j @ rho))
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return (ws*(-(s1+s2))
                    + wz*(target[0]-zl)**2
                    + wx*(target[1]-xl)**2
                    + wy*(target[2]-yl)**2)
        return jax.value_and_grad(_loss)(params)

    @jax.jit
    def batch_stab_avg(params, rho_chunk):
        out = qae_batch_chunk(params, rho_chunk)
        def per(r):
            return (jnp.real(jnp.trace(S1j @ r)) + jnp.real(jnp.trace(S2j @ r))) / 2.0
        return jnp.mean(jax.vmap(per)(out))

    @jax.jit
    def batch_fidelity_arr(params, rho_noisy_chunk, rho_clean_chunk):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(ro, rc): return jnp.real(jnp.trace(ro @ rc))
        return jax.vmap(per)(out, rho_clean_chunk)

    @jax.jit
    def batch_logical_errors_arr(params, rho_noisy_chunk, zt, xt, yt):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(rho, z, x, y):
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return jnp.abs(z-zl), jnp.abs(x-xl), jnp.abs(y-yl)
        return jax.vmap(per)(out, zt, xt, yt)

    @jax.jit
    def batch_baseline_loss_arr(params, rho_noisy_chunk, zt, xt, yt):
        out = qae_batch_chunk(params, rho_noisy_chunk)
        def per(rho, z, x, y):
            s1 = jnp.real(jnp.trace(S1j @ rho)); s2 = jnp.real(jnp.trace(S2j @ rho))
            zl = jnp.real(jnp.trace(ZLj @ rho))
            xl = jnp.real(jnp.trace(XLj @ rho))
            yl = jnp.real(jnp.trace(YLj @ rho))
            return (W_STAB_BASE*(-(s1+s2))
                    + W_ZL_BASE*(z-zl)**2 + W_XL_BASE*(x-xl)**2 + W_YL_BASE*(y-yl)**2)
        return jax.vmap(per)(out, zt, xt, yt)

    return dict(n_params=n_p,
                qae_single=qae_single,
                qae_batch_chunk=qae_batch_chunk,
                loss_and_grad=loss_and_grad,
                batch_stab_avg=batch_stab_avg,
                batch_fidelity_arr=batch_fidelity_arr,
                batch_logical_errors_arr=batch_logical_errors_arr,
                batch_baseline_loss_arr=batch_baseline_loss_arr)

# ================================================================
# 8. Datasets (built once)
# ================================================================
def fibonacci_sphere(n):
    golden = (1 + np.sqrt(5)) / 2
    return [(np.arccos(1 - 2*(i+0.5)/n), (2*np.pi*i/golden) % (2*np.pi)) for i in range(n)]

def generate_dataset(n_states, phi=PHI):
    data = []
    for theta, phi_ang in fibonacci_sphere(n_states):
        psi   = logical_state(theta, phi_ang)
        rho_c = state_to_dm(psi)
        rho_n = dephasing_channel(rho_c, phi)
        data.append(dict(
            theta=theta, phi=phi_ang, rho_clean=rho_c, rho_noisy=rho_n,
            zl=float(np.real(np.trace(ZL_mat @ rho_c))),
            xl=float(np.real(np.trace(XL_mat @ rho_c))),
            yl=float(np.real(np.trace(YL_mat @ rho_c)))))
    return data

print("  Generating training data ...")
training_data = generate_dataset(N_TRAIN_STATES)
print("  Generating test data ...")
test_data     = generate_dataset(N_TEST_STATES)

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

print(f"  Precomputing standard QEC on {N_TEST_STATES} test states ...")
rho_std_test = np.stack([standard_qec_phaseflip(d['rho_noisy']) for d in test_data])
f_std_test   = np.array([float(np.real(np.trace(rho_std_test[i] @ test_data[i]['rho_clean'])))
                         for i in range(N_TEST_STATES)])
f_noisy_test = np.array([float(np.real(np.trace(test_data[i]['rho_noisy'] @ test_data[i]['rho_clean'])))
                         for i in range(N_TEST_STATES)])
print(f"  No-correction F = {f_noisy_test.mean():.5f} | Std QEC F = {f_std_test.mean():.5f}")

print(f"  Precomputing Petz recovery on {N_TEST_STATES} test states ...")
petz_fid_fn = make_petz_fid(PHI)
f_petz_test = petz_fid_fn(rho_noisy_test_stack, rho_clean_test_stack)
print(f"  Petz map F = {f_petz_test.mean():.5f}")

# ================================================================
# 9. Chunked evaluation helpers
# ================================================================
def eval_fidelity_chunked(params, qae_fn, rho_noisy_stack, rho_clean_stack,
                          chunk=CHUNK_EVAL_SIZE):
    n = rho_noisy_stack.shape[0]; fids = np.empty(n)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        rout = np.array(qae_fn(params, rho_noisy_stack[sl]))
        rcln = np.array(rho_clean_stack[sl])
        for i, (ro, rc) in enumerate(zip(rout, rcln)):
            fids[sl.start+i] = float(np.real(np.trace(ro @ rc)))
    return fids

def eval_logical_errors_chunked(params, qae_fn, rho_noisy_stack, zt, xt, yt,
                                chunk=CHUNK_EVAL_SIZE):
    n = rho_noisy_stack.shape[0]
    dz = np.empty(n); dx = np.empty(n); dy = np.empty(n)
    zt = np.array(zt); xt = np.array(xt); yt = np.array(yt)
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        rout = np.array(qae_fn(params, rho_noisy_stack[sl]))
        for i, ro in enumerate(rout):
            roj = jnp.array(ro)
            dz[sl.start+i] = abs(zt[sl.start+i] - float(jnp.real(jnp.trace(ZLj @ roj))))
            dx[sl.start+i] = abs(xt[sl.start+i] - float(jnp.real(jnp.trace(XLj @ roj))))
            dy[sl.start+i] = abs(yt[sl.start+i] - float(jnp.real(jnp.trace(YLj @ roj))))
    return dz, dx, dy

# ================================================================
# 10. Adaptive weight helpers
# ================================================================
def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab - STAB_SAT_LOW)/(STAB_SAT_HIGH - STAB_SAT_LOW), 0, 1))
    ws = W_STAB_HIGH - (W_STAB_HIGH - W_STAB_LOW)*sat
    wz = W_ZL_LOW + (W_ZL_HIGH - W_ZL_LOW)*sat
    wx = W_XL_LOW + (W_XL_HIGH - W_XL_LOW)*sat
    wy = W_YL_LOW + (W_YL_HIGH - W_YL_LOW)*sat
    return float(ws), float(wz), float(wx), float(wy)

def step_weight(cur, tgt):
    return cur + float(np.clip(tgt - cur, -MAX_W_DELTA, MAX_W_DELTA))

# ================================================================
# 11. One epoch
# ================================================================
def run_epoch(params, opt_state, optimizer, ws, wz, wx, wy, rng_key, loss_and_grad_fn):
    jws, jwz, jwx, jwy = (jnp.float64(ws), jnp.float64(wz),
                          jnp.float64(wx), jnp.float64(wy))
    steps = int(np.ceil(N_TRAIN_STATES/BATCH_SIZE))
    rng_key, sk = jax.random.split(rng_key)
    idx_all = np.array(jax.random.permutation(sk, N_TRAIN_STATES))
    ep_loss = 0.0; nb = 0
    for step in range(steps):
        b = idx_all[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        g_acc = jnp.zeros_like(params); l_acc = 0.0
        for i in b:
            lv, gv = loss_and_grad_fn(params, rho_noisy_train_stack[i],
                                      targets_train_stack[i], jws, jwz, jwx, jwy)
            gv = jnp.where(jnp.isnan(gv), 0.0, gv)
            lv = jnp.where(jnp.isnan(lv), 0.0, lv)
            g_acc += gv; l_acc += float(lv)
        updates, opt_state = optimizer.update(g_acc/len(b), opt_state, params)
        params = optax.apply_updates(params, updates)
        ep_loss += l_acc/len(b); nb += 1
    return params, opt_state, ep_loss/max(nb, 1), rng_key

# ================================================================
# 12. Full training run for one layer count
# ================================================================
def train_one_config(n_layers, rng_seed=42, init_params=None):
    n_p = count_params(n_layers)
    print(f"\n{'='*68}")
    print(f"  N_LAYERS = {n_layers}   params = {n_p}")
    print(f"  Ansatz: Model B [Ry+Rz] / CRX+CRY+CRZ (all-to-all)")
    if init_params is not None:
        print("  Init: LAYERWISE WARM-START (identity-grown from previous L)")
    else:
        print(f"  Init: MULTI-START warm-up ({N_INIT_ROUNDS} rounds)")
    print(f"{'='*68}")

    comps = make_layer_components(n_layers)
    loss_and_grad            = comps['loss_and_grad']
    qae_batch_chunk          = comps['qae_batch_chunk']
    batch_stab_avg           = comps['batch_stab_avg']
    batch_fidelity_arr       = comps['batch_fidelity_arr']
    batch_logical_errors_arr = comps['batch_logical_errors_arr']
    batch_baseline_loss_arr  = comps['batch_baseline_loss_arr']

    probe_rho_noisy = rho_noisy_train_stack
    probe_rho_clean = rho_clean_train_stack
    probe_zl = targets_train_stack[:, 0]
    probe_xl = targets_train_stack[:, 1]
    probe_yl = targets_train_stack[:, 2]

    def probe_metrics(p):
        bl = np.asarray(batch_baseline_loss_arr(p, probe_rho_noisy, probe_zl, probe_xl, probe_yl))
        fd = np.asarray(batch_fidelity_arr(p, probe_rho_noisy, probe_rho_clean))
        dz, dx, dy = batch_logical_errors_arr(p, probe_rho_noisy, probe_zl, probe_xl, probe_yl)
        dz = np.asarray(dz); dx = np.asarray(dx); dy = np.asarray(dy)
        return (float(bl.mean()), float(bl.std()),
                float(fd.mean()), float(fd.std()),
                float(dz.mean()), float(dz.std()),
                float(dx.mean()), float(dx.std()),
                float(dy.mean()), float(dy.std()))

    print("  JIT warm-up ...", end=" ", flush=True)
    t0 = time.time(); _p0 = jnp.zeros(n_p)
    loss_and_grad(_p0, rho_noisy_train_stack[0], targets_train_stack[0],
                  jnp.float64(W_STAB_HIGH), jnp.float64(W_ZL_LOW),
                  jnp.float64(W_XL_LOW), jnp.float64(W_YL_LOW))
    batch_stab_avg(_p0, probe_rho_noisy)
    batch_baseline_loss_arr(_p0, probe_rho_noisy, probe_zl, probe_xl, probe_yl)
    print(f"done in {time.time()-t0:.1f}s")

    steps_per_ep = int(np.ceil(N_TRAIN_STATES/BATCH_SIZE))
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(LR, TOTAL_EPOCHS*steps_per_ep)))
    rng = jax.random.PRNGKey(rng_seed)

    best_p = best_opt = None; best_f = -1.0
    best_ws = W_STAB_HIGH; best_wz = W_ZL_LOW; best_wx = W_XL_LOW; best_wy = W_YL_LOW
    best_rng = rng; best_round_hist = None

    if init_params is not None:
        print(f"\n  Phase 1 : LAYERWISE WARM-START — single {INIT_EPOCHS}-epoch refinement")
        tp = jnp.asarray(init_params, dtype=jnp.float64); to = optimizer.init(tp)
        ws, wz, wx, wy = W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW
        rng, rr = jax.random.split(rng); round_hist = []
        for ep in range(INIT_EPOCHS):
            if ep % WEIGHT_UPDATE_FREQ == 0:
                avg_s = float(batch_stab_avg(tp, probe_rho_noisy))
                tw, twz, twx, twy = compute_target_weights(avg_s)
                ws = step_weight(ws, tw); wz = step_weight(wz, twz)
                wx = step_weight(wx, twx); wy = step_weight(wy, twy)
            tp, to, _, rr = run_epoch(tp, to, optimizer, ws, wz, wx, wy, rr, loss_and_grad)
            round_hist.append((ep, 1, *probe_metrics(tp)))
        best_f = round_hist[-1][4]
        best_p, best_opt = tp, to; best_ws, best_wz, best_wx, best_wy = ws, wz, wx, wy
        best_rng = rr; best_round_hist = round_hist
        print(f"  Warm-start refinement F = {best_f:.4f}")
    else:
        print(f"\n  Phase 1 : {N_INIT_ROUNDS} rounds x {INIT_EPOCHS} epochs")
        print(f"  {'Rd':<3} {'Init style':<20} {'F_end':>7} {'⟨S⟩':>6} "
              f"{'ΔZL':>6} {'ΔXL':>6} {'ΔYL':>6}")
        for rd in range(N_INIT_ROUNDS):
            rng, sk = jax.random.split(rng); s = rd % 4
            if   s == 0: tp = jax.random.uniform(sk, (n_p,), minval=-np.pi, maxval=np.pi); sn = "Uniform[-π,π]"
            elif s == 1: tp = jax.random.normal(sk, (n_p,))*0.15; sn = "Normal(0,0.15)"
            elif s == 2: tp = jax.random.normal(sk, (n_p,))*1.0;  sn = "Normal(0,1.0)"
            else:        tp = jax.random.uniform(sk, (n_p,), minval=-0.1, maxval=0.1); sn = "Unif[-0.1,0.1]"
            to = optimizer.init(tp); ws, wz, wx, wy = W_STAB_HIGH, W_ZL_LOW, W_XL_LOW, W_YL_LOW
            rng, rr = jax.random.split(rng); round_hist = []
            for ep in range(INIT_EPOCHS):
                if ep % WEIGHT_UPDATE_FREQ == 0:
                    avg_s = float(batch_stab_avg(tp, probe_rho_noisy))
                    tw, twz, twx, twy = compute_target_weights(avg_s)
                    ws = step_weight(ws, tw); wz = step_weight(wz, twz)
                    wx = step_weight(wx, twx); wy = step_weight(wy, twy)
                tp, to, _, rr = run_epoch(tp, to, optimizer, ws, wz, wx, wy, rr, loss_and_grad)
                round_hist.append((ep, 1, *probe_metrics(tp)))
            f_end = round_hist[-1][4]
            avg_s = float(batch_stab_avg(tp, probe_rho_noisy))
            dzl, dxl, dyl = round_hist[-1][6], round_hist[-1][8], round_hist[-1][10]
            star = ""
            if f_end > best_f:
                best_f = f_end; best_p, best_opt = tp, to
                best_ws, best_wz, best_wx, best_wy = ws, wz, wx, wy
                best_rng = rr; best_round_hist = round_hist; star = " ★"
            print(f"  {rd:<3} {sn:<20} {f_end:>7.4f} {avg_s:>6.3f} "
                  f"{dzl:>6.4f} {dxl:>6.4f} {dyl:>6.4f}{star}")
            if star == "": del to
            gc.collect()
        print(f"  Best warm-up F = {best_f:.4f}")

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

    print(f"\n  Phase 2 : {MAIN_EPOCHS} continuation epochs "
          f"(global {INIT_EPOCHS} … {TOTAL_EPOCHS-1})")
    params = best_p; opt_state = best_opt
    ws, wz, wx, wy = best_ws, best_wz, best_wx, best_wy
    rng_key = best_rng
    print(f"  {'Ep':<5} {'BaseLoss':>9} {'F_tr':>7} {'⟨S⟩':>6} "
          f"{'ΔZL':>7} {'ΔXL':>7} {'ΔYL':>7} {'t':>5}")
    for ep in range(MAIN_EPOCHS):
        t0 = time.time()
        if ep % WEIGHT_UPDATE_FREQ == 0:
            avg_s = float(batch_stab_avg(params, probe_rho_noisy))
            tw, twz, twx, twy = compute_target_weights(avg_s)
            ws = step_weight(ws, tw); wz = step_weight(wz, twz)
            wx = step_weight(wx, twx); wy = step_weight(wy, twy)
        params, opt_state, _, rng_key = run_epoch(
            params, opt_state, optimizer, ws, wz, wx, wy, rng_key, loss_and_grad)
        (bm, bs, fm, fs, zm, zs, xm, xs, ym, ys) = probe_metrics(params)
        avg_s = float(batch_stab_avg(params, probe_rho_noisy))
        te = INIT_EPOCHS + ep
        hist_epoch.append(te);      hist_phase.append(2)
        hist_loss.append(bm);       hist_loss_std.append(bs)
        hist_fid.append(fm);        hist_fid_std.append(fs)
        hist_dzl.append(zm);        hist_dzl_std.append(zs)
        hist_dxl.append(xm);        hist_dxl_std.append(xs)
        hist_dyl.append(ym);        hist_dyl_std.append(ys)
        if ep % 10 == 0 or ep == MAIN_EPOCHS-1:
            print(f"  {te:<5} {bm:>9.3f} {fm:>7.4f} {avg_s:>6.3f} "
                  f"{zm:>7.4f} {xm:>7.4f} {ym:>7.4f} {time.time()-t0:>4.1f}s")
        if ep % 20 == 0: gc.collect()

    to_arr = lambda l, dt: np.array(l, dtype=dt)
    hist = dict(
        hist_epoch=to_arr(hist_epoch, np.int64), hist_phase=to_arr(hist_phase, np.int64),
        hist_loss=to_arr(hist_loss, float), hist_loss_std=to_arr(hist_loss_std, float),
        hist_fid=to_arr(hist_fid, float), hist_fid_std=to_arr(hist_fid_std, float),
        hist_dzl=to_arr(hist_dzl, float), hist_dzl_std=to_arr(hist_dzl_std, float),
        hist_dxl=to_arr(hist_dxl, float), hist_dxl_std=to_arr(hist_dxl_std, float),
        hist_dyl=to_arr(hist_dyl, float), hist_dyl_std=to_arr(hist_dyl_std, float))

    print(f"\n  Test eval ({N_TEST_STATES} states) ...", end=" ", flush=True)
    t0 = time.time()
    f_qae = eval_fidelity_chunked(params, qae_batch_chunk,
                                  rho_noisy_test_stack, rho_clean_test_stack)
    dzl_t, dxl_t, dyl_t = eval_logical_errors_chunked(
        params, qae_batch_chunk, rho_noisy_test_stack, zl_test, xl_test, yl_test)
    print(f"done in {time.time()-t0:.1f}s  F̄_qae={f_qae.mean():.5f}  "
          f"ΔZL={dzl_t.mean():.4f} ΔXL={dxl_t.mean():.4f} ΔYL={dyl_t.mean():.4f}")

    return dict(n_layers=n_layers, n_params=n_p, params=np.array(params),
                f_qae_test=f_qae, f_std_test=f_std_test, f_noisy_test=f_noisy_test,
                f_petz_test=f_petz_test,
                dzl_test=dzl_t, dxl_test=dxl_t, dyl_test=dyl_t, **hist)

# ================================================================
# 13. Run sweep
# ================================================================
print("\n" + "="*68); print("  LAYER SWEEP"); print("="*68)
sweep_results = {}; t_sweep = time.time()
prev_nl = None; prev_params = None
for nl in LAYER_SWEEP:
    if (not USE_LAYERWISE) or (prev_params is None) or (nl <= prev_nl):
        sweep_results[nl] = train_one_config(nl, rng_seed=42+nl)
    else:
        warm = grow_params(prev_params, prev_nl, nl)
        sweep_results[nl] = train_one_config(nl, rng_seed=42+nl, init_params=warm)
    prev_nl = nl; prev_params = sweep_results[nl]['params']
print(f"\n  Total sweep time : {time.time()-t_sweep:.0f}s")

# ================================================================
# 14. Save results
# ================================================================
save_dict = {'f_noisy_test': f_noisy_test, 'f_std_test': f_std_test,
             'f_petz_test': f_petz_test,                            # <-- NEW
             'meta_test_thetas': test_thetas, 'meta_test_phis': test_phis,
             'meta_sigma': np.array(SIGMA_THETA), 'meta_rho_corr': np.array(RHO_CORR),
             'meta_mu': np.array(MU_COHERENT),
             'meta_layers': np.array(LAYER_SWEEP),
             'meta_INIT_EPOCHS': np.array(INIT_EPOCHS),
             'meta_MAIN_EPOCHS': np.array(MAIN_EPOCHS),
             'meta_layerwise': np.array(int(USE_LAYERWISE))}
for nl, res in sweep_results.items():
    for k in ['params', 'hist_epoch', 'hist_phase',
              'hist_loss', 'hist_loss_std', 'hist_fid', 'hist_fid_std',
              'hist_dzl', 'hist_dzl_std', 'hist_dxl', 'hist_dxl_std',
              'hist_dyl', 'hist_dyl_std',
              'f_qae_test', 'dzl_test', 'dxl_test', 'dyl_test']:
        save_dict[f'L{nl}_{k}'] = res[k]

best_nl_save = max(LAYER_SWEEP, key=lambda nl: sweep_results[nl]['f_qae_test'].mean())
save_dict['best_n_layers'] = np.array(best_nl_save)
save_dict['best_params']   = sweep_results[best_nl_save]['params']
save_dict['best_f_qae']    = np.array(sweep_results[best_nl_save]['f_qae_test'].mean())
np.savez(f"dephasing_results_{FILE_TAG}.npz", **save_dict)
print(f"  Saved : dephasing_results_{FILE_TAG}.npz  (best L={best_nl_save})")

with open(f"dephasing_history_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "global_epoch", "phase", "baseline_loss", "baseline_loss_std",
                "fidelity", "fidelity_std", "dzl", "dzl_std", "dxl", "dxl_std", "dyl", "dyl_std"])
    for nl, res in sweep_results.items():
        for row in zip(res['hist_epoch'], res['hist_phase'],
                       res['hist_loss'], res['hist_loss_std'],
                       res['hist_fid'], res['hist_fid_std'],
                       res['hist_dzl'], res['hist_dzl_std'],
                       res['hist_dxl'], res['hist_dxl_std'],
                       res['hist_dyl'], res['hist_dyl_std']):
            e, ph = int(row[0]), int(row[1])
            w.writerow([nl, e, ph] + [f"{v:.6f}" for v in row[2:]])
print(f"  Saved : dephasing_history_{FILE_TAG}.csv")

with open(f"dephasing_results_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_layers", "n_params", "F_noisy_mean", "F_stdQEC_mean", "F_petz_mean",
                "F_qae_mean", "F_qae_std", "dzl_mean", "dxl_mean", "dyl_mean", "gap_closed_pct"])
    for nl, res in sweep_results.items():
        fn = res['f_noisy_test']; fs = res['f_std_test']; fq = res['f_qae_test']
        fp = res['f_petz_test']
        gap = fs.mean() - fn.mean()
        pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
        w.writerow([nl, res['n_params'], f"{fn.mean():.5f}", f"{fs.mean():.5f}",
                    f"{fp.mean():.5f}", f"{fq.mean():.5f}", f"{fq.std():.5f}",
                    f"{res['dzl_test'].mean():.5f}", f"{res['dxl_test'].mean():.5f}",
                    f"{res['dyl_test'].mean():.5f}", f"{pct:.1f}"])
print(f"  Saved : dephasing_results_{FILE_TAG}.csv")

# ================================================================
# 15. Plot — 4-panel comparison
# ================================================================
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
ls_ops = ['-', '--', '-', '--', '-']
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax = axes[0, 0]
for (nl, res), col, ls in zip(sweep_results.items(), colors, ls_ops):
    ax.plot(res['hist_epoch'], res['hist_fid'], color=col, ls=ls, lw=2,
            label=f"L={nl} ({res['n_params']}p) F̄={res['f_qae_test'].mean():.4f}")
    ax.fill_between(res['hist_epoch'], res['hist_fid']-res['hist_fid_std'],
                    res['hist_fid']+res['hist_fid_std'], color=col, alpha=0.12)
ax.axhline(f_std_test.mean(), color='navy', ls='--', lw=1.5, label=f"Std QEC {f_std_test.mean():.4f}")
ax.axhline(f_petz_test.mean(), color=PETZ_COLOR, ls='-.', lw=1.5, label=f"Petz map {f_petz_test.mean():.4f}")
ax.axhline(f_noisy_test.mean(), color='gray', ls=':', lw=1.5, label=f"No corr {f_noisy_test.mean():.4f}")
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel("Global epoch (Phase 1 → Phase 2)"); ax.set_ylabel(f"Avg fidelity ({N_TRAIN_STATES} train)")
ax.set_title("Training fidelity vs epoch (±1σ band)", fontweight='bold')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
for (nl, res), col, ls in zip(sweep_results.items(), colors, ls_ops):
    ax.plot(res['hist_epoch'], res['hist_loss'], color=col, ls=ls, lw=2, label=f"L={nl}")
    ax.fill_between(res['hist_epoch'], res['hist_loss']-res['hist_loss_std'],
                    res['hist_loss']+res['hist_loss_std'], color=col, alpha=0.12)
ax.axvline(INIT_EPOCHS, color='gray', ls=':', lw=1.2, alpha=0.6)
ax.set_xlabel("Global epoch"); ax.set_ylabel("Fixed-weight baseline spectral loss")
ax.set_title(f"Baseline loss vs epoch\nS={W_STAB_BASE}, ZL={W_ZL_BASE}, XL={W_XL_BASE}, YL={W_YL_BASE}",
             fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
x = np.arange(len(LAYER_SWEEP)); bw = 0.2
fn_m = [sweep_results[nl]['f_noisy_test'].mean() for nl in LAYER_SWEEP]
fs_m = [sweep_results[nl]['f_std_test'].mean() for nl in LAYER_SWEEP]
fp_m = [sweep_results[nl]['f_petz_test'].mean() for nl in LAYER_SWEEP]
fq_m = [sweep_results[nl]['f_qae_test'].mean() for nl in LAYER_SWEEP]
fq_s = [sweep_results[nl]['f_qae_test'].std() for nl in LAYER_SWEEP]
ax.bar(x-1.5*bw, fn_m, bw, label="No correction", color='#e74c3c', alpha=0.8)
ax.bar(x-0.5*bw, fs_m, bw, label="Standard QEC", color='#3498db', alpha=0.8)
ax.bar(x+0.5*bw, fp_m, bw, label="Petz map", color=PETZ_COLOR, alpha=0.8)
bars = ax.bar(x+1.5*bw, fq_m, bw, label="Spectral QEC", color='#2ecc71', alpha=0.8)
ax.errorbar(x+1.5*bw, fq_m, yerr=fq_s, fmt='none', color='black', capsize=4, lw=1.5)
for bar, val in zip(bars, fq_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f"{val:.4f}",
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel(f"Mean fidelity ({N_TEST_STATES} test)")
ax.set_title("Test fidelity vs layer count", fontweight='bold')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
dzl_m = [sweep_results[nl]['dzl_test'].mean() for nl in LAYER_SWEEP]
dxl_m = [sweep_results[nl]['dxl_test'].mean() for nl in LAYER_SWEEP]
dyl_m = [sweep_results[nl]['dyl_test'].mean() for nl in LAYER_SWEEP]
bw2 = 0.25
ax.bar(x-bw2, dzl_m, bw2, label="ΔZL", color='#e74c3c', alpha=0.85)
ax.bar(x,     dxl_m, bw2, label="ΔXL", color='#3498db', alpha=0.85)
ax.bar(x+bw2, dyl_m, bw2, label="ΔYL", color='#9b59b6', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in LAYER_SWEEP])
ax.set_ylabel("Mean |target − recovered|")
ax.set_title("Logical spectrum error vs layer count", fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

plt.suptitle(f"Spectral QAE — [[3,1,3]] Phase-Flip, correlated coherent dephasing "
             f"(σ={SIGMA_THETA}, ρ_corr={RHO_CORR}, μ={MU_COHERENT} "
             f"[{'NON-diag PTM' if MU_COHERENT!=0 else 'diag PTM'}])  |  "
             f"JAX {jax.default_backend().upper()}  |  layerwise={USE_LAYERWISE}",
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f"dephasing_fidelity_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : dephasing_fidelity_{FILE_TAG}.png")

# ================================================================
# 16. Final summary
# ================================================================
print("\n" + "="*88)
print(f"  FINAL SUMMARY — {N_TEST_STATES} test states  |  σ={SIGMA_THETA}, ρ_corr={RHO_CORR}, μ={MU_COHERENT}")
print("="*88)
hdr = (f"  {'L':>2}  {'Params':>7}  {'F_noisy':>9}  {'F_stdQEC':>9}  {'F_petz':>8}  "
       f"{'F_QAE':>8}  {'±std':>7}  {'ΔZL':>7}  {'ΔXL':>7}  {'ΔYL':>7}  {'Gap%':>6}")
print(hdr); print("  " + "─"*(len(hdr)-2))
for nl in LAYER_SWEEP:
    res = sweep_results[nl]
    fn = res['f_noisy_test']; fs = res['f_std_test']; fq = res['f_qae_test']
    fp = res['f_petz_test']
    gap = fs.mean()-fn.mean()
    pct = (fq.mean()-fn.mean())/gap*100 if gap > 1e-6 else 0.0
    print(f"  {nl:>2}  {res['n_params']:>7}  {fn.mean():>9.5f}  {fs.mean():>9.5f}  "
          f"{fp.mean():>8.5f}  {fq.mean():>8.5f}  {fq.std():>7.5f}  {res['dzl_test'].mean():>7.4f}  "
          f"{res['dxl_test'].mean():>7.4f}  {res['dyl_test'].mean():>7.4f}  {pct:>5.1f}%")
best_nl = best_nl_save
print(f"\n  Best layer : L={best_nl}  F̄={sweep_results[best_nl]['f_qae_test'].mean():.5f}")
print(f"  Petz map F̄ = {f_petz_test.mean():.5f}")

# ================================================================
# 17. Fidelity vs sigma (best model), rho_corr and mu fixed
#     Uses exact Phi rescaling + reuses the clean test-state DMs.
# ================================================================
print("\n" + "="*68)
print("  FIDELITY vs NOISE STRENGTH σ  (best model)")
print("="*68)
best_comps  = make_layer_components(best_nl)
best_qae    = best_comps['qae_batch_chunk']
best_params = jnp.array(sweep_results[best_nl]['params'])
print(f"  Best model: L={best_nl}  trained at σ={SIGMA_THETA}, ρ_corr={RHO_CORR}, μ={MU_COHERENT}")

SIGMA_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]
fqae_vs_s = []; fqae_std_vs_s = []; fnoisy_vs_s = []; fstd_vs_s = []; fpetz_vs_s = []
for sig in SIGMA_SWEEP:
    phi_s = build_phi(sig, RHO_CORR, mu=MU_COHERENT)
    rho_n_np = phi_s[None, :, :] * rho_clean_test_np             # exact, elementwise
    rho_n_j  = jnp.array(rho_n_np)
    fq = eval_fidelity_chunked(best_params, best_qae, rho_n_j, rho_clean_test_stack)
    fn = np.array([float(np.real(np.trace(rho_n_np[i] @ rho_clean_test_np[i])))
                   for i in range(N_TEST_STATES)])
    fstd = np.array([float(np.real(np.trace(
                standard_qec_phaseflip(rho_n_np[i]) @ rho_clean_test_np[i])))
                for i in range(N_TEST_STATES)])
    fpz = make_petz_fid(phi_s)(rho_n_j, rho_clean_test_stack)    # Petz for this sigma
    fqae_vs_s.append(fq.mean()); fqae_std_vs_s.append(fq.std())
    fnoisy_vs_s.append(fn.mean()); fstd_vs_s.append(fstd.mean()); fpetz_vs_s.append(fpz.mean())
    print(f"  σ={sig:>4.2f}  F_QAE={fq.mean():.4f}  F_petz={fpz.mean():.4f}  "
          f"F_std={fstd.mean():.4f}  F_noisy={fn.mean():.4f}")

SIGMA_SWEEP = np.array(SIGMA_SWEEP)
fqae_vs_s = np.array(fqae_vs_s); fqae_std_vs_s = np.array(fqae_std_vs_s)
fnoisy_vs_s = np.array(fnoisy_vs_s); fstd_vs_s = np.array(fstd_vs_s)
fpetz_vs_s = np.array(fpetz_vs_s)

np.savez(f"dephasing_fidelity_vs_sigma_{FILE_TAG}.npz",
         best_n_layers=np.array(best_nl), best_params=sweep_results[best_nl]['params'],
         rho_corr=np.array(RHO_CORR), mu=np.array(MU_COHERENT),
         sigma=SIGMA_SWEEP,
         f_qae=fqae_vs_s, f_qae_std=fqae_std_vs_s, f_noisy=fnoisy_vs_s,
         f_std=fstd_vs_s, f_petz=fpetz_vs_s)
with open(f"dephasing_fidelity_vs_sigma_{FILE_TAG}.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["sigma", "F_qae", "F_qae_std", "F_noisy", "F_stdQEC", "F_petz"])
    for i in range(len(SIGMA_SWEEP)):
        w.writerow([f"{SIGMA_SWEEP[i]:.3f}", f"{fqae_vs_s[i]:.6f}", f"{fqae_std_vs_s[i]:.6f}",
                    f"{fnoisy_vs_s[i]:.6f}", f"{fstd_vs_s[i]:.6f}", f"{fpetz_vs_s[i]:.6f}"])
print(f"  Saved : dephasing_fidelity_vs_sigma_{FILE_TAG}.npz / .csv")

fig2, ax = plt.subplots(figsize=(9, 6))
ax.plot(SIGMA_SWEEP, fqae_vs_s, 'o-', color='#2ecc71', lw=2, label=f"Spectral QAE (L={best_nl})")
ax.fill_between(SIGMA_SWEEP, fqae_vs_s-fqae_std_vs_s, fqae_vs_s+fqae_std_vs_s, color='#2ecc71', alpha=0.15)
ax.plot(SIGMA_SWEEP, fstd_vs_s,  's--', color='#3498db', lw=2, label="Standard QEC")
ax.plot(SIGMA_SWEEP, fpetz_vs_s, 'd-.', color=PETZ_COLOR, lw=2, label="Petz map")
ax.plot(SIGMA_SWEEP, fnoisy_vs_s, '^-', color='#e74c3c', lw=2, label="No correction")
ax.axvline(SIGMA_THETA, color='black', ls=':', lw=1.2, alpha=0.6)
_ylo, _yhi = ax.get_ylim()
ax.text(SIGMA_THETA + 0.02, _ylo + 0.05*(_yhi-_ylo),
        f"trained @ σ={SIGMA_THETA}", fontsize=8, rotation=0, va='bottom')
ax.set_xlabel("σ  (per-qubit rotation-angle std dev)"); ax.set_ylabel("Mean fidelity")
ax.set_title(f"Fidelity vs noise strength — best model "
             f"(L={best_nl}, ρ_corr={RHO_CORR}, μ={MU_COHERENT})",
             fontweight='bold')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig(f"dephasing_fidelity_vs_sigma_{FILE_TAG}.png", dpi=150, bbox_inches='tight')
print(f"  Saved : dephasing_fidelity_vs_sigma_{FILE_TAG}.png")
print("\n  Done ✓")