# ============================================================================
# qae5_noisy_common_prune.py  
#
# [[5,1,3]] perfect code.  external = single-qubit depolarizing at p_i per HALF
# interval; internal = gate depolarizing at p1q/p2q.
#
# PROTOCOL:
#   (a) |psi_L> -> D_{p_i} -> noisy QAE -> D_{p_i} -> perfect QEC
#   (b) |psi_L> -> D_{p_i} ->  nothing  -> D_{p_i} -> perfect QEC
#   (c) |psi>   -> D_{p_i} ->  nothing  -> D_{p_i}
# ============================================================================
import warnings; warnings.filterwarnings("ignore")
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE",  "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import jax
jax.config.update("jax_enable_x64", True)
import logging; logging.getLogger("jax").setLevel(logging.WARNING)
try:
    jax.config.update("jax_platform_name", "gpu"); jax.devices()
except Exception:
    jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from functools import reduce

# ---------------------------------------------------------------- dimensions
N_CODE, N_ANC = 5, 1
N_TOTAL = N_CODE + N_ANC          # 6
SYS_DIM = 2 ** N_CODE             # 32
ANC_DIM = 2 ** N_ANC              # 2
TOT_DIM = 2 ** N_TOTAL            # 64

# ------------------------------------------- external noise (PER HALF INTERVAL)
PEXT_TRAIN         = 0.05         # p_i the noiseless model was trained at
PEXT_SWEEP_DEFAULT = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]

# ------------------------------------------------- internal-noise parameters
P2Q_SWEEP = [0.0, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
P1Q_RATIO = 0.1
NOISY_RZ    = True
NOISY_RESET = False
P_RESET     = 0.0

# ------------------------------------------------------------ circuit depth
L_TARGET       = 3
LAYERWISE_PATH = list(range(1, L_TARGET + 1))
REMAT_GROUPS   = 8

# ------------------------------------------------------------ gate pruning
PRUNE_TOL_ROT   = 1e-4            # drop a single rotation
PRUNE_TOL_BLOCK = 1e-3            # drop a whole C-U block (incl. 2 CNOTs)

# ================================================================
# 1. Pauli / code operators
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0, 1], [1, 0]], dtype=complex)
Y_ = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_ = np.array([[1, 0], [0, -1]], dtype=complex)

def kron_list(ops): return reduce(np.kron, ops)

def pauli_string(s):
    g = {'I': I2, 'X': X_, 'Y': Y_, 'Z': Z_}
    return kron_list([g[c] for c in s])

S1_mat = pauli_string('XZZXI'); S2_mat = pauli_string('IXZZX')
S3_mat = pauli_string('XIXZZ'); S4_mat = pauli_string('ZXIXZ')
stabilizers_mat = [S1_mat, S2_mat, S3_mat, S4_mat]
XL_mat = pauli_string('XXXXX')
ZL_mat = pauli_string('ZZZZZ')
YL_mat = pauli_string('YYYYY')

S1j = jnp.array(S1_mat); S2j = jnp.array(S2_mat)
S3j = jnp.array(S3_mat); S4j = jnp.array(S4_mat)
ZLj = jnp.array(ZL_mat); XLj = jnp.array(XL_mat); YLj = jnp.array(YL_mat)

ANC_0_J       = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0, 0].set(1.0)
FRESH_00000_J = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0, 0].set(1.0)

def compute_logical_states():
    dim  = 2 ** N_CODE
    proj = np.eye(dim, dtype=complex)
    for S in stabilizers_mat:
        proj = proj @ ((np.eye(dim) + S) / 2)
    evals, evecs = np.linalg.eigh(proj)
    basis = [evecs[:, i] for i in range(dim) if abs(evals[i] - 1.0) < 1e-6]
    assert len(basis) == 2
    v0, v1 = basis
    zl = np.array([[v0 @ ZL_mat @ v0, v0 @ ZL_mat @ v1],
                   [v1 @ ZL_mat @ v0, v1 @ ZL_mat @ v1]])
    ev, evec = np.linalg.eigh(zl)
    if ev[0] > ev[1]:
        k0 = evec[0, 0]*v0 + evec[1, 0]*v1; k1 = evec[0, 1]*v0 + evec[1, 1]*v1
    else:
        k0 = evec[0, 1]*v0 + evec[1, 1]*v1; k1 = evec[0, 0]*v0 + evec[1, 0]*v1
    return k0/np.linalg.norm(k0), k1/np.linalg.norm(k1)

ket_0L, ket_1L = compute_logical_states()

def logical_state(theta, phi):
    return np.cos(theta/2)*ket_0L + np.exp(1j*phi)*np.sin(theta/2)*ket_1L

def state_to_dm(psi):
    psi = psi.reshape(-1, 1); return psi @ psi.conj().T

# ================================================================
# 2. External noise: single-qubit depolarizing on all 5 code qubits
# ================================================================
def depolarizing_channel(rho, p):
    """Reference (numpy, explicit Kraus) -- used for datasets + validation."""
    result = rho.copy()
    for q in range(N_CODE):
        nr = (1 - p) * result
        for P in (X_, Y_, Z_):
            ops = [I2]*N_CODE; ops[q] = P
            E = kron_list(ops)
            nr += (p/3) * E @ result @ E.conj().T
        result = nr
    return result

_SPLIT1_CODE = {q: (2**q, 2**(N_CODE-q-1)) for q in range(N_CODE)}
I2J = jnp.eye(2, dtype=complex)

def _mix1_code(rho, q):
    """I_q tensor Tr_q rho  on the 32-dim code register, rho (B,32,32)."""
    B = rho.shape[0]; l, m = _SPLIT1_CODE[q]
    t  = rho.reshape(B, l, 2, m, l, 2, m)
    tr = jnp.trace(t, axis1=2, axis2=5)                 # (B,l,m,l,m)
    return jnp.einsum('bimjn,xy->bixmjyn', tr, I2J).reshape(B, SYS_DIM, SYS_DIM)

def depol_code_j(rho, p):
    """Closed-form single-qubit depolarizing on all 5 code qubits, (B,32,32)."""
    for q in range(N_CODE):
        rho = (1.0 - 4.0*p/3.0)*rho + (2.0*p/3.0)*_mix1_code(rho, q)
    return rho

def validate_external_depol(p=0.137, seed=1, verbose=True):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(SYS_DIM, SYS_DIM)) + 1j*rng.normal(size=(SYS_DIM, SYS_DIM))
    rho = A @ A.conj().T; rho /= np.trace(rho).real
    ref = depolarizing_channel(rho, p)
    got = np.asarray(depol_code_j(jnp.array(rho[None]), p))[0]
    err = float(np.max(np.abs(got - ref)))
    if verbose:
        print(f"  [validate] closed-form external depol vs Kraus: {err:.3e}")
    assert err < 1e-12, "external depolarizing (JAX) wrong"
    return err

def fibonacci_sphere(n):
    golden = (1 + np.sqrt(5)) / 2
    return [(np.arccos(1 - 2*(i+0.5)/n), (2*np.pi*i/golden) % (2*np.pi))
            for i in range(n)]

# ================================================================
# 3. FINAL PERFECT QEC (the measuring instrument)
# ================================================================
CORRECTABLE_LABELS = ['IIIII'] + ['I'*k + P + 'I'*(N_CODE-1-k)
                                  for k in range(N_CODE) for P in 'XYZ']
CORR_OPS = np.stack([pauli_string(s) for s in CORRECTABLE_LABELS])  # (16,32,32)
N_CORR   = CORR_OPS.shape[0]

def recovery_vectors(psi_batch):
    """psi (B,32) -> V (B,16,32) with V[b,s] = E_s |psi_b>."""
    return np.einsum('sij,bj->bsi', CORR_OPS, np.asarray(psi_batch))

def final_qec_fidelity_np(rho, V):
    return np.real(np.einsum('bsi,bij,bsj->b', V.conj(), rho, V))

def _final_qec_fidelity_j(rho, V):
    return jnp.real(jnp.einsum('bsi,bij,bsj->b', jnp.conj(V), rho, V))

def validate_recovery(n_states=16, verbose=True):
    worst_proj, worst_corr, worst_noerr = 0.0, 0.0, 0.0
    for th, ph in fibonacci_sphere(n_states):
        psi = logical_state(th, ph)
        V   = recovery_vectors(psi[None])[0]
        Pi  = np.einsum('si,sj->ij', V, V.conj())
        worst_proj = max(worst_proj, float(np.max(np.abs(Pi @ Pi - Pi))),
                         float(abs(np.trace(Pi).real - N_CORR)),
                         float(np.max(np.abs(Pi - Pi.conj().T))))
        r0 = state_to_dm(psi)
        worst_noerr = max(worst_noerr, abs(np.real(np.trace(Pi @ r0)) - 1.0))
        for s in CORRECTABLE_LABELS[1:]:
            E = pauli_string(s)
            worst_corr = max(worst_corr,
                             abs(np.real(np.trace(Pi @ E @ r0 @ E)) - 1.0))
    if verbose:
        print(f"  [validate] Pi projector/rank        : {worst_proj:.2e}")
        print(f"  [validate] P(no error)          = 1 : {worst_noerr:.2e}")
        print(f"  [validate] P(any weight-1 err)  = 1 : {worst_corr:.2e}")
    assert max(worst_proj, worst_corr, worst_noerr) < 1e-10, \
        "final perfect-QEC recovery is wrong"
    return worst_proj

# ================================================================
# 4. Analytic baselines for TWO half intervals   (EXACT)
# ================================================================
def p_eff_two_rounds(p):
    """Composition of two single-qubit depolarizing channels of strength p."""
    return 0.75*(1.0 - (1.0 - 4.0*np.asarray(p, float)/3.0)**2)

def P_bare_physical(p_i):
    """Scenario (c): bare physical qubit, two half intervals, no QEC. Exact."""
    return 1.0 - (2.0/3.0)*p_eff_two_rounds(p_i)

def uncorr_analytic_5q(p_i):
    """EXACT <P> for  depol(p_i) -> depol(p_i) -> perfect [[5,1,3]] QEC."""
    q = p_eff_two_rounds(p_i)               # composed single-qubit depol prob.
    a, b = 1.0 - q, q / 3.0                 # P(I) and P(each Pauli) per qubit
    pI = (a**5 + 15.0*a**4*b + 60.0*a**2*b**3
          + 135.0*a*b**4 + 45.0*b**5)
    return (1.0 + 2.0*pI) / 3.0

# keep the historical name as an alias so plotting scripts pick up the fix
P_uncorr_analytic = uncorr_analytic_5q

def break_even_p_i(lo=1e-6, hi=0.5, tol=1e-12):
    """p_i at which the uncorrected LOGICAL qubit stops beating the BARE
    physical qubit (the pseudothreshold).  ~0.0723 for this code+noise."""
    f = lambda p: float(uncorr_analytic_5q(p) - P_bare_physical(p))
    a, b = lo, hi
    if f(a) <= 0 or f(b) >= 0:
        return float('nan')
    while b - a > tol:
        m = 0.5*(a + b)
        if f(m) > 0: a = m
        else:        b = m
    return 0.5*(a + b)

def validate_uncorr_formula(verbose=True):
    """Enumerator normalisation + maximally-mixed limit + pseudothreshold."""
    counts = {0: 1, 1: 15, 3: 60, 4: 135, 5: 45}
    assert sum(counts.values()) == 256, "weight enumerator does not sum to 256"
    half = float(uncorr_analytic_5q(0.75))          # q = 3/4 -> fully mixed
    dev  = abs(half - 0.5)
    be   = break_even_p_i()
    if verbose:
        print(f"  [validate] uncorr formula, maximally-mixed limit: "
              f"<P>={half:.12f} (dev {dev:.2e})")
        print(f"  [info]     logical-vs-bare break-even (pseudothreshold): "
              f"p_i* = {be:.4f}  (q* = {p_eff_two_rounds(be):.4f})")
        print(f"             -> for p_i > {be:.3f} the BARE physical qubit is "
              f"better than the uncorrected logical qubit.  Expected: the "
              f"[[5,1,3]] code has no correctable weight-2 error.")
    assert dev < 1e-12, "uncorr_analytic_5q fails the q=3/4 limit"
    return dev

# ================================================================
# 5. Protocol dataset
# ================================================================
def generate_protocol_dataset(n_states, p_i):
    PSI, RC, RH, TG = [], [], [], []
    for th, ph in fibonacci_sphere(n_states):
        psi = logical_state(th, ph); r = state_to_dm(psi)
        PSI.append(psi); RC.append(r); RH.append(depolarizing_channel(r, p_i))
        TG.append([float(np.real(np.trace(ZL_mat @ r))),
                   float(np.real(np.trace(XL_mat @ r))),
                   float(np.real(np.trace(YL_mat @ r)))])
    PSI = np.stack(PSI); RC = np.stack(RC); RH = np.stack(RH); TG = np.array(TG)
    V = recovery_vectors(PSI)
    # scenario (b): second half interval, then perfect QEC (no QAE in between)
    R2 = np.stack([depolarizing_channel(RH[i], p_i) for i in range(n_states)])
    p_un  = final_qec_fidelity_np(R2, V)
    f_raw = np.real(np.einsum('bij,bji->b', RH, RC))
    return dict(psi=PSI, rho_clean=jnp.array(RC), rho_half=jnp.array(RH),
                tgt=jnp.array(TG), V=jnp.array(V),
                p_i=float(p_i), p_uncorr=p_un, f_raw_nocorr=f_raw,
                n=int(n_states))

def generate_dataset(n_states, p_ext):                 # legacy alias
    D = generate_protocol_dataset(n_states, p_ext)
    return (np.asarray(D['rho_clean']), np.asarray(D['rho_half']),
            np.asarray(D['tgt']))

def check_uncorr_against_analytic(D, tol_warn=2e-3, tol_fail=1e-2):
    """Cross-check scenario (b) against the EXACT closed form.  Residual is
    pure Fibonacci-sampling error (~1e-4 at 1000 states)."""
    num, ana = float(D['p_uncorr'].mean()), float(uncorr_analytic_5q(D['p_i']))
    dev = abs(num - ana)
    tag = "OK" if dev < tol_warn else "CHECK"
    print(f"    p_i={D['p_i']:.3f}: P_uncorr numeric {num:.6f} vs "
          f"analytic {ana:.6f}  (dev {dev:.2e}) [{tag}]")
    assert dev < tol_fail, "scenario (b) disagrees with the closed form -> " \
                           "projector or channel composition is wrong"
    return dev

# ================================================================
# 6. Ansatz: parameter layout + C-U(beta,gamma,delta) gate schedule
# ================================================================
def count_params(n_layers, n_total=N_TOTAL):
    return (n_layers+1)*n_total*2 + n_layers*n_total*(n_total-1)*3

def build_schedule(n_layers, n_total=N_TOTAL):
    ops = []; idx = 0
    for q in range(n_total):
        ops.append(("ROT", "Y", q, ((idx, 1.0),))); idx += 1
        ops.append(("ROT", "Z", q, ((idx, 1.0),))); idx += 1
    for _ in range(n_layers):
        for c in range(n_total):
            for t in range(n_total):
                if c == t: continue
                b, g, d = idx, idx+1, idx+2; idx += 3
                ops.append(("ROT", "Z", t, ((d,  0.5), (b, -0.5))))
                ops.append(("CNOT", c, t))
                ops.append(("ROT", "Z", t, ((d, -0.5), (b, -0.5))))
                ops.append(("ROT", "Y", t, ((g, -0.5),)))
                ops.append(("CNOT", c, t))
                ops.append(("ROT", "Y", t, ((g,  0.5),)))
                ops.append(("ROT", "Z", t, ((b,  1.0),)))
        for q in range(n_total):
            ops.append(("ROT", "Y", q, ((idx, 1.0),))); idx += 1
            ops.append(("ROT", "Z", q, ((idx, 1.0),))); idx += 1
    assert idx == count_params(n_layers, n_total), "param layout mismatch"
    return ops

def adjoint_schedule(ops):
    out = []
    for op in reversed(ops):
        if op[0] == "ROT":
            _, ax, q, terms = op
            out.append(("ROT", ax, q, tuple((i, -c) for i, c in terms)))
        else:
            out.append(op)
    return out

def gate_counts(ops):
    nr  = sum(1 for o in ops if o[0] == "ROT")
    nc  = sum(1 for o in ops if o[0] == "CNOT")
    nrz = sum(1 for o in ops if o[0] == "ROT" and o[1] == "Z")
    return nr, nc, nrz

def effective_C(n_cnot_full, n_rot_full, n_rz_full,
                p1q_ratio=P1Q_RATIO, noisy_rz=NOISY_RZ):
    n_noisy_rot = n_rot_full if noisy_rz else (n_rot_full - n_rz_full)
    return (16.0/15.0)*n_cnot_full + (4.0/3.0)*p1q_ratio*n_noisy_rot

# ================================================================
# 7. Gate application on batched density matrices 
# ================================================================
def _rot2(axis, theta):
    P = jnp.array(Y_) if axis == "Y" else jnp.array(Z_)
    return jnp.cos(theta/2)*I2J - 1j*jnp.sin(theta/2)*P

_SPLIT1 = {q: (2**q, 2**(N_TOTAL-q-1)) for q in range(N_TOTAL)}

def _apply_1q(rho, U, q):
    B = rho.shape[0]; l, m = _SPLIT1[q]
    t = rho.reshape(B, l, 2, m, TOT_DIM)
    t = jnp.einsum('yx,bixjc->biyjc', U, t).reshape(B, TOT_DIM, TOT_DIM)
    t = t.reshape(B, TOT_DIM, l, 2, m)
    Ud = jnp.conj(U).T
    return jnp.einsum('bcixj,xy->bciyj', t, Ud).reshape(B, TOT_DIM, TOT_DIM)

def _cnot_perm(c, t, n=N_TOTAL):
    idx  = np.arange(2**n)
    cbit = (idx >> (n-1-c)) & 1
    return jnp.array(idx ^ (cbit << (n-1-t)))

CNOT_PERM = {(c, t): _cnot_perm(c, t)
             for c in range(N_TOTAL) for t in range(N_TOTAL) if c != t}

def _apply_cnot(rho, c, t):
    perm = CNOT_PERM[(c, t)]
    return jnp.take(jnp.take(rho, perm, axis=1), perm, axis=2)

# ================================================================
# 8. Internal noise: local depolarizing in closed form (64-dim register)
# ================================================================
def _mix1(rho, q):
    B = rho.shape[0]; l, m = _SPLIT1[q]
    t  = rho.reshape(B, l, 2, m, l, 2, m)
    tr = jnp.trace(t, axis1=2, axis2=5)
    return jnp.einsum('bimjn,xy->bixmjyn', tr, I2J).reshape(B, TOT_DIM, TOT_DIM)

_SPLIT2 = {}
for _c in range(N_TOTAL):
    for _t in range(N_TOTAL):
        if _c == _t: continue
        q1, q2 = sorted((_c, _t))
        _SPLIT2[(_c, _t)] = (2**q1, 2**(q2-q1-1), 2**(N_TOTAL-q2-1))

def _mix2(rho, c, t):
    B = rho.shape[0]; a, b, d = _SPLIT2[(c, t)]
    x = rho.reshape(B, a, 2, b, 2, d, a, 2, b, 2, d)
    x = jnp.trace(x, axis1=2, axis2=7)
    x = jnp.trace(x, axis1=3, axis2=7)
    return jnp.einsum('bikpjlq,xy,zw->bixkzpjylwq', x, I2J, I2J).reshape(
        B, TOT_DIM, TOT_DIM)

def depol1(rho, q, p):
    return (1.0 - 4.0*p/3.0)*rho + (2.0*p/3.0)*_mix1(rho, q)

def depol2(rho, c, t, p):
    return (1.0 - 16.0*p/15.0)*rho + (4.0*p/15.0)*_mix2(rho, c, t)

def _embed(pairs, n_total=N_TOTAL):
    return kron_list([pairs.get(q, I2) for q in range(n_total)])

def validate_depol_channels(p=0.137, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(TOT_DIM, TOT_DIM)) + 1j*rng.normal(size=(TOT_DIM, TOT_DIM))
    rho = A @ A.conj().T; rho /= np.trace(rho).real
    rj = jnp.array(rho[None]); worst = 0.0
    for q in range(N_TOTAL):
        ks = [np.sqrt(1-p)*np.eye(TOT_DIM)] + \
             [np.sqrt(p/3)*_embed({q: P}) for P in (X_, Y_, Z_)]
        ref = sum(K @ rho @ K.conj().T for K in ks)
        worst = max(worst, float(np.max(np.abs(
            np.asarray(depol1(rj, q, p))[0] - ref))))
    P4 = [I2, X_, Y_, Z_]
    for (c, t) in [(0, 1), (2, 5), (4, 1)]:
        ks = [np.sqrt(1-p)*np.eye(TOT_DIM)]
        for i, Aop in enumerate(P4):
            for j, Bop in enumerate(P4):
                if i == 0 and j == 0: continue
                ks.append(np.sqrt(p/15)*_embed({c: Aop, t: Bop}))
        ref = sum(K @ rho @ K.conj().T for K in ks)
        worst = max(worst, float(np.max(np.abs(
            np.asarray(depol2(rj, c, t, p))[0] - ref))))
    tr1 = float(abs(np.trace(np.asarray(depol1(rj, 3, p))[0]).real - 1.0))
    tr2 = float(abs(np.trace(np.asarray(depol2(rj, 0, 4, p))[0]).real - 1.0))
    if verbose:
        print(f"  [validate] internal depol vs Kraus : {worst:.3e}")
        print(f"  [validate] trace preservation      : {max(tr1,tr2):.3e}")
    assert worst < 1e-12 and max(tr1, tr2) < 1e-12, "depolarizing channel wrong"
    return worst

# ================================================================
# 9. Partial traces / kron helpers
# ================================================================
def _ptrace_system(rho):
    r = rho.reshape(-1, SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=1, axis2=3)

def _ptrace_ancilla(rho):
    r = rho.reshape(-1, SYS_DIM, ANC_DIM, SYS_DIM, ANC_DIM)
    return jnp.trace(r, axis1=2, axis2=4)

def _kron_batch_sys_anc(sys_mat, anc_batch):
    o = jnp.einsum('ij,bkl->bikjl', sys_mat, anc_batch)
    return o.reshape(-1, TOT_DIM, TOT_DIM)

def _kron_sysbatch_anc(sys_batch, anc_mat):
    o = jnp.einsum('bij,kl->bikjl', sys_batch, anc_mat)
    return o.reshape(-1, TOT_DIM, TOT_DIM)

# ================================================================
# 10. GATE PRUNING -- value-dependent schedule
# ================================================================
def _dev_from_trace(c):
    """||U - I||_2 for U in SU(2) with Re(tr U)/2 = c: sqrt(2 - 2c)."""
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 - 2.0*c)))

def rot_dev(theta):
    """dev = 2|sin(theta/4)|; exactly 0 iff theta = 0 mod 4pi."""
    return _dev_from_trace(np.cos(float(theta)/2.0))

def block_dev(beta, gamma, delta):
    """||C-U - I||_2 with U = RZ(beta) RY(gamma) RZ(delta).  The control makes
    the global phase observable -> U = -I is NOT prunable."""
    return _dev_from_trace(np.cos(float(gamma)/2.0) *
                           np.cos((float(beta) + float(delta))/2.0))

def block_param_idx(n_layers, n_total=N_TOTAL):
    idx = 2*n_total; rows = []
    for _ in range(n_layers):
        for c in range(n_total):
            for t in range(n_total):
                if c == t: continue
                rows.append((idx, idx+1, idx+2)); idx += 3
        idx += 2*n_total
    assert idx == count_params(n_layers, n_total)
    return np.asarray(rows, dtype=int)

def _theta_of(p, terms):
    return float(sum(co*p[i] for i, co in terms))

def _simplify(ops, p, tol_rot):
    """Fixed-point peephole.  EXACT: CNOT^2 = I, R(t1)R(t2) = R(t1+t2) on the
    same axis/qubit.  TOLERANCE-BASED: drop R with dev < tol_rot."""
    changed = True
    while changed:
        changed = False; out = []; i = 0; n = len(ops)
        while i < n:
            a = ops[i]; b = ops[i+1] if i+1 < n else None
            if (b is not None and a[0] == "CNOT" and b[0] == "CNOT"
                    and a[1] == b[1] and a[2] == b[2]):
                i += 2; changed = True; continue
            if (b is not None and a[0] == "ROT" and b[0] == "ROT"
                    and a[1] == b[1] and a[2] == b[2]):
                out.append(("ROT", a[1], a[2], tuple(a[3]) + tuple(b[3])))
                i += 2; changed = True; continue
            if a[0] == "ROT" and rot_dev(_theta_of(p, a[3])) < tol_rot:
                i += 1; changed = True; continue
            out.append(a); i += 1
        ops = out
    return ops

def build_pruned_schedule(params, n_layers, n_total=N_TOTAL,
                          tol_rot=PRUNE_TOL_ROT, tol_block=PRUNE_TOL_BLOCK,
                          simplify=True):
    """Same layout as build_schedule(), with identity gates removed.  Dropping
    a C-U block also drops its TWO CNOTs -> the internal-noise saving."""
    p = np.asarray(params, float).ravel()
    assert p.size == count_params(n_layers, n_total), "param count mismatch"
    ops = []; idx = 0; n_blk = 0; n_keep = 0
    for q in range(n_total):
        ops.append(("ROT", "Y", q, ((idx, 1.0),))); idx += 1
        ops.append(("ROT", "Z", q, ((idx, 1.0),))); idx += 1
    for _ in range(n_layers):
        for c in range(n_total):
            for t in range(n_total):
                if c == t: continue
                b, g, d = idx, idx+1, idx+2; idx += 3
                n_blk += 1
                if block_dev(p[b], p[g], p[d]) < tol_block:
                    continue
                n_keep += 1
                ops.append(("ROT", "Z", t, ((d,  0.5), (b, -0.5))))
                ops.append(("CNOT", c, t))
                ops.append(("ROT", "Z", t, ((d, -0.5), (b, -0.5))))
                ops.append(("ROT", "Y", t, ((g, -0.5),)))
                ops.append(("CNOT", c, t))
                ops.append(("ROT", "Y", t, ((g,  0.5),)))
                ops.append(("ROT", "Z", t, ((b,  1.0),)))
        for q in range(n_total):
            ops.append(("ROT", "Y", q, ((idx, 1.0),))); idx += 1
            ops.append(("ROT", "Z", q, ((idx, 1.0),))); idx += 1
    assert idx == count_params(n_layers, n_total), "param layout mismatch"
    if simplify:
        ops = _simplify(ops, p, tol_rot)
    nr,  nc,  nrz  = gate_counts(ops)
    nr0, nc0, nrz0 = gate_counts(build_schedule(n_layers, n_total))
    used = sorted({i for o in ops if o[0] == "ROT" for i, _ in o[3]})
    st = dict(n_layers=int(n_layers), tol_rot=float(tol_rot),
              tol_block=float(tol_block),
              n_blocks=int(n_blk), n_blocks_kept=int(n_keep),
              n_rot_half=int(nr), n_cnot_half=int(nc), n_rz_half=int(nrz),
              n_rot_half_full=int(nr0), n_cnot_half_full=int(nc0),
              n_rz_half_full=int(nrz0), n_params_active=len(used),
              n_params_total=int(count_params(n_layers, n_total)))
    return ops, st

def prune_is_trivial(st):
    return (st['n_cnot_half'] == st['n_cnot_half_full'] and
            st['n_rot_half']  == st['n_rot_half_full'])

def sparsify_params(params, ops):
    """Zero every parameter that NO surviving gate references."""
    p = np.asarray(params, float).ravel().copy()
    used = {i for o in ops if o[0] == "ROT" for i, _ in o[3]}
    mask = np.ones(p.size, bool); mask[sorted(used)] = False
    p[mask] = 0.0
    return p, int(mask.sum())

def validate_pruning(params, n_layers, tol_rot=PRUNE_TOL_ROT,
                     tol_block=PRUNE_TOL_BLOCK, verbose=True):
    """Returns (unitary deviation, pruned ops, stats, sparsified params)."""
    p = np.asarray(params, float).ravel()
    ops, st = build_pruned_schedule(p, n_layers, tol_rot=tol_rot,
                                   tol_block=tol_block)
    ps, nz = sparsify_params(p, ops)
    U0 = schedule_unitary(p,  build_schedule(n_layers))
    U1 = schedule_unitary(ps, ops)
    dev = float(np.linalg.norm(U0 - U1, 2))
    ops2, _ = build_pruned_schedule(ps, n_layers, tol_rot=tol_rot,
                                    tol_block=tol_block)
    if gate_counts(ops2) != gate_counts(ops):
        print("  [prune] WARNING: pruning is not idempotent")
    st['unitary_dev'] = dev; st['n_params_zeroed'] = nz
    if verbose:
        print(f"  [prune] L={n_layers}: blocks {st['n_blocks_kept']}/"
              f"{st['n_blocks']} | CNOT/half {st['n_cnot_half']}/"
              f"{st['n_cnot_half_full']} | ROT/half {st['n_rot_half']}/"
              f"{st['n_rot_half_full']} | active params "
              f"{st['n_params_active']}/{st['n_params_total']} | "
              f"||U_pruned - U_full||_2 = {dev:.3e}")
    return dev, ops, st, ps

def param_sparsity_report(params, n_layers, n_total=N_TOTAL,
                          tols=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2), verbose=True):
    """DIAGNOSTIC: how many rotations / C-U blocks are (near) identity?"""
    p = np.asarray(params, float).ravel()
    single, idx = [], 0
    for q in range(n_total):
        single += [idx, idx+1]; idx += 2
    blocks = []
    for _ in range(n_layers):
        for c in range(n_total):
            for t in range(n_total):
                if c == t: continue
                blocks.append((idx, idx+1, idx+2)); idx += 3
        for q in range(n_total):
            single += [idx, idx+1]; idx += 2
    ds = np.array([rot_dev(p[i]) for i in single])
    db = np.array([block_dev(p[b], p[g], p[d]) for b, g, d in blocks])
    rows = [(t, int((ds < t).sum()), int((db < t).sum())) for t in tols]
    if verbose:
        print(f"    [sparsity] L={n_layers}: deviation from the identity")
        print(f"      {'tol':>8} {'1q rots ~ I':>17} {'C-U blocks ~ I':>19}")
        for t, a, b in rows:
            print(f"      {t:>8.0e} {a:>10}/{len(ds):<6} {b:>10}/{len(db):<8}")
    return dict(dev_single=ds, dev_block=db, rows=rows)

# ================================================================
# 11. Noisy QAE forward map + SPECTRAL loss + protocol metrics
# ================================================================
def _run_ops(rho, params, ops, p1q, p2q, noisy_rz):
    for op in ops:
        if op[0] == "ROT":
            _, ax, q, terms = op
            th  = sum(co*params[i] for i, co in terms)
            rho = _apply_1q(rho, _rot2(ax, th), q)
            if p1q > 0 and (ax != "Z" or noisy_rz):
                rho = depol1(rho, q, p1q)
        else:
            _, c, t = op
            rho = _apply_cnot(rho, c, t)
            if p2q > 0:
                rho = depol2(rho, c, t, p2q)
    return rho

def _make_runner(ops, p1q, p2q, noisy_rz, n_groups):
    if len(ops) == 0:
        return lambda rho, params: rho
    if n_groups is None or n_groups <= 1:
        return lambda rho, params: _run_ops(rho, params, ops, p1q, p2q, noisy_rz)
    n_groups = int(min(n_groups, len(ops)))
    fns = []
    for g in np.array_split(np.arange(len(ops)), n_groups):
        sub = [ops[i] for i in g]
        if not sub: continue
        def f(rho, params, sub=sub):
            return _run_ops(rho, params, sub, p1q, p2q, noisy_rz)
        fns.append(jax.checkpoint(f))
    def run(rho, params):
        for f in fns: rho = f(rho, params)
        return rho
    return run

def make_noisy_qae(n_layers, p2q, p1q=None, noisy_rz=NOISY_RZ,
                   noisy_reset=NOISY_RESET, p_reset=P_RESET,
                   remat_groups=REMAT_GROUPS, verbose=False,
                   prune_params=None, prune_tol_rot=PRUNE_TOL_ROT,
                   prune_tol_block=PRUNE_TOL_BLOCK, group_l1=0.0):
    """prune_params : if given, the schedule is pruned for those values and the
                      decoder is the adjoint of the ACTUAL (pruned) encoder.
       group_l1     : optional group lasso over each (beta,gamma,delta) triple.
    """
    if p1q is None: p1q = P1Q_RATIO * p2q
    if prune_params is None:
        ops_f, pstat = build_schedule(n_layers), None
    else:
        ops_f, pstat = build_pruned_schedule(prune_params, n_layers,
                                             tol_rot=prune_tol_rot,
                                             tol_block=prune_tol_block)
    ops_b = adjoint_schedule(ops_f)
    run_f = _make_runner(ops_f, p1q, p2q, noisy_rz, remat_groups)
    run_b = _make_runner(ops_b, p1q, p2q, noisy_rz, remat_groups)
    BLK = jnp.array(block_param_idx(n_layers)) if group_l1 > 0 else None

    def _pen(p):
        if group_l1 <= 0: return 0.0
        return group_l1*jnp.sum(jnp.sqrt(jnp.sum(p[BLK]**2, axis=1) + 1e-12))

    def _fwd(params, rho_code):                       # (B,32,32) -> (B,32,32)
        rho = _kron_sysbatch_anc(rho_code.astype(complex), ANC_0_J)
        rho = run_f(rho, params)
        rho_a = _ptrace_system(rho)
        rho = _kron_batch_sys_anc(FRESH_00000_J, rho_a)
        if noisy_reset and p_reset > 0:
            for q in range(N_CODE):
                rho = depol1(rho, q, p_reset)
        rho = run_b(rho, params)
        return _ptrace_ancilla(rho)

    qae_batch = jax.jit(_fwd)

    def _ev(A, out): return jnp.real(jnp.einsum('ij,bji->b', A, out))

    # ---- SPECTRAL LOSS -- the only training objective ----------------------
    @jax.jit
    def loss_and_grad(params, rho_b, tgt_b, ws, wz, wx, wy):
        def _loss(p):
            out = _fwd(p, rho_b)
            s1 = _ev(S1j, out); s2 = _ev(S2j, out)
            s3 = _ev(S3j, out); s4 = _ev(S4j, out)
            zl = _ev(ZLj, out); xl = _ev(XLj, out); yl = _ev(YLj, out)
            per = (ws*(-(s1+s2+s3+s4)) + wz*(tgt_b[:, 0]-zl)**2
                   + wx*(tgt_b[:, 1]-xl)**2 + wy*(tgt_b[:, 2]-yl)**2)
            return jnp.mean(per) + _pen(p)
        return jax.value_and_grad(_loss)(params)

    # ---- PROTOCOL METRICS -- EVALUATION ONLY, never differentiated ---------
    @jax.jit
    def protocol_metrics(params, rho_half_b, p_i, V_b, rho_clean_b, tgt_b):
        out   = _fwd(params, rho_half_b)          # QAE (with internal noise)
        out2  = depol_code_j(out, p_i)            # second half interval
        P     = _final_qec_fidelity_j(out2, V_b)  # final perfect QEC
        f_raw = jnp.real(jnp.einsum('bij,bji->b', out, rho_clean_b))
        s1 = _ev(S1j, out); s2 = _ev(S2j, out)
        s3 = _ev(S3j, out); s4 = _ev(S4j, out)
        zl = _ev(ZLj, out); xl = _ev(XLj, out); yl = _ev(YLj, out)
        tr = jnp.real(jnp.einsum('bii->b', out))
        return (P, f_raw, (s1+s2+s3+s4)/4.0,
                jnp.abs(tgt_b[:, 0]-zl), jnp.abs(tgt_b[:, 1]-xl),
                jnp.abs(tgt_b[:, 2]-yl), tr)

    nr, nc, nrz = gate_counts(ops_f)
    ratio = (p1q/p2q) if p2q > 0 else P1Q_RATIO
    C = effective_C(2*nc, 2*nr, 2*nrz, ratio, noisy_rz)
    if verbose:
        tag = "" if pstat is None else \
            f" | PRUNED blocks {pstat['n_blocks_kept']}/{pstat['n_blocks']}"
        print(f"    L={n_layers}: {count_params(n_layers)} params | "
              f"full QAE {2*nc} CNOT + {2*nr} rot ({2*nrz} RZ) | "
              f"C_eff={C:.0f} -> lambda({p2q:.0e})={np.exp(-C*p2q):.4f}{tag}")
    return dict(qae_batch=qae_batch, loss_and_grad=loss_and_grad,
                protocol_metrics=protocol_metrics,
                n_params=count_params(n_layers), n_layers=n_layers,
                p1q=p1q, p2q=p2q, n_rot_half=nr, n_cnot_half=nc,
                n_rz_half=nrz, C_eff=C, prune=pstat)

# ================================================================
# 12. Chunked protocol evaluation
# ================================================================
_PM_KEYS = ['P', 'f_raw', 'stab', 'dzl', 'dxl', 'dyl', 'trace']

def eval_protocol_chunked(pm_fn, params, D, chunk=100):
    n = D['rho_half'].shape[0]
    out = {k: np.empty(n) for k in _PM_KEYS}
    p_i = jnp.float64(D['p_i'])
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        res = pm_fn(params, D['rho_half'][sl], p_i, D['V'][sl],
                    D['rho_clean'][sl], D['tgt'][sl])
        for name, v in zip(_PM_KEYS, res):
            out[name][sl] = np.asarray(v)
    return out

# ================================================================
# 13. Parameter conversion (CRX,CRY,CRZ) -> ZYZ
# ================================================================
def _rx(t): return np.array([[np.cos(t/2), -1j*np.sin(t/2)],
                             [-1j*np.sin(t/2), np.cos(t/2)]], dtype=complex)
def _ry(t): return np.array([[np.cos(t/2), -np.sin(t/2)],
                             [np.sin(t/2),  np.cos(t/2)]], dtype=complex)
def _rz(t): return np.array([[np.exp(-1j*t/2), 0],
                             [0, np.exp(1j*t/2)]], dtype=complex)

def zyz_from_su2(U, tol=1e-10):
    U = np.asarray(U, dtype=complex)
    c, s = abs(U[0, 0]), abs(U[1, 0])
    gamma = 2*np.arctan2(s, c)
    if c > tol and s > tol:
        ap = np.angle(U[1, 1]); am = np.angle(U[1, 0])
        beta, delta = ap + am, ap - am
    elif c <= tol:
        beta, delta = 2*np.angle(U[1, 0]), 0.0
    else:
        beta, delta = 2*np.angle(U[1, 1]), 0.0
    err = np.max(np.abs(_rz(beta) @ _ry(gamma) @ _rz(delta) - U))
    assert err < 1e-10, f"ZYZ extraction failed (err={err:.2e})"
    return float(beta), float(gamma), float(delta)

def convert_params_to_zyz(params_old, n_layers, n_total=N_TOTAL):
    p = np.asarray(params_old, dtype=float)
    out = p.copy(); idx = 2*n_total
    for _ in range(n_layers):
        for c in range(n_total):
            for t in range(n_total):
                if c == t: continue
                t1, t2, t3 = p[idx], p[idx+1], p[idx+2]
                out[idx:idx+3] = zyz_from_su2(_rz(t3) @ _ry(t2) @ _rx(t1))
                idx += 3
        idx += 2*n_total
    return out

def _cnot_np(c, t, n_total=N_TOTAL):
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    return _embed({c: P0}, n_total) + _embed({c: P1, t: X_}, n_total)

def schedule_unitary(params, ops):
    U = np.eye(TOT_DIM, dtype=complex)
    for op in ops:
        if op[0] == "ROT":
            _, ax, q, terms = op
            th = float(sum(co*params[i] for i, co in terms))
            G = np.cos(th/2)*np.eye(TOT_DIM) - 1j*np.sin(th/2)*_embed(
                {q: (Y_ if ax == "Y" else Z_)})
        else:
            G = _cnot_np(op[1], op[2])
        U = G @ U
    return U

def validate_decomposition(params_old, n_layers, verbose=True):
    p_new = convert_params_to_zyz(params_old, n_layers)
    try:
        import pennylane as qml
    except Exception:
        print("  [validate] WARNING: PennyLane missing -> check SKIPPED")
        return 0.0, p_new

    def circ(pr):
        idx = 0
        for q in range(N_TOTAL):
            qml.RY(pr[idx], wires=q); idx += 1
            qml.RZ(pr[idx], wires=q); idx += 1
        for _ in range(n_layers):
            for c in range(N_TOTAL):
                for t in range(N_TOTAL):
                    if c != t:
                        qml.CRX(pr[idx], wires=[c, t]); idx += 1
                        qml.CRY(pr[idx], wires=[c, t]); idx += 1
                        qml.CRZ(pr[idx], wires=[c, t]); idx += 1
            for q in range(N_TOTAL):
                qml.RY(pr[idx], wires=q); idx += 1
                qml.RZ(pr[idx], wires=q); idx += 1
    V_ref = np.asarray(qml.matrix(circ, wire_order=range(N_TOTAL))(
        np.asarray(params_old)))
    V_new = schedule_unitary(p_new, build_schedule(n_layers))
    err = float(np.max(np.abs(V_ref - V_new)))
    if verbose:
        print(f"  [validate] L={n_layers}: max abs dev "
              f"(decomposed vs pennylane) = {err:.3e}")
    return err, p_new

# ================================================================
# 14. Adaptive weights for the SPECTRAL loss
# ================================================================
W_STAB_HIGH, W_STAB_LOW = 5.0, 0.5
W_ZL_LOW, W_ZL_HIGH = 5.0, 20.0
W_XL_LOW, W_XL_HIGH = 5.0, 20.0
W_YL_LOW, W_YL_HIGH = 5.0, 20.0
STAB_SAT_LOW, STAB_SAT_HIGH = 0.70, 0.99
MAX_W_DELTA = 1.0
W_STAB_BASE, W_ZL_BASE, W_XL_BASE, W_YL_BASE = 1.0, 10.0, 10.0, 10.0

def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab - STAB_SAT_LOW)/(STAB_SAT_HIGH - STAB_SAT_LOW),
                        0, 1))
    return (float(W_STAB_HIGH - (W_STAB_HIGH - W_STAB_LOW)*sat),
            float(W_ZL_LOW + (W_ZL_HIGH - W_ZL_LOW)*sat),
            float(W_XL_LOW + (W_XL_HIGH - W_XL_LOW)*sat),
            float(W_YL_LOW + (W_YL_HIGH - W_YL_LOW)*sat))

def step_weight(cur, tgt):
    return cur + float(np.clip(tgt - cur, -MAX_W_DELTA, MAX_W_DELTA))

def baseline_loss_from_metrics(stab, dzl, dxl, dyl):
    """Spectral loss at the fixed BASE weights -- comparable across epochs."""
    return (W_STAB_BASE*(-stab) + W_ZL_BASE*dzl**2
            + W_XL_BASE*dxl**2 + W_YL_BASE*dyl**2)

# ================================================================
# 15. Load noiseless LAYERWISE reference + layer growth
# ================================================================
NOISELESS_PEXT_NPZ    = "../perfect_code/data/perfect_fidelity_vs_p_layerwise.npz"
NOISELESS_RESULTS_NPZ = "../perfect_code/data/perfect_results_layerwise.npz"

def load_noiseless_reference(pext_npz=NOISELESS_PEXT_NPZ,
                             results_npz=NOISELESS_RESULTS_NPZ,
                             l_override=None):
    ds = np.load(pext_npz,    allow_pickle=True)
    dr = np.load(results_npz, allow_pickle=True)

    layers_avail = [int(x) for x in np.asarray(dr['meta_layers'])]
    L = int(l_override) if l_override is not None else int(L_TARGET)
    if L not in layers_avail:
        raise SystemExit(f"ABORT: requested L={L} not in noiseless npz "
                         f"(available: {layers_avail})")
    miss = [k for k in LAYERWISE_PATH if k not in layers_avail]
    if miss:
        print(f"  WARNING: LAYERWISE_PATH needs L={miss}, not in npz")

    params_all = {k: np.asarray(dr[f'L{k}_params'], float) for k in layers_avail}
    for k, p in params_all.items():
        assert len(p) == count_params(k), f"L={k} param count mismatch"

    best_L = int(ds['best_n_layers'])
    ref = dict(layer_sweep=layers_avail, best_n_layers=best_L, n_layers=L,
               p_ext=np.asarray(ds['p_eval'], float),      # <- p_i grid
               f_noiseless_qae=np.asarray(ds['f_qae'], float),
               f_noiseless_qae_std=np.asarray(ds['f_qae_std'], float),
               f_qae_is_target_L=(L == best_L),
               f_raw_nocorr_stored=np.asarray(ds['f_noisy'], float),
               f_stdqec_1round=np.asarray(ds['f_std'], float),
               params_old=params_all[L], params_old_all=params_all,
               f_test_at_train_pext={
                   k: float(np.asarray(dr[f'L{k}_f_qae_test']).mean())
                   for k in layers_avail},
               layerwise=bool(int(dr['meta_layerwise'])))
    assert best_L == int(dr['best_n_layers']), "best_L mismatch between npz"
    if not ref['layerwise']:
        print("  WARNING: results npz has meta_layerwise=0 (multistart file!)")
    return ref

def grow_params(p_prev, prev_L, L):
    """Identity growth.  NOTE: the padded C-U blocks are EXACT identities and
    would still cost 2 noisy CNOTs each -> use prune_params to remove them."""
    extra = count_params(L) - count_params(prev_L)
    assert extra >= 0, "cannot shrink"
    return np.concatenate([np.asarray(p_prev, float), np.zeros(extra)])