# ============================================================================
# qae_noisy_common.py  
#
# Phase-flip [[3,1,3]] code, correlated Gaussian dephasing (external),
# gate-level depolarizing noise (internal), C-U decomposition.
#
# PROTOCOL (Locher et al., Fig. 10):
#   (a)  |psi_L> -> N_{sigma_i} -> noisy QAE -> N_{sigma_i} -> perfect QEC
#   (b)  |psi_L> -> N_{sigma_i} ->  nothing  -> N_{sigma_i} -> perfect QEC
#   (c)  |psi>   -> N_{sigma_i} ->  nothing  -> N_{sigma_i}      (bare qubit)
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
N_CODE, N_ANC = 3, 1
N_TOTAL = N_CODE + N_ANC          # 4
SYS_DIM = 2 ** N_CODE             # 8
ANC_DIM = 2 ** N_ANC              # 2
TOT_DIM = 2 ** N_TOTAL            # 16
VEC_DIM = TOT_DIM * TOT_DIM       # 256

# ------------------------------------------- external noise (PER HALF INTERVAL)
SIGMA_TRAIN = 0.5                 # sigma_i the noiseless model was trained at
RHO_CORR    = 0.3
MU_COHERENT = 0.3

# ------------------------------------------------- internal-noise parameters
P2Q_SWEEP = [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
P1Q_RATIO = 0.1                   # p1q = P1Q_RATIO * p2q
NOISY_RZ  = True
NOISY_RESET = False
P_RESET     = 0.0

SIGMA_SWEEP_DEFAULT = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]

# ------------------------------------------------------------ circuit depth
L_TARGET       = 2
LAYERWISE_PATH = list(range(1, L_TARGET + 1))

# ------------------------------------------------------------ gate pruning
# tolerances are on the OPERATOR-NORM deviation from the identity,
#   rotation : dev = 2|sin(theta/4)| ~ |theta|/2
#   C-U block: dev = ||U - I||_2     ~ sqrt(gamma^2 + (beta+delta)^2)/2
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

S1_mat = pauli_string('XXI'); S2_mat = pauli_string('IXX')
XL_mat = pauli_string('XXX')
ZL_mat = pauli_string('ZZZ')
YL_mat = pauli_string('YYY')

S1j = jnp.array(S1_mat); S2j = jnp.array(S2_mat)
ZLj = jnp.array(ZL_mat); XLj = jnp.array(XL_mat); YLj = jnp.array(YL_mat)

ket_p = np.array([1, 1],  dtype=complex) / np.sqrt(2)
ket_m = np.array([1, -1], dtype=complex) / np.sqrt(2)
ket_0L = kron_list([ket_p, ket_p, ket_p]).flatten()
ket_1L = kron_list([ket_m, ket_m, ket_m]).flatten()

def logical_state(theta, phi):
    return np.cos(theta/2)*ket_0L + np.exp(1j*phi)*np.sin(theta/2)*ket_1L

def state_to_dm(psi):
    psi = psi.reshape(-1, 1); return psi @ psi.conj().T

ANC_0_J     = jnp.zeros((ANC_DIM, ANC_DIM), dtype=complex).at[0, 0].set(1.0)
FRESH_000_J = jnp.zeros((SYS_DIM, SYS_DIM), dtype=complex).at[0, 0].set(1.0)

# ================================================================
# 2. External noise: exact correlated Gaussian dephasing (ONE half interval)
# ================================================================
_zvecs = np.array([[1 - 2*((b >> (N_CODE-1-i)) & 1) for i in range(N_CODE)]
                   for b in range(SYS_DIM)], dtype=float)

def build_phi(sigma, rho_corr=RHO_CORR, mu=MU_COHERENT, n=N_CODE, zvecs=None):
    zv = _zvecs if zvecs is None else zvecs
    dim = 2 ** n
    M     = np.eye(n)*(1-rho_corr) + rho_corr*np.ones((n, n))
    Sigma = (sigma**2)*M
    mu_v  = np.full(n, mu)
    Phi = np.empty((dim, dim), dtype=complex)
    for a in range(dim):
        for b in range(dim):
            c = (zv[a] - zv[b]) / 2.0
            Phi[a, b] = np.exp(-1j*(mu_v @ c) - 0.5*(c @ Sigma @ c))
    return Phi

def dephase_np(rho, Phi):
    """rho (...,8,8) -> Phi * rho  (elementwise)."""
    return Phi * rho

def validate_phi_composition(sigma=0.37, verbose=True):
    """Two half intervals of sigma_i == one interval of (sqrt2 sigma_i, 2 mu)."""
    P1 = build_phi(sigma)
    P2 = build_phi(sigma*np.sqrt(2.0), RHO_CORR, 2.0*MU_COHERENT)
    err = float(np.max(np.abs(P1*P1 - P2)))
    if verbose:
        print(f"  [validate] Phi(s)^2 == Phi(sqrt2 s, 2mu): max dev {err:.2e}")
    assert err < 1e-12, "dephasing composition wrong"
    return err

def fibonacci_sphere(n):
    golden = (1 + np.sqrt(5)) / 2
    return [(np.arccos(1 - 2*(i+0.5)/n), (2*np.pi*i/golden) % (2*np.pi))
            for i in range(n)]

# ================================================================
# 3. FINAL PERFECT QEC (the measuring instrument)
# ================================================================
CORRECTABLE_LABELS = ['III', 'ZII', 'IZI', 'IIZ']
CORR_OPS   = np.stack([pauli_string(s) for s in CORRECTABLE_LABELS])   # (4,8,8)
N_CORR     = CORR_OPS.shape[0]

def recovery_vectors(psi_batch):
    """psi (B,8) -> V (B,4,8) with V[b,s] = E_s |psi_b>."""
    return np.einsum('sij,bj->bsi', CORR_OPS, np.asarray(psi_batch))

def final_qec_fidelity_np(rho, V):
    """rho (B,8,8), V (B,4,8) -> P (B,) = Tr[Pi rho]."""
    return np.real(np.einsum('bsi,bij,bsj->b', V.conj(), rho, V))

def _final_qec_fidelity_j(rho, V):
    return jnp.real(jnp.einsum('bsi,bij,bsj->b', jnp.conj(V), rho, V))

def validate_recovery(n_states=32, verbose=True):
    """Pi is a rank-4 projector; every single Z error is recovered exactly."""
    worst_proj, worst_corr, worst_noerr = 0.0, 0.0, 0.0
    for th, ph in fibonacci_sphere(n_states):
        psi = logical_state(th, ph)
        V   = recovery_vectors(psi[None])[0]                    # (4,8)
        Pi  = np.einsum('si,sj->ij', V, V.conj())
        worst_proj = max(worst_proj,
                         float(np.max(np.abs(Pi @ Pi - Pi))),
                         float(abs(np.trace(Pi).real - N_CORR)),
                         float(np.max(np.abs(Pi - Pi.conj().T))))
        r0 = state_to_dm(psi)
        worst_noerr = max(worst_noerr, abs(np.real(np.trace(Pi @ r0)) - 1.0))
        for q in range(N_CODE):
            E = pauli_string(''.join('Z' if k == q else 'I'
                                     for k in range(N_CODE)))
            worst_corr = max(worst_corr,
                             abs(np.real(np.trace(Pi @ E @ r0 @ E)) - 1.0))
    if verbose:
        print(f"  [validate] Pi projector/rank : {worst_proj:.2e}")
        print(f"  [validate] P(no error)   = 1 : {worst_noerr:.2e}")
        print(f"  [validate] P(single Z)   = 1 : {worst_corr:.2e}")
    assert max(worst_proj, worst_corr, worst_noerr) < 1e-10, \
        "final perfect-QEC recovery is wrong"
    return worst_proj

# ================================================================
# 4. Baseline (c): bare physical qubit, TWO half intervals
# ================================================================
def P_bare_physical(sigma_i, mu=MU_COHERENT):
    c = np.exp(-np.asarray(sigma_i, float)**2) * np.cos(2.0*mu)
    return 0.5 + (2.0*c + 1.0) / 6.0

def validate_bare_physical(sigma_i=0.37, n_states=2000, verbose=True):
    zv1 = np.array([[1.0], [-1.0]])
    Phi1 = build_phi(sigma_i, 0.0, MU_COHERENT, n=1, zvecs=zv1)
    Phi2 = Phi1 * Phi1
    fs = []
    for th, ph in fibonacci_sphere(n_states):
        v = np.array([np.cos(th/2), np.exp(1j*ph)*np.sin(th/2)], dtype=complex)
        r = np.outer(v, v.conj())
        fs.append(float(np.real(v.conj() @ (Phi2 * r) @ v)))
    num, ana = float(np.mean(fs)), float(P_bare_physical(sigma_i))
    if verbose:
        print(f"  [validate] bare qubit @sigma_i={sigma_i}: "
              f"numeric {num:.10f} vs analytic {ana:.10f} "
              f"(dev {abs(num-ana):.2e})")
    assert abs(num - ana) < 5e-4, "bare-qubit baseline mismatch"
    return abs(num - ana)

# ================================================================
# 5. Protocol dataset
# ================================================================
def generate_protocol_dataset(n_states, sigma_i,
                              rho_corr=RHO_CORR, mu=MU_COHERENT):
    Phi  = build_phi(sigma_i, rho_corr, mu)
    Phi2 = Phi * Phi
    PSI, RC, RH, TG = [], [], [], []
    for th, ph in fibonacci_sphere(n_states):
        psi = logical_state(th, ph); r = state_to_dm(psi)
        PSI.append(psi); RC.append(r); RH.append(dephase_np(r, Phi))
        TG.append([float(np.real(np.trace(ZL_mat @ r))),
                   float(np.real(np.trace(XL_mat @ r))),
                   float(np.real(np.trace(YL_mat @ r)))])
    PSI = np.stack(PSI); RC = np.stack(RC); RH = np.stack(RH); TG = np.array(TG)
    V   = recovery_vectors(PSI)
    p_un = final_qec_fidelity_np(Phi2[None]*RC, V)
    f_raw = np.real(np.einsum('bij,bji->b', RH, RC))
    return dict(psi=PSI,
                rho_clean=jnp.array(RC), rho_half=jnp.array(RH),
                tgt=jnp.array(TG), V=jnp.array(V), phi=jnp.array(Phi),
                p_uncorr=p_un, f_raw_nocorr=f_raw,
                sigma_i=float(sigma_i), n=int(n_states))

def generate_dataset(n_states, sigma, rho_corr=RHO_CORR, mu=MU_COHERENT):
    D = generate_protocol_dataset(n_states, sigma, rho_corr, mu)
    return (np.asarray(D['rho_clean']), np.asarray(D['rho_half']),
            np.asarray(D['tgt']))

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
                ops.append(("ROT", "Z", t, ((d,  0.5), (b, -0.5))))   # C
                ops.append(("CNOT", c, t))
                ops.append(("ROT", "Z", t, ((d, -0.5), (b, -0.5))))   # B
                ops.append(("ROT", "Y", t, ((g, -0.5),)))             # B
                ops.append(("CNOT", c, t))
                ops.append(("ROT", "Y", t, ((g,  0.5),)))             # A
                ops.append(("ROT", "Z", t, ((b,  1.0),)))             # A
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
    """lambda = exp(-C p2q) damping coefficient of the full QAE."""
    n_noisy_rot = n_rot_full if noisy_rz else (n_rot_full - n_rz_full)
    return (16.0/15.0)*n_cnot_full + (4.0/3.0)*p1q_ratio*n_noisy_rot

# ================================================================
# 7. Embedded operators on the 4-qubit register (qubit 0 = most significant)
# ================================================================
def embed(pairs, n_total=N_TOTAL):
    return kron_list([pairs.get(q, I2) for q in range(n_total)])

EMB_Y = {q: jnp.array(embed({q: Y_})) for q in range(N_TOTAL)}
EMB_Z = {q: jnp.array(embed({q: Z_})) for q in range(N_TOTAL)}
EYE_J = jnp.eye(TOT_DIM, dtype=complex)

def _cnot_np(c, t, n_total=N_TOTAL):
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)
    return embed({c: P0}, n_total) + embed({c: P1, t: X_}, n_total)

CNOT_J = {(c, t): jnp.array(_cnot_np(c, t))
          for c in range(N_TOTAL) for t in range(N_TOTAL) if c != t}

def rot_mat(axis, q, theta):
    P = EMB_Y[q] if axis == "Y" else EMB_Z[q]
    return jnp.cos(theta/2)*EYE_J - 1j*jnp.sin(theta/2)*P

# ================================================================
# 8. Internal noise: exact depolarizing Kraus -> Liouville superoperators
# ================================================================
def _sup_from_kraus(Ks):
    return sum(np.kron(K, K.conj()) for K in Ks)

def kraus_depol1(q, p):
    Ks = [np.sqrt(1-p)*np.eye(TOT_DIM, dtype=complex)]
    Ks += [np.sqrt(p/3)*embed({q: P}) for P in (X_, Y_, Z_)]
    return Ks

def kraus_depol2(c, t, p):
    Ks = [np.sqrt(1-p)*np.eye(TOT_DIM, dtype=complex)]
    P4 = [I2, X_, Y_, Z_]
    for i, A in enumerate(P4):
        for j, B in enumerate(P4):
            if i == 0 and j == 0: continue
            Ks.append(np.sqrt(p/15)*embed({c: A, t: B}))
    return Ks

def _check_tp(S, tag):
    T = np.asarray(S).reshape(TOT_DIM, TOT_DIM, TOT_DIM, TOT_DIM)
    err = np.max(np.abs(np.einsum('iijk->jk', T) - np.eye(TOT_DIM)))
    assert err < 1e-11, f"{tag}: channel not trace preserving ({err:.2e})"

def build_noise_superops(p1q, p2q, verbose=False):
    sup1 = None; sup2 = None
    if p1q > 0:
        sup1 = {}
        for q in range(N_TOTAL):
            S = _sup_from_kraus(kraus_depol1(q, p1q)); _check_tp(S, f"1q q{q}")
            sup1[q] = jnp.array(S)
    if p2q > 0:
        sup2 = {}
        for (c, t) in CNOT_J:
            S = _sup_from_kraus(kraus_depol2(c, t, p2q)); _check_tp(S, f"2q {c}{t}")
            sup2[(c, t)] = jnp.array(S)
    if verbose:
        n1 = 0 if sup1 is None else len(sup1)
        n2 = 0 if sup2 is None else len(sup2)
        print(f"    superops built: {n1} x 1q, {n2} x 2q  "
              f"({(n1+n2)*VEC_DIM*VEC_DIM*16/1e6:.1f} MB)")
    return sup1, sup2

def apply_sup(rho, S):
    B = rho.shape[0]
    return (rho.reshape(B, VEC_DIM) @ S.T).reshape(B, TOT_DIM, TOT_DIM)

# ================================================================
# 9. GATE PRUNING -- value-dependent schedule
# ================================================================
def _dev_from_trace(c):
    """||U - I||_2 for U in SU(2) with Re(tr U)/2 = c.  The eigenvalues of U
    are exp(-+ i w/2) with cos(w/2) = c, hence ||U - I||_2 = sqrt(2 - 2c)."""
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.sqrt(max(0.0, 2.0 - 2.0*c)))

def rot_dev(theta):
    """R_ax(theta) = cos(t/2) I - i sin(t/2) P  ->  dev = 2|sin(theta/4)|.
    Exactly 0 iff theta = 0 mod 4pi (NOT mod 2pi: R(2pi) = -I)."""
    return _dev_from_trace(np.cos(float(theta)/2.0))

def block_dev(beta, gamma, delta):
    """||C-U - I||_2 with U = RZ(beta) RY(gamma) RZ(delta).  The |0><0| branch
    is exactly I, so this equals ||U - I||_2 with
    Re(tr U)/2 = cos(gamma/2) cos((beta+delta)/2).
    The control makes the global phase OBSERVABLE -> compared with +I only,
    so U = -I is (correctly) NOT prunable."""
    return _dev_from_trace(np.cos(float(gamma)/2.0) *
                           np.cos((float(beta) + float(delta))/2.0))

def block_param_idx(n_layers, n_total=N_TOTAL):
    """(n_blocks, 3) index array of the (beta,gamma,delta) triples."""
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
    """Fixed-point peephole.  EXACT: CNOT^2 = I and R(t1)R(t2) = R(t1+t2) on
    the same axis/qubit.  TOLERANCE-BASED: drop R with dev < tol_rot."""
    changed = True
    while changed:
        changed = False; out = []; i = 0; n = len(ops)
        while i < n:
            a = ops[i]; b = ops[i+1] if i+1 < n else None
            if (b is not None and a[0] == "CNOT" and b[0] == "CNOT"
                    and a[1] == b[1] and a[2] == b[2]):
                i += 2; changed = True; continue                      # exact
            if (b is not None and a[0] == "ROT" and b[0] == "ROT"
                    and a[1] == b[1] and a[2] == b[2]):
                out.append(("ROT", a[1], a[2], tuple(a[3]) + tuple(b[3])))
                i += 2; changed = True; continue                      # exact
            if a[0] == "ROT" and rot_dev(_theta_of(p, a[3])) < tol_rot:
                i += 1; changed = True; continue                      # tol
            out.append(a); i += 1
        ops = out
    return ops

def build_pruned_schedule(params, n_layers, n_total=N_TOTAL,
                          tol_rot=PRUNE_TOL_ROT, tol_block=PRUNE_TOL_BLOCK,
                          simplify=True):
    """Same layout as build_schedule(), but gates that act as the identity for
    THESE parameter values are removed.  Dropping a C-U block also drops its
    TWO CNOTs -- that is the internal-noise saving."""
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
                    continue                     # <- kills 2 CNOTs + 5 rots
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
    """Zero every parameter that NO surviving gate references, so that the
    saved vector and the pruned circuit agree (and pruning is idempotent).
    Parameters still used by a surviving gate are never touched -- beta and
    gamma each appear in several gates of one block."""
    p = np.asarray(params, float).ravel().copy()
    used = {i for o in ops if o[0] == "ROT" for i, _ in o[3]}
    mask = np.ones(p.size, bool); mask[sorted(used)] = False
    p[mask] = 0.0
    return p, int(mask.sum())

def validate_pruning(params, n_layers, tol_rot=PRUNE_TOL_ROT,
                     tol_block=PRUNE_TOL_BLOCK, verbose=True):
    """Returns (unitary deviation, pruned ops, stats, sparsified params).
    The deviation compares the DEPLOYED circuit (pruned ops + sparsified
    params) against the INTENDED one (full ops + original params)."""
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
# 10. Noisy QAE forward map + PROTOCOL metrics / losses
# ================================================================
def _apply_ops(rho, params, ops, sup1, sup2, noisy_rz):
    for op in ops:
        if op[0] == "ROT":
            _, ax, q, terms = op
            th = sum(co*params[i] for i, co in terms)
            U  = rot_mat(ax, q, th)
            rho = U @ rho @ U.conj().T
            if sup1 is not None and (ax != "Z" or noisy_rz):
                rho = apply_sup(rho, sup1[q])
        else:
            _, c, t = op
            U = CNOT_J[(c, t)]
            rho = U @ rho @ U.conj().T
            if sup2 is not None:
                rho = apply_sup(rho, sup2[(c, t)])
    return rho

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

def make_noisy_qae(n_layers, p2q, p1q=None, noisy_rz=NOISY_RZ,
                   noisy_reset=NOISY_RESET, p_reset=P_RESET, verbose=False,
                   prune_params=None, prune_tol_rot=PRUNE_TOL_ROT,
                   prune_tol_block=PRUNE_TOL_BLOCK, group_l1=0.0):
    """p2q = 0 reproduces the exact noiseless unitary QAE.

    prune_params : if given, the gate schedule is PRUNED for these parameter
                   values (identity rotations and identity C-U blocks incl.
                   their 2 CNOTs are removed) and the decoder is the adjoint
                   of the ACTUAL (pruned) encoder.
    group_l1     : optional group-lasso over each (beta,gamma,delta) triple,
                   added to the training loss.  0.0 = paper-faithful.
    """
    if p1q is None: p1q = P1Q_RATIO * p2q
    if prune_params is None:
        ops_f, pstat = build_schedule(n_layers), None
    else:
        ops_f, pstat = build_pruned_schedule(prune_params, n_layers,
                                            tol_rot=prune_tol_rot,
                                            tol_block=prune_tol_block)
    ops_b = adjoint_schedule(ops_f)
    sup1, sup2 = build_noise_superops(p1q, p2q, verbose=verbose)
    sup_reset = None
    if noisy_reset and p_reset > 0:
        sup_reset = [jnp.array(_sup_from_kraus(kraus_depol1(q, p_reset)))
                     for q in range(N_CODE)]
    BLK = jnp.array(block_param_idx(n_layers)) if group_l1 > 0 else None

    def _pen(p):
        if group_l1 <= 0: return 0.0
        return group_l1*jnp.sum(jnp.sqrt(jnp.sum(p[BLK]**2, axis=1) + 1e-12))

    def _fwd(params, rho_code):                       # (B,8,8) -> (B,8,8)
        rho = _kron_sysbatch_anc(rho_code.astype(complex), ANC_0_J)
        rho = _apply_ops(rho, params, ops_f, sup1, sup2, noisy_rz)
        rho_a = _ptrace_system(rho)
        rho = _kron_batch_sys_anc(FRESH_000_J, rho_a)
        if sup_reset is not None:
            for S in sup_reset: rho = apply_sup(rho, S)
        rho = _apply_ops(rho, params, ops_b, sup1, sup2, noisy_rz)
        return _ptrace_ancilla(rho)

    qae_batch = jax.jit(_fwd)

    def _ev(A, out): return jnp.real(jnp.einsum('ij,bji->b', A, out))

    # ---- spectral loss ----
    @jax.jit
    def loss_and_grad(params, rho_b, tgt_b, ws, wz, wx, wy):
        def _loss(p):
            out = _fwd(p, rho_b)
            s1 = _ev(S1j, out); s2 = _ev(S2j, out)
            zl = _ev(ZLj, out); xl = _ev(XLj, out); yl = _ev(YLj, out)
            per = (ws*(-(s1+s2)) + wz*(tgt_b[:, 0]-zl)**2
                   + wx*(tgt_b[:, 1]-xl)**2 + wy*(tgt_b[:, 2]-yl)**2)
            return jnp.mean(per) + _pen(p)
        return jax.value_and_grad(_loss)(params)

    # ---- PROTOCOL loss: 1 - <Tr[Pi N_i(QAE(N_i(psi)))]> -------------------
    @jax.jit
    def protocol_loss_and_grad(params, rho_half_b, phi_j, V_b):
        def _loss(p):
            out  = _fwd(p, rho_half_b)
            out2 = phi_j[None] * out
            return 1.0 - jnp.mean(_final_qec_fidelity_j(out2, V_b)) + _pen(p)
        return jax.value_and_grad(_loss)(params)

    # ---- PROTOCOL metrics: everything reported, in one pass ----------------
    @jax.jit
    def protocol_metrics(params, rho_half_b, phi_j, V_b, rho_clean_b, tgt_b):
        out   = _fwd(params, rho_half_b)
        out2  = phi_j[None] * out
        P     = _final_qec_fidelity_j(out2, V_b)
        f_raw = jnp.real(jnp.einsum('bij,bji->b', out, rho_clean_b))
        s1 = _ev(S1j, out); s2 = _ev(S2j, out)
        zl = _ev(ZLj, out); xl = _ev(XLj, out); yl = _ev(YLj, out)
        tr = jnp.real(jnp.einsum('bii->b', out))
        return (P, f_raw, (s1+s2)/2.0,
                jnp.abs(tgt_b[:, 0]-zl), jnp.abs(tgt_b[:, 1]-xl),
                jnp.abs(tgt_b[:, 2]-yl), tr)

    nr, nc, nrz = gate_counts(ops_f)
    ratio = (p1q/p2q) if p2q > 0 else P1Q_RATIO
    C = effective_C(2*nc, 2*nr, 2*nrz, ratio, noisy_rz)
    if verbose:
        tag = "" if pstat is None else \
            f" | PRUNED blocks {pstat['n_blocks_kept']}/{pstat['n_blocks']}"
        print(f"    L={n_layers}: {count_params(n_layers)} params | full QAE "
              f"{2*nc} CNOT + {2*nr} rot ({2*nrz} RZ) | C_eff={C:.0f} -> "
              f"lambda({p2q:.0e})={np.exp(-C*p2q):.4f}{tag}")
    return dict(qae_batch=qae_batch, loss_and_grad=loss_and_grad,
                protocol_loss_and_grad=protocol_loss_and_grad,
                protocol_metrics=protocol_metrics,
                n_params=count_params(n_layers), n_layers=n_layers,
                p1q=p1q, p2q=p2q,
                n_rot_half=nr, n_cnot_half=nc, n_rz_half=nrz,
                C_eff=C, prune=pstat)

# ================================================================
# 11. Chunked protocol evaluation
# ================================================================
_PM_KEYS = ['P', 'f_raw', 'stab', 'dzl', 'dxl', 'dyl', 'trace']

def eval_protocol_chunked(pm_fn, params, D, chunk=100):
    n = D['rho_half'].shape[0]
    out = {k: np.empty(n) for k in _PM_KEYS}
    for k in range(int(np.ceil(n/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, n))
        res = pm_fn(params, D['rho_half'][sl], D['phi'], D['V'][sl],
                    D['rho_clean'][sl], D['tgt'][sl])
        for name, v in zip(_PM_KEYS, res):
            out[name][sl] = np.asarray(v)
    return out

# ================================================================
# 12. Parameter conversion: (CRX,CRY,CRZ) -> ZYZ (beta,gamma,delta)
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

def schedule_unitary(params, ops):
    U = np.eye(TOT_DIM, dtype=complex)
    for op in ops:
        if op[0] == "ROT":
            _, ax, q, terms = op
            th = float(sum(co*params[i] for i, co in terms))
            G = np.cos(th/2)*np.eye(TOT_DIM) - 1j*np.sin(th/2)*embed(
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
# 13. Adaptive loss weights 
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
    return (W_STAB_BASE*(-stab) + W_ZL_BASE*dzl**2
            + W_XL_BASE*dxl**2 + W_YL_BASE*dyl**2)

# ================================================================
# 14. Load noiseless LAYERWISE reference + layer growth
# ================================================================
NOISELESS_SIGMA_NPZ   = "../phaseflip_rep_code/data/dephasing_fidelity_vs_sigma_layerwise.npz"
NOISELESS_RESULTS_NPZ = "../phaseflip_rep_code/data/dephasing_results_layerwise.npz"

def _opt(d, key, default=None):
    return d[key] if key in getattr(d, "files", []) else default

def load_noiseless_reference(sigma_npz=NOISELESS_SIGMA_NPZ,
                             results_npz=NOISELESS_RESULTS_NPZ,
                             l_override=None):
    ds = np.load(sigma_npz,   allow_pickle=True)
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
    ref = dict(
        layer_sweep=layers_avail, best_n_layers=best_L, n_layers=L,
        sigma=np.asarray(ds['sigma'], float),          # <- now sigma_i grid
        f_noiseless_qae=np.asarray(ds['f_qae'], float),
        f_noiseless_qae_std=np.asarray(ds['f_qae_std'], float),
        f_qae_is_target_L=(L == best_L),
        f_raw_nocorr_stored=np.asarray(ds['f_noisy'], float),
        params_old=params_all[L], params_old_all=params_all,
        f_test_at_train_sigma={k: float(np.asarray(dr[f'L{k}_f_qae_test']).mean())
                               for k in layers_avail},
        rho_corr=float(ds['rho_corr']), mu=float(ds['mu']),
        layerwise=bool(int(dr['meta_layerwise'])))
    assert abs(ref['rho_corr'] - RHO_CORR) < 1e-12, "rho_corr mismatch"
    assert abs(ref['mu'] - MU_COHERENT) < 1e-12, "mu mismatch"
    if not ref['layerwise']:
        print("  WARNING: results npz has meta_layerwise=0 (multistart file!)")
    return ref

def grow_params(p_prev, prev_L, L):
    """Identity growth.  NOTE: the padded C-U blocks are EXACT identities and
    would still cost 2 noisy CNOTs each -> use prune_params to remove them."""
    extra = count_params(L) - count_params(prev_L)
    assert extra >= 0, "cannot shrink"
    return np.concatenate([np.asarray(p_prev, float), np.zeros(extra)])