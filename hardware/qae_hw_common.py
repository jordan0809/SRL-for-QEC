# ============================================================================
# qae_hw_common.py   -- shared library (IBM hardware / Qiskit)
# ============================================================================
"""Shared layer for the [[3,1,3]] phase-flip spectral-QAE hardware run.

EXTERNAL CHANNEL:

    N = AmplitudeDamping(gamma)  o  CorrelatedGaussianDephasing(mu, sigma, rho_c)


FOUR BASELINES, all built by build_circuit():
    'bare'    1 physical qubit
    'uncorr'  encoded, SOFTWARE ideal-recovery decode() applied afterwards
    'qae'     encoded, learned unitary mid-storage, SOFTWARE decode()
    'qec'     encoded, REAL mid-circuit syndrome measurement + conditional Z
              correction (dynamic circuit), SOFTWARE decode() applied on the
              residual afterwards for consistency with the other arms.

Requires qiskit>=1.2, qiskit-ibm-runtime>=0.30.  qiskit-aer optional."""
from __future__ import annotations
import itertools, math
from collections import defaultdict
from dataclasses import dataclass, field
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Operator
try:
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
except ImportError:
    from qiskit.transpiler import generate_preset_pass_manager

# ------------------------------------------------------------------ code
NQ, N_CODE = 4, 3
L_HW = 2                          
CODE, ANC = (0, 1, 2), 3          
AX_SET = {"Z": ("XXX",), "X": ("ZZZ",), "Y": ("YYY", "ZZY", "YZZ", "ZYZ")}
TRAIN_SET = ("XXX", "ZZZ", "YYY")
ALL_SET = ("XXX", "ZZZ", "YYY", "ZZY", "YZZ", "ZYZ")
M_S1, M_S2 = 0b011, 0b110
AXI = {"X": 0, "Y": 1, "Z": 2}
OP_OF_AX = {"Z": "XXX", "X": "ZZZ", "Y": "YYY"}


def axis_states():
    h = np.pi / 2
    return [("Z+", 0.0, 0.0, "Z", +1), ("Z-", np.pi, 0.0, "Z", -1),
            ("X+", h, 0.0, "X", +1),   ("X-", h, np.pi, "X", -1),
            ("Y+", h, h, "Y", +1),     ("Y-", h, -h, "Y", -1)]


def fib_states(k):
    ga = np.pi * (3 - np.sqrt(5))
    return [(f"f{i}", float(np.arccos(1 - 2 * (i + .5) / k)),
             float((i * ga) % (2 * np.pi)), None, 0) for i in range(k)]


# --------------------------------------------------- parities and decoder
def _sgn(n):
    d = 1 << n
    S = np.empty((d, d))
    for m in range(d):
        S[m] = [(-1.0) ** bin(m & b).count("1") for b in range(d)]
    return S
_SG = {n: _sgn(n) for n in (1, 2, 3, 4)}


def parities(probs, n):
    """probs(...,2^n) -> expectation values indexed by Z-mask (bit i = slot i)."""
    return probs @ _SG[n].T


def decode(ev):
    """Ideal majority-vote recovery on expectation values"""
    o = {}
    if "XXX" in ev:
        e = ev["XXX"]
        o["Z"] = 0.5 * (e[..., 1] + e[..., 2] + e[..., 4] - e[..., 7])
        o["S1"], o["S2"], o["XXX"] = e[..., M_S1], e[..., M_S2], e[..., 7]
    if "ZZZ" in ev:
        o["X"] = o["ZZZ"] = ev["ZZZ"][..., 7]
    if "YYY" in ev:
        o["YYY"] = ev["YYY"][..., 7]
    if all(k in ev for k in AX_SET["Y"]):
        o["Y"] = -0.5 * sum(ev[k][..., 7] for k in AX_SET["Y"])
    return o


# ------------------------------------------------------------- ansatz
DEF_OPTS = dict(anc=3, reset="code", phase=1)


def wire_slot(o):
    a = int(o["anc"]); codes = [w for w in range(NQ) if w != a]
    m = {w: CODE[i] for i, w in enumerate(codes)}
    m[a] = ANC
    return m


def walk(L, o):
    """token stream in PennyLane parameter order: ('rot',slot,g) / ('pair',c,t)."""
    m = wire_slot(o)
    rots = [("rot", m[w], g) for w in range(NQ) for g in ("y", "z")]
    tok = list(rots)
    for _ in range(L):
        tok += [("pair", m[c], m[t])
                for c in range(NQ) for t in range(NQ) if c != t]
        tok += list(rots)
    return tok


def n_params(L, o=None, mode="hw"):
    o = dict(DEF_OPTS) if o is None else o
    per = 3 if mode == "sim" else 5
    return sum(1 if t[0] == "rot" else per for t in walk(L, o))


def _rx(a): return np.array([[np.cos(a/2), -1j*np.sin(a/2)],
                            [-1j*np.sin(a/2), np.cos(a/2)]], complex)
def _ry(a): return np.array([[np.cos(a/2), -np.sin(a/2)],
                            [np.sin(a/2), np.cos(a/2)]], complex)
def _rz(a): return np.diag([np.exp(-1j*a/2), np.exp(1j*a/2)]).astype(complex)
def _rot_mat(g, a): return {"x": _rx, "y": _ry, "z": _rz}[g](a)


def _cu(qc, c, t, p):
    """controlled-SU(2), ABC form: 2 CX + 5 rotations (verified exact by [t1])."""
    qc.rz(p[4], t); qc.cx(c, t)
    qc.rz(p[2], t); qc.ry(p[3], t); qc.cx(c, t)
    qc.ry(p[1], t); qc.rz(p[0], t)


def su2_zyz(U):
    U = np.asarray(U, complex); U = U / np.sqrt(np.linalg.det(U))
    c, s = abs(U[0, 0]), abs(U[1, 0]); gam = 2 * math.atan2(s, c)
    if c > 1e-12 and s > 1e-12:
        phi = np.angle(U[1, 1]) + np.angle(U[1, 0])
        dl = np.angle(U[1, 1]) - np.angle(U[1, 0])
    elif s <= 1e-12: phi, dl = 2 * np.angle(U[1, 1]), 0.0
    else:            phi, dl = 2 * np.angle(U[1, 0]), 0.0
    return float(phi), float(gam), float(dl)


def abc_params(U):
    phi, gam, dl = su2_zyz(U)
    return np.array([phi, gam / 2, -(dl + phi) / 2, -gam / 2, (dl - phi) / 2])


def build_u(theta, L, o=None, mode="hw"):
    o = dict(DEF_OPTS) if o is None else o
    qc, i = QuantumCircuit(NQ), 0
    for t in walk(L, o):
        if t[0] == "rot":
            getattr(qc, "r" + t[2])(theta[i], t[1]); i += 1
        else:
            _, c, tq = t
            if mode == "sim":
                qc.crx(theta[i], c, tq); qc.cry(theta[i+1], c, tq)
                qc.crz(theta[i+2], c, tq); i += 3
            else:
                _cu(qc, c, tq, theta[i:i + 5]); i += 5
    return qc


def convert_sim_params(sim, L, o=None):
    """PennyLane angles -> ABC angles. CRX,CRY,CRZ in circuit order means the
    target unitary is U = Rz(c)Ry(b)Rx(a); det U = 1 so controlled-U is exact.
    96 logical angles -> 144 physical angles, exactly and one-way."""
    o = dict(DEF_OPTS) if o is None else o
    sim = np.asarray(sim, float).ravel()
    need = n_params(L, o, "sim")
    if sim.size != need:
        raise ValueError(f"parameter vector has {sim.size} entries, "
                         f"L={L} needs {need}")
    out, i = [], 0
    for t in walk(L, o):
        if t[0] == "rot":
            out.append(sim[i]); i += 1
        else:
            U = _rz(sim[i+2]) @ _ry(sim[i+1]) @ _rx(sim[i]); i += 3
            out += list(abc_params(U))
    return np.array(out)


def check_conversion(sim, L, o=None):
    """||U(sim) - U(abc)||_max"""
    o = dict(DEF_OPTS) if o is None else o
    A = Operator(build_u(np.asarray(sim, float), L, o, "sim")).data
    B = Operator(build_u(convert_sim_params(sim, L, o), L, o, "hw")).data
    ph = np.vdot(A.ravel(), B.ravel())
    ph = ph / abs(ph) if abs(ph) > 1e-12 else 1.0
    return float(np.abs(A - B / ph).max())


# --------------------------------------- exact density-matrix reference
_P1 = {"I": np.eye(2, dtype=complex), "X": np.array([[0, 1], [1, 0]], complex),
       "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]).astype(complex)}
PAR8 = np.array([(-1.0) ** bin(a).count("1") for a in range(8)])


def pauli_op(label):
    """label[i] acts on slot i; slot 0 is the least significant index."""
    M = np.eye(1, dtype=complex)
    for ch in label:
        M = np.kron(_P1[ch], M)
    return M


def coh_mask(n, mu, sigma, rhoc):
    """Phi_ab = exp(-i mu sum c - 1/2 c^T Sigma c),  c_j = b_j - a_j."""
    d = 1 << n
    bits = ((np.arange(d)[:, None] >> np.arange(n)) & 1).astype(float)
    c = bits[None, :, :] - bits[:, None, :]
    S = sigma ** 2 * ((1 - rhoc) * np.eye(n) + rhoc * np.ones((n, n)))
    return np.exp(-1j * mu * c.sum(-1) - 0.5 * np.einsum("abi,ij,abj->ab", c, S, c))


# ---------------- amplitude damping and the combined external channel ----
_SM2 = np.array([[0.0, 1.0], [0.0, 0.0]], complex)     # |0><1|
_P12 = np.array([[0.0, 0.0], [0.0, 1.0]], complex)     # |1><1|
_AD_CACHE: dict = {}


def _emb1(q, n, M2):
    """embed a 1-qubit operator on slot q; slot 0 = least significant index
    (identical convention to pauli_op)."""
    A = np.eye(1, dtype=complex)
    for j in range(n):
        A = np.kron(M2 if j == q else np.eye(2, dtype=complex), A)
    return A


def _ad_ops(n, gamma):
    key = (int(n), round(float(gamma), 12))
    if key not in _AD_CACHE:
        s = math.sqrt(max(1.0 - gamma, 0.0)); sg = math.sqrt(max(gamma, 0.0))
        I = np.eye(1 << n, dtype=complex)
        _AD_CACHE[key] = [(I - (1.0 - s) * _emb1(q, n, _P12),
                           sg * _emb1(q, n, _SM2)) for q in range(n)]
    return _AD_CACHE[key]


def ad_channel(rho, gamma, n=None):
    """Independent amplitude damping of rate gamma on every one of n qubits.
    K0 = |0><0| + sqrt(1-g)|1><1| ,  K1 = sqrt(g)|0><1| , per qubit.
    Accepts a single (d,d) matrix or a stack (...,d,d)."""
    if gamma is None or gamma <= 0.0:
        return np.asarray(rho, complex)
    r = np.asarray(rho, complex)
    d = r.shape[-1]
    n = int(round(math.log2(d))) if n is None else int(n)
    for K0, K1 in _ad_ops(n, gamma):
        r = K0 @ r @ K0.conj().T + K1 @ r @ K1.conj().T
    return r


def noise_channel(rho, mu, sigma, rhoc, gamma=0.0, n=3):
    """ONE external interval:  N = AD(gamma) o Dephase(mu, sigma, rho_c)."""
    return ad_channel(np.asarray(rho, complex) * coh_mask(n, mu, sigma, rhoc),
                      gamma, n)


def code_amp(th, ph, phase=1):
    p = np.ones(8) / np.sqrt(8); m = PAR8 / np.sqrt(8)
    return np.cos(th / 2) * p + np.exp(1j * phase * ph) * np.sin(th / 2) * m


def code_state(th, ph, phase=1):
    v = code_amp(th, ph, phase)
    return np.outer(v, v.conj())


def qae_map(rho8, U16, o=None, form="selfinv"):
    o = dict(DEF_OPTS) if o is None else o
    full = np.zeros((16, 16), complex); full[:8, :8] = rho8      # ancilla |0>
    s = U16 @ full @ U16.conj().T
    if form == "single":
        return s[:8, :8] + s[8:, 8:]
    if o["reset"] == "code":                    # latent = ancilla (your case)
        rl = np.array([[np.trace(s[0:8, 0:8]),  np.trace(s[0:8, 8:16])],
                       [np.trace(s[8:16, 0:8]), np.trace(s[8:16, 16:16] if False else s[8:16, 8:16])]])
        t = np.zeros((16, 16), complex)
        t[0, 0], t[0, 8], t[8, 0], t[8, 8] = rl[0, 0], rl[0, 1], rl[1, 0], rl[1, 1]
    else:                                       # latent = 3 code qubits
        t = np.zeros((16, 16), complex)
        t[:8, :8] = s[:8, :8] + s[8:, 8:]
    w = U16.conj().T @ t @ U16
    return w[:8, :8] + w[8:, 8:]


def map_superop(U16, o=None, form="selfinv"):
    S = np.zeros((64, 64), complex)
    for k in range(64):
        E = np.zeros((8, 8), complex); E[k // 8, k % 8] = 1.0
        S[:, k] = qae_map(E, U16, o, form).reshape(64)
    return S


def apply_superop(S, rho8):
    return (S @ np.asarray(rho8, complex).reshape(64)).reshape(8, 8)


def exact_spectrum(rho8):
    ev = lambda l: float(np.real(np.trace(pauli_op(l) @ rho8)))
    zL = 0.5 * (ev("XII") + ev("IXI") + ev("IIX") - ev("XXX"))
    yL = -0.5 * (ev("YYY") + ev("ZZY") + ev("YZZ") + ev("ZYZ"))
    return dict(Z=zL, X=ev("ZZZ"), Y=yL, S1=ev("XXI"), S2=ev("IXX"),
                XXX=ev("XXX"), YYY=ev("YYY"), ZZZ=ev("ZZZ"))


def fid_noisy_batch(th, ph, mu, sigma, rhoc, gamma=0.0, phase=1):
    """ONE noise interval, no QAE:  F = Tr[ N(rho0) rho0 ]."""
    v = np.stack([code_amp(t, p, phase) for t, p in zip(th, ph)])
    r0 = v[:, :, None] * v.conj()[:, None, :]
    rn = noise_channel(r0, mu, sigma, rhoc, gamma, 3)
    return np.real(np.einsum("nab,nba->n", r0, rn))


def fid_qae_batch(th, ph, mu, sigma, rhoc, S=None, phase=1, gamma=0.0):
    """ONE noise interval + (optional) QAE, fidelity with the clean state."""
    v = np.stack([code_amp(t, p, phase) for t, p in zip(th, ph)])
    r0 = v[:, :, None] * v.conj()[:, None, :]
    rn = noise_channel(r0, mu, sigma, rhoc, gamma, 3)
    out = rn if S is None else (S @ rn.reshape(-1, 64).T).T.reshape(-1, 8, 8)
    return np.real(np.einsum("nab,nba->n", r0, out))


def exact_run(variant, mu, sigma, rhoc, theta=None, L=L_HW, o=None,
              form="selfinv", states=None, gamma=0.0):
    """(P_memory, F_corrected, spectrum_of_first_state).
    P: 2 noise intervals, optional QAE between, ideal majority-vote recovery.
    F: 1 noise interval + QAE, state fidelity. variant: 'bare' | 'uncorr' | 'qae'."""
    o = dict(DEF_OPTS) if o is None else o
    st = axis_states() if states is None else states
    if variant == "bare":
        Ps = []
        for _, th, ph, ax, s in st:
            if ax is None: continue
            v = np.array([np.cos(th / 2), np.exp(1j * ph) * np.sin(th / 2)])
            r = np.outer(v, v.conj())
            for _ in range(2):                       # two storage intervals
                r = noise_channel(r, mu, sigma, 0.0, gamma, 1)
            b = [float(np.real(np.trace(_P1[k] @ r))) for k in "XYZ"]
            Ps.append(0.5 * (1 + s * b[AXI[ax]]))
        return (float(np.mean(Ps)) if Ps else float("nan")), float("nan"), {}
    S = None
    if variant == "qae":
        U = Operator(build_u(np.asarray(theta, float), L, o, "hw")).data
        S = map_superop(U, o, form)
    Ps, Fs, spec0 = [], [], None
    for k, (_, th, ph, ax, s) in enumerate(st):
        rho0 = code_state(th, ph, o.get("phase", 1))
        r = noise_channel(rho0, mu, sigma, rhoc, gamma, 3)       # interval 1
        if S is not None: r = apply_superop(S, r)                # QAE
        Fs.append(float(np.real(np.trace(r @ rho0))))
        sp = exact_spectrum(noise_channel(r, mu, sigma, rhoc, gamma, 3))
        if k == 0: spec0 = sp
        if ax is not None: Ps.append(0.5 * (1 + s * sp[ax]))
    return ((float(np.mean(Ps)) if Ps else float("nan")),
            float(np.mean(Fs)), spec0)


# ------------------------------------------------- analytic / numeric targets
def P_bare_theory(mu, sig, gamma=0.0):
    """Single unencoded qubit, TWO intervals of AD(gamma) o Dephase(mu,sig).
    Exact: damping and dephasing commute on one qubit up to a phase that
    cancels in K rho K^dagger.
        lam_perp = exp(-sig^2)(1-gamma)     (total transverse damping)
        lam_z    = (1-gamma)^2              (total longitudinal contraction)
        P = 1/2 [ 1 + (2 lam_perp cos(2mu) + lam_z)/3 ]"""
    lp = math.exp(-sig * sig) * (1.0 - gamma)
    lz = (1.0 - gamma) ** 2
    return 0.5 * (1.0 + (2.0 * lp * math.cos(2 * mu) + lz) / 3.0)


def P_unc_theory(mu, sig, rho, gamma=0.0):
    if gamma and gamma > 0.0:
        return float("nan")
    m, v = 2*mu, 2*sig**2
    C1 = math.cos(m)*math.exp(-v/2)
    C3 = .25*(math.cos(3*m)*math.exp(-1.5*v*(1+2*rho))
              + 3*math.cos(m)*math.exp(-.5*v*(3-2*rho)))
    return 2/3 + C1/2 - C3/6


def P_unc_exact(mu, sig, rho, gamma=0.0):
    """Encoded, 2 intervals, NO QAE, ideal recovery. Exact for any gamma."""
    return exact_run("uncorr", mu, sig, rho, gamma=gamma)[0]


def P_ideal_exact(theta, mu, sig, rho, gamma=0.0, L=L_HW, o=None,
                  form="selfinv"):
    """Encoded, interval / QAE / interval, ideal recovery. Noiseless QAE."""
    return exact_run("qae", mu, sig, rho, theta=theta, L=L, o=o, form=form,
                     gamma=gamma)[0]


def p2q_crit(P_ideal, P_ref, C):
    a, b = P_ideal - 0.5, P_ref - 0.5
    if a <= 0 or b <= 0 or C <= 0: return float("nan")
    return math.log(a / b) / C


# ------------------------------------------------- device idle <-> knobs
def tphi_from(T1_us, T2_us):
    """1/T2 = 1/(2 T1) + 1/T_phi."""
    x = 1.0 / T2_us - 0.5 / T1_us
    return float("inf") if x <= 0 else 1.0 / x


def noise_at_tau(tau_half_us, T1_us, Tphi_us):
    """PER-INTERVAL (sigma, gamma) for an idle of tau_half microseconds."""
    sig = math.sqrt(max(2.0 * tau_half_us / Tphi_us, 0.0)) if Tphi_us > 0 else 0.0
    gam = 1.0 - math.exp(-tau_half_us / T1_us) if T1_us > 0 else 0.0
    return float(sig), float(gam)


def sigma_equiv(lam_meas, mu=0.0):
    """lam_meas = measured TOTAL two-interval transverse damping."""
    x = lam_meas / max(math.cos(2*mu), 1e-9)
    return math.sqrt(max(-math.log(max(x, 1e-12)), 0.0))


def sigma_split(sig_eq, g2):
    """Remove the T1 share.  lam_perp = exp(-sigma^2) sqrt(1-g2), so
       sigma_deph^2 = sigma_eq^2 + 0.5 ln(1-g2),   g2 = 1-(1-gamma)^2.
    This is the part the phase-flip code can actually correct."""
    return math.sqrt(max(sig_eq**2 + 0.5*math.log(max(1.0-g2, 1e-12)), 0.0))


# ------------------------------------------------------------ circuits
@dataclass
class Cfg:
    L: int = L_HW
    o: dict = field(default_factory=lambda: dict(DEF_OPTS))
    form: str = "selfinv"
    idle_impl: str = "delay"      # 'delay' = hardware | 'rz' = local sim only
    mu: float = 0.0               # NEVER injected on hardware; 0 always
    sigma: float = 0.2            # only for idle_impl='rz'
    rho: float = 0.05
    n_angle: int = 256
    qec_mode: str = "serial"    # 'serial' (1 ancilla) | 'parallel' (2 ancillas)


def add_prep(qc, th, ph):
    qc.ry(th, CODE[0]); qc.rz(ph, CODE[0])
    qc.cx(CODE[0], CODE[1]); qc.cx(CODE[0], CODE[2])
    for q in CODE: qc.h(q)


def add_meas(qc, setting, slots, creg=None):
    """creg lets 'qec' share the register layout with an extra syndrome
    register already present in the circuit; other variants pass None and
    get the default 0..len(slots)-1 classical bits."""
    for b, q in zip(setting, slots):
        if b == "X": qc.h(q)
        elif b == "Y": qc.sdg(q); qc.h(q)
    bits = list(creg) if creg is not None else list(range(len(slots)))
    qc.measure(list(slots), bits)


def add_std_qec_serial(qc, code, anc, syn):
    """ONE round of REAL syndrome extraction for the [[3,1,3]] phase-flip
    code (X-type stabilizers S1=X_{q0}X_{q1}, S2=X_{q1}X_{q2}), reusing a
    SINGLE ancilla TWICE (measure S1, reset, measure S2), followed by a
    MID-CIRCUIT conditional Z correction (dynamic circuit: measure ->
    if_test -> Z). 

    syndrome bit convention (syn[0], syn[1]) -> integer value of the 2-bit
    register (syn[0] = LSB):
        0 (00) no error   1 (01) error on q0   3 (11) error on q1
        2 (10) error on q2
    """
    q0, q1, q2 = code
    qc.reset(anc)
    qc.h(anc); qc.cx(anc, q0); qc.cx(anc, q1); qc.h(anc)
    qc.measure(anc, syn[0])
    qc.reset(anc)
    qc.h(anc); qc.cx(anc, q1); qc.cx(anc, q2); qc.h(anc)
    qc.measure(anc, syn[1])
    with qc.if_test((syn, 1)):
        qc.z(q0)
    with qc.if_test((syn, 3)):
        qc.z(q1)
    with qc.if_test((syn, 2)):
        qc.z(q2)


def add_std_qec_parallel(qc, code, anc1, anc2, syn):
    """ONE round of REAL syndrome extraction for the [[3,1,3]] phase-flip
    code (X-type stabilizers S1=X_{q0}X_{q1}, S2=X_{q1}X_{q2}), using TWO
    ancillas so S1 and S2 are measured CONCURRENTLY (anc1 -> S1, anc2 ->
    S2), followed by a MID-CIRCUIT conditional Z correction (dynamic
    circuit: measure -> if_test -> Z). Needs a 5th physical qubit (anc2)
    beyond the usual 4-qubit layout.
    """
    q0, q1, q2 = code
    qc.reset(anc1); qc.reset(anc2)
    qc.h(anc1); qc.h(anc2)
    qc.cx(anc1, q0); qc.cx(anc2, q2)
    qc.cx(anc1, q1); qc.cx(anc2, q1)
    qc.h(anc1); qc.h(anc2)
    qc.measure(anc1, syn[0])
    qc.measure(anc2, syn[1])
    with qc.if_test((syn, 1)):
        qc.z(q0)
    with qc.if_test((syn, 3)):
        qc.z(q1)
    with qc.if_test((syn, 2)):
        qc.z(q2)


def _interval(qc, slots, k, cfg, idle_dt, fv):
    """One external-noise interval. On hardware (idle_impl='delay') this is a
    bare device idle: NOTHING is injected. cfg.mu must stay 0."""
    if cfg.mu:
        for q in slots: qc.rz(cfg.mu, q)          # virtual Z, free; unused
    if cfg.idle_impl == "rz" and fv is not None:
        for q in slots: qc.rz(fv[3 * k + q], q)
    elif cfg.idle_impl == "delay" and idle_dt > 0:
        for q in slots: qc.delay(idle_dt, q, unit="dt")


def trash_slots(o):
    return list(CODE) if o.get("reset", "code") == "code" else [ANC]


def build_circuit(cfg, state, setting, variant, idle_dt, mid_dt, tv, fv,
                  slot=0, extra_dt=0, extra_dt_pre=0):
    """variant: 'qae' | 'uncorr' | 'bare' | 'qec' | 'ref'."""
    _, th, ph, _, _ = state
    slots = (slot,) if variant == "bare" else CODE
    creg = ClassicalRegister(len(slots), "c")
    regs = [QuantumRegister(NQ, "q")]
    syn = anc2 = None
    if variant == "qec":
        syn = ClassicalRegister(2, "syn")
        if cfg.qec_mode == "parallel":
            anc2_reg = QuantumRegister(1, "anc2")
            regs.append(anc2_reg)
            anc2 = anc2_reg[0]
        regs.append(syn)
    regs.append(creg)
    qc = QuantumCircuit(*regs)
    if variant == "bare":
        qc.ry(th, slot); qc.rz(ph, slot)
    else:
        add_prep(qc, th, ph)
    if variant != "ref":
        qc.barrier(); _interval(qc, slots, 0, cfg, idle_dt, fv); qc.barrier()
        if variant == "qae":
            u = build_u(tv, cfg.L, cfg.o, "hw")
            qc.compose(u, inplace=True)
            if cfg.form == "selfinv":
                qc.barrier()
                for q in trash_slots(cfg.o): qc.reset(q)
                qc.barrier()
                qc.compose(u.inverse(), inplace=True)
            if extra_dt > 0:
                for q in CODE: qc.delay(extra_dt, q, unit="dt")
        elif variant == "qec":
            if extra_dt_pre > 0:
                for q in CODE: qc.delay(extra_dt_pre, q, unit="dt")
            if cfg.qec_mode == "parallel":
                add_std_qec_parallel(qc, CODE, ANC, anc2, syn)
            else:
                add_std_qec_serial(qc, CODE, ANC, syn)
            if extra_dt > 0:
                for q in CODE: qc.delay(extra_dt, q, unit="dt")
        elif mid_dt > 0:
            for q in slots: qc.delay(mid_dt, q, unit="dt")
        qc.barrier(); _interval(qc, slots, 1, cfg, idle_dt, fv); qc.barrier()
    add_meas(qc, setting, slots, creg)
    return qc


def qec_native_seconds(run, isa_qec, isa_ref, fb_latency_us=0.5):
    """Best-effort net duration of the std-QEC mid-block, in SECONDS."""
    raw = max(0.0, run.circuit_seconds(isa_qec) - run.circuit_seconds(isa_ref))
    return raw + fb_latency_us * 1e-6


def sample_angles(cfg, S, seed=0):
    rng = np.random.default_rng(seed)
    Sg = cfg.sigma ** 2 * ((1 - cfg.rho) * np.eye(3) + cfg.rho * np.ones((3, 3)))
    Lc = np.linalg.cholesky(Sg + 1e-15 * np.eye(3))
    return np.concatenate([rng.standard_normal((S, 3)) @ Lc.T,
                           rng.standard_normal((S, 3)) @ Lc.T], 1)


# ------------------------------------------------------------ backends
FAKES = {"fez": "FakeFez", "torino": "FakeTorino", "marrakesh": "FakeMarrakesh",
         "kingston": "FakeKingston", "sherbrooke": "FakeSherbrooke",
         "brisbane": "FakeBrisbane"}
DEF_DUR = {"sx": 3.2e-8, "x": 3.2e-8, "rz": 0.0, "id": 3.2e-8, "cz": 8.4e-8,
           "ecr": 3.0e-7, "measure": 1.4e-6, "reset": 1.6e-6}


class _IdPM:
    def run(self, qc): return qc


class Runner:
    """transpile target, pass manager, sampler, session, timing, usage."""
    def __init__(self, spec, rep_delay_us=100.0, chunk=48, max_exec_s=3600,
                 session="batch"):
        self.spec, self.rep, self.chunk = spec, rep_delay_us, chunk
        self.max_exec, self.session_kind = max_exec_s, session
        self.ctx, self.usage, self.jobs = None, 0.0, 0
        if ":" in spec:
            kind, short = spec.split(":", 1)
            from qiskit_ibm_runtime import fake_provider as fp
            self.tb = getattr(fp, FAKES.get(short, short))()
            self.kind, self.real = ("fake" if kind == "fake" else "aer"), False
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            self.svc = QiskitRuntimeService()
            self.tb = (self.svc.least_busy(operational=True, simulator=False,
                                           min_num_qubits=5)
                       if spec in ("least_busy", "auto") else self.svc.backend(spec))
            self.kind, self.real = "real", True
        self.name = getattr(self.tb, "name", str(self.tb))
        self.dt = getattr(self.tb, "dt", None) or 0.5e-9

    def pass_manager(self, layout=None):
        if self.kind == "aer": return _IdPM()
        return generate_preset_pass_manager(optimization_level=1, backend=self.tb,
                                            initial_layout=layout)

    def open(self):
        if self.real:
            from qiskit_ibm_runtime import Batch, Session
            cls = Batch if self.session_kind == "batch" else Session
            self.ctx = cls(backend=self.tb)
        return self

    def close(self):
        if self.ctx:
            try: self.ctx.close()
            except Exception: pass
            self.ctx = None

    def _rep_delay(self):
        if self.rep is None: return None
        r = self.rep * 1e-6
        try:
            lo, hi = self.tb.rep_delay_range
            r = min(max(r, lo), hi)
        except Exception: pass
        return r

    def _sampler(self, shots):
        if self.kind == "aer":
            try:
                from qiskit_aer.primitives import SamplerV2 as AS
                return AS(default_shots=shots)
            except Exception:
                from qiskit.primitives import StatevectorSampler
                return StatevectorSampler(default_shots=shots)
        from qiskit_ibm_runtime import SamplerV2
        s = SamplerV2(mode=self.ctx or self.tb)
        s.options.default_shots = shots
        if self.real:
            for f in (
                lambda: setattr(s.options.dynamical_decoupling, "enable", False),
                lambda: setattr(s.options.twirling, "enable_gates", False),
                lambda: setattr(s.options.twirling, "enable_measure", True),
                lambda: setattr(s.options.execution, "init_qubits", True),
                lambda: setattr(s.options.execution, "rep_delay", self._rep_delay()),
                lambda: setattr(s.options, "max_execution_time", int(self.max_exec)),
            ):
                try: f()
                except Exception: pass
        return s

    def run(self, pubs, shots, tag=""):
        out = []
        for i in range(0, len(pubs), self.chunk):
            sub = pubs[i:i + self.chunk]
            job = self._sampler(shots).run(sub)
            if self.real:
                try:
                    ue = job.usage_estimation
                    if ue: print(f"    [{tag}] IBM estimate "
                                 f"{ue.get('quantum_seconds', 0):.1f} QPU s "
                                 f"({len(sub)} circuits)")
                except Exception: pass
            out.extend(list(job.result()))
            self.jobs += 1
            try: self.usage += float(job.metrics()["usage"]["quantum_seconds"])
            except Exception: pass
        return out

    def gate_seconds(self, name, qargs):
        try:
            p = self.tb.target[name][tuple(qargs)]
            if p is not None and p.duration is not None: return float(p.duration)
        except Exception: pass
        return DEF_DUR.get(name, 0.0)

    def circuit_seconds(self, isa):
        clk = defaultdict(float)
        for ins in isa.data:
            qs = [isa.find_bit(q).index for q in ins.qubits]
            nm = ins.operation.name
            if nm == "barrier":
                t = max((clk[q] for q in qs), default=0.0)
                for q in qs: clk[q] = t
                continue
            if nm == "delay":
                u = ins.operation.unit; d = float(ins.operation.duration)
                d = d * self.dt if u == "dt" else d * {"s": 1, "ms": 1e-3,
                                                       "us": 1e-6, "ns": 1e-9}[u]
            else:
                d = self.gate_seconds(nm, qs)
            t0 = max((clk[q] for q in qs), default=0.0)
            for q in qs: clk[q] = t0 + d
        return max(clk.values(), default=0.0)

    def estimate(self, isa_list, n_exec, shots):
        rep = (self.rep or 0.0) * 1e-6
        return sum(n * shots * (self.circuit_seconds(isa) + rep)
                   for isa, n in zip(isa_list, n_exec))

    def idle_dt(self, us):
        g = 16
        try: g = int(self.tb.target.granularity) or 16
        except Exception: pass
        return int(round(us * 1e-6 / self.dt / g)) * g

    def coherence(self, qubits):
        """median (T1, T2, T_phi) in microseconds over the given qubits."""
        t1, t2 = [], []
        for q in qubits:
            try:
                p = self.tb.qubit_properties(q)
                if p.t1: t1.append(p.t1 * 1e6)
                if p.t2: t2.append(p.t2 * 1e6)
            except Exception: pass
        if not t1 or not t2: return (float("nan"),) * 3
        a, b = float(np.median(t1)), float(np.median(t2))
        return a, b, tphi_from(a, b)

    def median_p2q(self):
        errs = []
        for nm in ("cz", "ecr", "cx"):
            if nm in self.tb.target.operation_names:
                for q, p in self.tb.target[nm].items():
                    if p is not None and p.error is not None: errs.append(p.error)
        return float(np.median(errs)) if errs else float("nan")


# --------------------------------------------------------- layout choice
def pick_layout(runner, idle_us=15.0, L=L_HW, top=10, verbose=True):
    """4 physical qubits for the all-to-all ansatz."""
    tgt = runner.tb.target
    edges = set()
    for nm in ("cz", "ecr", "cx"):
        if nm in tgt.operation_names:
            for q in tgt[nm]:
                if q and len(q) == 2: edges.add(tuple(sorted(q)))
    adj = defaultdict(set)
    for a, b in edges: adj[a].add(b); adj[b].add(a)

    def e2(a, b):
        for nm in ("cz", "ecr", "cx"):
            try:
                p = tgt[nm].get((a, b)) or tgt[nm].get((b, a))
                if p and p.error is not None: return p.error
            except Exception: pass
        return 3e-3

    def ero(q):
        try:
            p = tgt["measure"][(q,)]
            return p.error if p and p.error is not None else 1.5e-2
        except Exception: return 1.5e-2

    def t2(q):
        try: return runner.tb.qubit_properties(q).t2 or 1e-4
        except Exception: return 1e-4

    subs = set()
    def grow(cur):
        if len(cur) == 4: subs.add(frozenset(cur)); return
        for q in set().union(*[adj[c] for c in cur]) - cur: grow(cur | {q})
    for s in list(adj): grow({s})

    scored = []
    for sub in subs:
        sub = sorted(sub)
        deg = {q: len(adj[q] & set(sub)) for q in sub}
        star = 1.0 if max(deg.values()) == 3 else 0.0
        pres = [(a, b) for a, b in itertools.combinations(sub, 2)
                if tuple(sorted((a, b))) in edges]
        cost = (-0.5 * star + sum(e2(a, b) for a, b in pres) * 10
                + 2.0 * sum(sorted(ero(q) for q in sub)[:3])
                + sum(2 * idle_us * 1e-6 / t2(q) for q in sub))
        scored.append((cost, sub))
    scored.sort(key=lambda x: x[0])
    best = None
    for _, sub in scored[:top]:
        order = sorted(sub, key=lambda q: ero(q))   # worst readout -> ancilla
        perm = order[:3] + [order[3]]
        try:
            pm = runner.pass_manager(perm)
            tv = ParameterVector("t", n_params(L, DEF_OPTS, "hw"))
            qc = build_circuit(Cfg(L=L), axis_states()[0], "XXX", "qae",
                               0, 0, tv, None)
            isa = pm.run(qc)
            n2 = sum(1 for i in isa.data if len(i.qubits) == 2
                     and i.operation.name != "barrier")
            err = 0.0
            for i in isa.data:
                if len(i.qubits) == 2 and i.operation.name != "barrier":
                    a, b = (isa.find_bit(q).index for q in i.qubits)
                    err += e2(a, b)
            sc = err + 2.0 * sum(ero(q) for q in perm[:3])
        except Exception:
            n2, sc = 999, 999.0
        if best is None or sc < best[0]: best = (sc, perm, n2)
    if verbose:
        print(f"  layout {best[1]} (code {best[1][:3]}, ancilla {best[1][3]}); "
              f"L={L}: {best[2]} 2q gates after routing, est. gate+readout error "
              f"{best[0]:.3f}")
    return best[1]


def gate_counts(isa):
    """(N2q, N1q_all, N1q_phys). rz is virtual on IBM hardware (zero error)."""
    skip = {"barrier", "delay", "measure", "reset", "id"}
    n2 = n1 = n1p = 0
    for ins in isa.data:
        nm = ins.operation.name
        if nm in skip: continue
        if len(ins.qubits) == 2: n2 += 1
        elif len(ins.qubits) == 1:
            n1 += 1
            if nm in ("sx", "x", "sxdg", "ry", "h", "sdg", "s"): n1p += 1
    return n2, n1, n1p


# ------------------------------------------------------- result helpers
def counts_probs(counts, nbits):
    p = np.zeros(1 << nbits); tot = sum(counts.values()) or 1
    for k, v in counts.items():
        b = k.replace(" ", "")
        p[int(b[::-1], 2) if b else 0] += v      # clbit 0 -> our LSB
    return p / tot


def pub_probs(field, nbits):
    if field.shape == ():
        return np.array(counts_probs(field.get_counts(), nbits))[None, :]
    flat = [counts_probs(field.get_counts(loc), nbits)
            for loc in np.ndindex(field.shape)]
    return np.array(flat).reshape(field.shape + (1 << nbits,))


def param_array(isa, theta_rows, ang_rows, tv, fv):
    B = theta_rows.shape[0]; S = 1 if ang_rows is None else ang_rows.shape[0]
    names = [p.name for p in list(tv)]
    if fv is not None: names += [p.name for p in list(fv)]
    idx = {n: i for i, n in enumerate(names)}
    full = np.zeros((B, S, len(names)))
    full[:, :, :len(tv)] = np.asarray(theta_rows)[:, None, :]
    if fv is not None and ang_rows is not None:
        full[:, :, len(tv):] = np.asarray(ang_rows)[None, :, :]
    cols = [idx[p.name] for p in isa.parameters]
    return full[:, :, cols]


def ro_circuits():
    cs = []
    for b in (0, 1):
        qc = QuantumCircuit(QuantumRegister(NQ, "q"), ClassicalRegister(NQ, "c"))
        if b: qc.x(range(NQ))
        qc.barrier(); qc.measure(range(NQ), range(NQ)); cs.append(qc)
    return cs


def ro_eps(p0, p1):
    eps = np.zeros((NQ, 2))
    for q in range(NQ):
        m0 = p0[[b for b in range(1 << NQ) if (b >> q) & 1]].sum()
        m1 = p1[[b for b in range(1 << NQ) if not (b >> q) & 1]].sum()
        eps[q] = (m0, m1)
    return eps


def ro_unfold(probs, slots, eps):
    A = np.ones((1, 1))
    for q in slots:
        a = np.array([[1 - eps[q, 0], eps[q, 1]], [eps[q, 0], 1 - eps[q, 1]]])
        A = np.kron(a, A)
    p = np.clip(np.asarray(probs) @ np.linalg.pinv(A).T, 0, None)
    return p / np.clip(p.sum(-1, keepdims=True), 1e-12, None)


def save(path, **kw):
    np.savez_compressed(path, **kw); print(f"  saved {path}")