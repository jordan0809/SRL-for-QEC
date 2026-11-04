"""
Spectral PTM analysis for the trained [[3,1,3]] PHASE-FLIP QAE
(measurement-free spectral QEC under correlated Gaussian dephasing).

Usage
-----
    python build_PTM_phaseflip.py ../phaseflp_rep_code/data/dephasing_results_layerwise.npz
    python build_PTM_phaseflip.py ../phaseflp_rep_code/data/dephasing_results_layerwise.npz --layer 3 --seed 7
    python build_PTM_phaseflip.py ../phaseflp_rep_code/data/dephasing_results_layerwise.npz --linthresh 0.005
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.ticker import LogLocator

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------
# Pauli primitives (same convention as the training script)
# ----------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
X_m = np.array([[0, 1], [1, 0]], dtype=complex)
Y_m = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_m = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {'I': I2, 'X': X_m, 'Y': Y_m, 'Z': Z_m}


def kron_list(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def pauli_string(s):
    return kron_list([PAULI[c] for c in s])


N_CODE, N_ANC = 3, 1
N_TOTAL = N_CODE + N_ANC        # 4 (3 code qubits + 1 ancilla)
SYS_DIM = 2 ** N_CODE            # 8
TOT_DIM = 2 ** N_TOTAL           # 16

ANC_0 = np.zeros((2, 2), dtype=complex); ANC_0[0, 0] = 1.0
FRESH_000 = np.zeros((SYS_DIM, SYS_DIM), dtype=complex); FRESH_000[0, 0] = 1.0


# ----------------------------------------------------------------------
# Encoder ansatz — identical structure to the training script
# ----------------------------------------------------------------------
def count_model_b_params(n_layers):
    n_rot = (n_layers + 1) * N_TOTAL * 2
    n_cr = n_layers * N_TOTAL * (N_TOTAL - 1) * 3
    return n_rot + n_cr


def model_b_ansatz(params, n_layers):
    idx = 0
    for q in range(N_TOTAL):
        qml.RY(params[idx], wires=q); idx += 1
        qml.RZ(params[idx], wires=q); idx += 1
    for _ in range(n_layers):
        for c in range(N_TOTAL):
            for t in range(N_TOTAL):
                if c != t:
                    qml.CRX(params[idx], wires=[c, t]); idx += 1
                    qml.CRY(params[idx], wires=[c, t]); idx += 1
                    qml.CRZ(params[idx], wires=[c, t]); idx += 1
        for q in range(N_TOTAL):
            qml.RY(params[idx], wires=q); idx += 1
            qml.RZ(params[idx], wires=q); idx += 1
    return idx


def make_encoder(n_layers):
    def circuit(params):
        model_b_ansatz(params, n_layers)
    return qml.matrix(circuit, wire_order=range(N_TOTAL))


def partial_trace_system(rho16):
    r = rho16.reshape(SYS_DIM, 2, SYS_DIM, 2)
    return jnp.trace(r, axis1=0, axis2=2)


def partial_trace_ancilla(rho16):
    r = rho16.reshape(SYS_DIM, 2, SYS_DIM, 2)
    return jnp.trace(r, axis1=1, axis2=3)


def make_recovery_channel(n_layers, params):
    """R_theta: 8x8 operator -> 8x8 operator (linear, works on any operator)."""
    encoder = make_encoder(n_layers)
    V = encoder(jnp.asarray(params, dtype=jnp.float64))
    Vd = jnp.conj(V.T)
    ANC0_j = jnp.array(ANC_0, dtype=complex)
    FRESH_j = jnp.array(FRESH_000, dtype=complex)

    def R(rho_3q):
        rho_in = jnp.kron(rho_3q.astype(complex), ANC0_j)
        rho_enc = V @ rho_in @ Vd
        rho_anc = partial_trace_system(rho_enc)
        rho_dec_in = jnp.kron(FRESH_j, rho_anc)
        rho_dec = Vd @ rho_dec_in @ V
        return partial_trace_ancilla(rho_dec)
    return R


# ----------------------------------------------------------------------
# EXACT correlated Gaussian-dephasing channel  (elementwise Phi)
#   (matches build_phi / dephasing_channel from the training script)
# ----------------------------------------------------------------------
_zvecs = np.array([[1 - 2 * ((b >> (N_CODE - 1 - i)) & 1) for i in range(N_CODE)]
                   for b in range(SYS_DIM)], dtype=float)     # (8,3), entries ±1


def build_phi(sigma, rho_corr, mu=0.0):
    """Exact damping matrix. mu != 0 => non-diagonal (coherent) PTM."""
    M = np.eye(N_CODE) * (1 - rho_corr) + rho_corr * np.ones((N_CODE, N_CODE))
    Sigma = (sigma ** 2) * M
    mu_vec = np.full(N_CODE, mu)
    Phi = np.empty((SYS_DIM, SYS_DIM), dtype=complex)
    for a in range(SYS_DIM):
        for b in range(SYS_DIM):
            c = (_zvecs[a] - _zvecs[b]) / 2.0
            Phi[a, b] = np.exp(-1j * (mu_vec @ c) - 0.5 * (c @ Sigma @ c))
    return Phi


def dephasing_channel(rho, phi):
    return phi * rho


# ----------------------------------------------------------------------
# Phase-flip logical states:  |+L> = |+++>,  |-L> = |--->
# ----------------------------------------------------------------------
ket_p = np.array([1, 1],  dtype=complex) / np.sqrt(2)
ket_m = np.array([1, -1], dtype=complex) / np.sqrt(2)
ket_0L = kron_list([ket_p, ket_p, ket_p]).flatten()
ket_1L = kron_list([ket_m, ket_m, ket_m]).flatten()


def logical_state(theta, phi):
    return np.cos(theta / 2) * ket_0L + np.exp(1j * phi) * np.sin(theta / 2) * ket_1L


def state_to_dm(psi):
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


# ----------------------------------------------------------------------
# Normalizer basis: N(S) = stabilizer group (4) x logical Pauli group (4)
#   Phase-flip: S1 = X1X2 = XXI, S2 = X2X3 = IXX ; XL=XXX, YL=YYY, ZL=ZZZ
# ----------------------------------------------------------------------
def build_normalizer_basis():
    S1 = pauli_string('XXI')
    S2 = pauli_string('IXX')
    stab = {'I': np.eye(8, dtype=complex), 'S1': S1, 'S2': S2, 'S1S2': S1 @ S2}

    XL, YL, ZL = pauli_string('XXX'), pauli_string('YYY'), pauli_string('ZZZ')
    logical = {'I': np.eye(8, dtype=complex), 'XL': XL, 'YL': YL, 'ZL': ZL}

    basis = {}
    for sn, sm in stab.items():
        for ln, lm in logical.items():
            if ln == 'I':
                name = sn
            elif sn == 'I':
                name = ln
            else:
                name = f"{sn}.{ln}"
            basis[name] = sm @ lm
    return basis


def spectrum(rho, basis):
    return {name: float(np.real(np.trace(op @ rho))) for name, op in basis.items()}


def channel_ptm(R, basis):
    names = list(basis.keys())
    n = len(names)
    dim = basis[names[0]].shape[0]
    K = np.zeros((n, n))
    for j, pname in enumerate(names):
        P = basis[pname]
        out = np.array(R(jnp.array(P, dtype=complex)))
        for i, qname in enumerate(names):
            Q = basis[qname]
            K[i, j] = float(np.real(np.trace(Q @ out))) / dim
    return K, names


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('npz_path', help="path to dephasing_results_*.npz")
    ap.add_argument('--layer', type=int, default=None,
                    help="use L{layer}_params instead of the overall best model")
    ap.add_argument('--sigma', type=float, default=None,
                    help="override per-qubit angle std (default: meta_sigma from npz)")
    ap.add_argument('--rho-corr', type=float, default=None,
                    help="override correlation (default: meta_rho_corr from npz)")
    ap.add_argument('--mu', type=float, default=None,
                    help="override mean angle mu (default: meta_mu from npz)")
    ap.add_argument('--seed', type=int, default=0, help="RNG seed for the random logical state")
    ap.add_argument('--linthresh', type=float, default=1e-2,
                    help="SymLogNorm linear threshold: entries below this stay near-white, "
                         "above it are log-scaled (default: 0.01)")
    ap.add_argument('--out-prefix', default='spectral_ptm_phaseflip', help="output file prefix")
    args = ap.parse_args()

    data = np.load(args.npz_path)
    if args.layer is not None:
        nl = args.layer
        params = data[f'L{nl}_params']
    else:
        nl = int(data['best_n_layers'])
        params = data['best_params']
    print(f"Loaded trained model: L={nl} layers, n_params={params.shape[0]}")

    # noise parameters — from npz meta unless overridden on the CLI
    sigma = float(args.sigma)    if args.sigma    is not None else float(data['meta_sigma'])
    rho_c = float(args.rho_corr) if args.rho_corr is not None else float(data['meta_rho_corr'])
    mu    = float(args.mu)       if args.mu       is not None else float(data['meta_mu'])
    print(f"Noise: sigma={sigma}, rho_corr={rho_c}, mu={mu}")

    PHI = build_phi(sigma, rho_c, mu=mu)
    max_im = float(np.max(np.abs(PHI.imag)))
    print(f"max|Im(PHI)| = {max_im:.3e}  "
          f"({'NON-diagonal / coherent noise' if max_im > 1e-12 else 'diagonal / dephasing noise'})")

    R = make_recovery_channel(nl, params)
    basis = build_normalizer_basis()

    for name, op in basis.items():
        herm_err = np.max(np.abs(op - op.conj().T))
        assert herm_err < 1e-8, f"{name} is not Hermitian (err={herm_err})"

    # ---- pick a random logical state and run it through noise + recovery
    rng = np.random.default_rng(args.seed)
    theta = np.arccos(1 - 2 * rng.random())
    phi = 2 * np.pi * rng.random()
    psi = logical_state(theta, phi)
    rho_clean = state_to_dm(psi)
    rho_noisy = dephasing_channel(rho_clean, PHI)
    rho_recovered = np.array(R(jnp.array(rho_noisy, dtype=complex)))

    spec_clean = spectrum(rho_clean, basis)
    spec_noisy = spectrum(rho_noisy, basis)
    spec_recovered = spectrum(rho_recovered, basis)

    # ---- build the PTM of the recovery channel over the normalizer subspace
    K, names = channel_ptm(R, basis)

    # ---- verify linearity: K @ noisy_vec should equal recovered_vec
    noisy_vec = np.array([spec_noisy[n] for n in names])
    clean_vec = np.array([spec_clean[n] for n in names])
    recovered_vec = np.array([spec_recovered[n] for n in names])
    predicted = K @ noisy_vec
    lin_err = np.max(np.abs(predicted - recovered_vec))
    print(f"Linearity check: max |K @ noisy_spectrum - recovered_spectrum| = {lin_err:.2e}")

    off_diag_mass = np.sum(np.abs(K - np.diag(np.diag(K))))
    diag_mass = np.sum(np.abs(np.diag(K)))
    verdict = "NON-diagonal" if off_diag_mass > 1e-3 else "diagonal"
    print(f"PTM off-diagonal L1 mass = {off_diag_mass:.4f}   diagonal L1 mass = {diag_mass:.4f}")
    print(f"--> The trained recovery-channel PTM is {verdict} on the normalizer subspace.")

    # ---- save numeric results
    out_npz = f"{args.out_prefix}_L{nl}.npz"
    np.savez(out_npz,
             names=np.array(names),
             K=K,
             spec_clean=clean_vec,
             spec_noisy=noisy_vec,
             spec_recovered=recovered_vec,
             theta=theta, phi=phi,
             sigma=sigma, rho_corr=rho_c, mu=mu, n_layers=nl,
             linearity_check_max_err=lin_err,
             off_diag_l1=off_diag_mass, diag_l1=diag_mass)
    print(f"Saved: {out_npz}")

    out_txt = f"{args.out_prefix}_L{nl}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Spectral PTM analysis -- [[3,1,3]] phase-flip QAE, L={nl}\n")
        f.write(f"Noise: sigma={sigma}, rho_corr={rho_c}, mu={mu}\n")
        f.write(f"Random logical state: theta={theta:.4f} rad, phi={phi:.4f} rad\n\n")
        f.write(f"{'Operator':<10}{'Clean':>10}{'Noisy':>10}{'Recovered':>12}\n")
        for n in names:
            f.write(f"{n:<10}{spec_clean[n]:>10.4f}{spec_noisy[n]:>10.4f}{spec_recovered[n]:>12.4f}\n")
        f.write(f"\nLinearity check: max|K@noisy - recovered| = {lin_err:.2e}\n")
        f.write(f"Off-diagonal L1 mass = {off_diag_mass:.4f}\n")
        f.write(f"Diagonal L1 mass     = {diag_mass:.4f}\n")
        f.write(f"--> PTM is {verdict} on the normalizer subspace.\n\n")
        f.write("PTM K (rows = output operator Q, cols = input operator P):\n")
        f.write("        " + "".join(f"{n:>9}" for n in names) + "\n")
        for i, n in enumerate(names):
            f.write(f"{n:<8}" + "".join(f"{K[i, j]:>9.3f}" for j in range(len(names))) + "\n")
    print(f"Saved: {out_txt}")

        # ---- plots (two separate figures, saved as PDF)
    LABEL_FS = 15
    TICK_FS = 12

    # ================= FIGURE 1: PTM heatmap (no title) =================
    fig1, ax = plt.subplots(figsize=(8, 7))
    linthresh = args.linthresh
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-1.0, vmax=1.0, base=10)
    im = ax.imshow(K, cmap='RdBu_r', norm=norm)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=TICK_FS)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICK_FS)
    #ax.set_xlabel("Input normalizer operator  P", fontsize=LABEL_FS)
    #ax.set_ylabel("Output normalizer operator  Q", fontsize=LABEL_FS)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    log_ticks = [-1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0]
    log_ticks = [t for t in log_ticks if abs(t) >= linthresh or t == 0.0]
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"{t:g}" for t in log_ticks])
    cbar.ax.tick_params(labelsize=TICK_FS)

    fig1.tight_layout()
    out_ptm = f"{args.out_prefix}_ptm_L{nl}.pdf"
    fig1.savefig(out_ptm, bbox_inches='tight')
    plt.close(fig1)
    print(f"Saved: {out_ptm}")

    # ================= FIGURE 2: normalizer spectrum =================
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    x = np.arange(len(names)); w = 0.27
    ax2.bar(x - w, clean_vec, w, label='Clean', color='#2ecc71')
    ax2.bar(x, noisy_vec, w, label='Noisy', color='#e74c3c')
    ax2.bar(x + w, recovered_vec, w, label='Recovered', color='#3498db')

    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=TICK_FS)
    ax2.tick_params(axis='y', labelsize=TICK_FS)
    #ax2.set_xlabel("Normalizer operator  P", fontsize=LABEL_FS)
    #ax2.set_ylabel(r"Spectral coefficient  $\langle P \rangle$", fontsize=LABEL_FS)
    ax2.legend(fontsize=TICK_FS)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', lw=0.7)

    fig2.tight_layout()
    out_spec = f"{args.out_prefix}_spectrum_L{nl}.pdf"
    fig2.savefig(out_spec, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out_spec}")


if __name__ == "__main__":
    main()