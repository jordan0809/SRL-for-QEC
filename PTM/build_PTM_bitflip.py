"""
Spectral PTM analysis for the trained [[3,1,3]] bit-flip QAE
(measurement-free spectral QEC).

Usage
-----
    python build_PTM_bitflip.py ../bitflip_rep_code/data/bitflip_results_layerwise.npz
    python build_PTM_bitflip.py ../bitflip_rep_code/data/bitflip_results_layerwise.npz --layer 3 --p 0.1 --seed 7

By default it uses the overall best model saved in the .npz
(`best_n_layers` / `best_params`). Pass --layer L to instead use the L-layer
model from that same sweep (`L{L}_params`).
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Encoder ansatz — identical structure to the training script, so that
# params loaded from the .npz reproduce exactly the trained unitary V.
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
    """Returns R_theta: 8x8 operator -> 8x8 operator (linear, works on any
    operator, not just physical density matrices)."""
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
# Noise channel + logical states
# ----------------------------------------------------------------------
def bit_flip_channel(rho, p):
    from itertools import product as iproduct
    out = np.zeros_like(rho)
    for i, j, k in iproduct([0, 1], repeat=3):
        prob = (p if i else 1 - p) * (p if j else 1 - p) * (p if k else 1 - p)
        E = kron_list([X_m if b else I2 for b in (i, j, k)])
        out += prob * E @ rho @ E.conj().T
    return out


ket_000 = np.zeros(8, dtype=complex); ket_000[0] = 1.0
ket_111 = np.zeros(8, dtype=complex); ket_111[7] = 1.0


def logical_state(theta, phi):
    return np.cos(theta / 2) * ket_000 + np.exp(1j * phi) * np.sin(theta / 2) * ket_111


def state_to_dm(psi):
    psi = psi.reshape(-1, 1)
    return psi @ psi.conj().T


# ----------------------------------------------------------------------
# Normalizer basis: N(S) = stabilizer group (4) x logical Pauli group (4)
# ----------------------------------------------------------------------
def build_normalizer_basis():
    S1 = pauli_string('ZZI')
    S2 = pauli_string('IZZ')
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
    ap.add_argument('npz_path', help="path to bitflip_results_*.npz")
    ap.add_argument('--layer', type=int, default=None,
                     help="use L{layer}_params instead of the overall best model")
    ap.add_argument('--p', type=float, default=0.1, help="bit-flip error rate for the demo state")
    ap.add_argument('--seed', type=int, default=0, help="RNG seed for the random logical state")
    ap.add_argument('--out-prefix', default='spectral_ptm', help="output file prefix")
    args = ap.parse_args()

    data = np.load(args.npz_path)
    if args.layer is not None:
        nl = args.layer
        params = data[f'L{nl}_params']
    else:
        nl = int(data['best_n_layers'])
        params = data['best_params']
    print(f"Loaded trained model: L={nl} layers, n_params={params.shape[0]}")

    R = make_recovery_channel(nl, params)
    basis = build_normalizer_basis()

    # sanity check: normalizer basis operators must be Hermitian
    for name, op in basis.items():
        herm_err = np.max(np.abs(op - op.conj().T))
        assert herm_err < 1e-8, f"{name} is not Hermitian (err={herm_err})"

    # ---- pick a random logical state and run it through noise + recovery
    rng = np.random.default_rng(args.seed)
    theta = np.arccos(1 - 2 * rng.random())
    phi = 2 * np.pi * rng.random()
    psi = logical_state(theta, phi)
    rho_clean = state_to_dm(psi)
    rho_noisy = bit_flip_channel(rho_clean, args.p)
    rho_recovered = np.array(R(jnp.array(rho_noisy, dtype=complex)))

    spec_clean = spectrum(rho_clean, basis)
    spec_noisy = spectrum(rho_noisy, basis)
    spec_recovered = spectrum(rho_recovered, basis)

    # ---- build the PTM of the recovery channel over the normalizer subspace
    K, names = channel_ptm(R, basis)

    # ---- verify linearity: K @ noisy_vec should equal the directly
    #      computed recovered_vec for THIS state (sanity check on K)
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
             theta=theta, phi=phi, p=args.p, n_layers=nl,
             linearity_check_max_err=lin_err,
             off_diag_l1=off_diag_mass, diag_l1=diag_mass)
    print(f"Saved: {out_npz}")

    out_txt = f"{args.out_prefix}_L{nl}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Spectral PTM analysis -- [[3,1,3]] bit-flip QAE, L={nl}, p={args.p}\n")
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

    # ---- plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    im = ax.imshow(K, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=12)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=12)
    #ax.set_xlabel("Input normalizer operator  P")
    #ax.set_ylabel("Output normalizer operator  Q")
    #ax.set_title(f"Recovery-channel PTM   K[Q,P] = Tr(Q · R(P)) / dim\n"
    #             f"L={nl} bit-flip QAE  |  off-diag L1 = {off_diag_mass:.3f}, diag L1 = {diag_mass:.3f}")
    #ax.set_title(f"Recovery-channel PTM", fontsize = 20)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.tick_params(labelsize=12)

    ax2 = axes[1]
    x = np.arange(len(names)); w = 0.27
    ax2.bar(x - w, clean_vec, w, label='Clean', color='#2ecc71')
    ax2.bar(x, noisy_vec, w, label='Noisy', color='#e74c3c')
    ax2.bar(x + w, recovered_vec, w, label='Recovered', color='#3498db')
    ax2.set_xticks(x); ax2.set_xticklabels(names, rotation=90, fontsize=12)
    ax2.tick_params(axis='both', labelsize=12)
    #ax2.set_ylabel("Spectral coefficient  ⟨P⟩")
    #ax2.set_title(f"Normalizer spectrum\n"
    #              f"(θ={theta:.2f}, ϕ={phi:.2f}, p={args.p})", fontsize = 20)
    ax2.legend(fontsize=12); ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(0, color='black', lw=0.7)

    #plt.suptitle("[[3,1,3]] recovery QAE PTM/spectrum analysis (bit-flip noise)",
    #             fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    out_png = f"{args.out_prefix}_L{nl}.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()