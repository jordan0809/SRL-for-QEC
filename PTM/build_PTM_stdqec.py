#!/usr/bin/env python3
"""
PTM of the STANDARD (syndrome-based) recovery channel for the [[3,1,3]]
PHASE-FLIP repetition code, restricted to the normalizer subspace.

Usage
-----
  python build_PTM_stdqec.py
  python build_PTM_stdqec.py --npz ../phaseflp_rep_code/data/dephasing_results_layerwise.npz
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# editable text in the PDF (TrueType instead of Type-3)
matplotlib.rcParams['pdf.fonttype'] = 42

# ----------------------------------------------------------------------
# Global font sizes
# ----------------------------------------------------------------------
LABEL_FS = 15
TICK_FS = 12

# ----------------------------------------------------------------------
# Pauli primitives (same convention as the training / QAE-PTM scripts)
# ----------------------------------------------------------------------
I2  = np.eye(2, dtype=complex)
X_m = np.array([[0, 1], [1, 0]], dtype=complex)
Y_m = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_m = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {'I': I2, 'X': X_m, 'Y': Y_m, 'Z': Z_m}

N_CODE  = 3
SYS_DIM = 2 ** N_CODE                     # 8


def kron_list(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def pauli_string(s):
    return kron_list([PAULI[c] for c in s])


# ----------------------------------------------------------------------
# EXACT correlated Gaussian-dephasing channel (elementwise Phi)
#   (matches build_phi / dephasing_channel from the training script)
# ----------------------------------------------------------------------
_zvecs = np.array([[1 - 2 * ((b >> (N_CODE - 1 - i)) & 1) for i in range(N_CODE)]
                   for b in range(SYS_DIM)], dtype=float)      # (8,3), entries +-1


def build_phi(sigma, rho_corr, mu=0.0):
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
# Phase-flip code: stabilizers, projectors, standard recovery
# ----------------------------------------------------------------------
I8 = np.eye(SYS_DIM, dtype=complex)
S1_mat = pauli_string('XXI')
S2_mat = pauli_string('IXX')
Z1 = pauli_string('ZII')
Z2 = pauli_string('IZI')
Z3 = pauli_string('IIZ')
XL_mat, YL_mat, ZL_mat = pauli_string('XXX'), pauli_string('YYY'), pauli_string('ZZZ')


def _proj(s1, s2):
    return ((I8 + s1 * S1_mat) / 2) @ ((I8 + s2 * S2_mat) / 2)


_STD_CORR = [(_proj(+1, +1), I8),
             (_proj(-1, +1), Z1),
             (_proj(-1, -1), Z2),
             (_proj(+1, -1), Z3)]

P_CODE = _proj(+1, +1)
TR_P   = float(np.real(np.trace(P_CODE)))          # = 2


def std_qec(A):
    """Standard syndrome-based recovery; linear, valid for any 8x8 operator."""
    out = np.zeros_like(A, dtype=complex)
    for Pi, R in _STD_CORR:
        out += R @ Pi @ A @ Pi @ R
    return out


# ----------------------------------------------------------------------
# Optional: LINEAR Petz (transpose) recovery map, sigma = P_code / 2
#   R(A) = sig^{1/2} N^dag( N(sig)^{-1/2} A N(sig)^{-1/2} ) sig^{1/2}
#   (no trace renormalization -> map stays linear, as a PTM requires)
# ----------------------------------------------------------------------
def make_petz_map(phi):
    sig      = P_CODE / TR_P
    sig_half = P_CODE / np.sqrt(TR_P)
    Nsig = phi * sig
    Nsig = (Nsig + Nsig.conj().T) / 2.0
    ev, evec = np.linalg.eigh(Nsig)
    tol = 1e-12 * float(ev.max())
    inv_sqrt = np.where(ev > tol, 1.0 / np.sqrt(np.where(ev > tol, ev, 1.0)), 0.0)
    Nih = (evec * inv_sqrt) @ evec.conj().T
    phi_c = np.conj(phi)

    def R(A):
        return sig_half @ (phi_c * (Nih @ A @ Nih)) @ sig_half
    return R


# ----------------------------------------------------------------------
# Logical states  |+L> = |+++>, |-L> = |--->
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
# Normalizer basis (IDENTICAL construction/order to build_PTM_phaseflip.py)
# ----------------------------------------------------------------------
def build_normalizer_basis():
    stab = {'I': I8.copy(), 'S1': S1_mat, 'S2': S2_mat, 'S1S2': S1_mat @ S2_mat}
    logical = {'I': I8.copy(), 'XL': XL_mat, 'YL': YL_mat, 'ZL': ZL_mat}
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


def split_name(n):
    if '.' in n:
        return tuple(n.split('.'))
    if n in ('XL', 'YL', 'ZL'):
        return ('I', n)
    return (n, 'I')


def spectrum(rho, basis):
    return {name: float(np.real(np.trace(op @ rho))) for name, op in basis.items()}


def channel_ptm(R, basis):
    """K[Q,P] = Tr(Q R(P)) / dim  plus per-column leakage out of span(basis)."""
    names = list(basis.keys())
    n = len(names)
    dim = basis[names[0]].shape[0]
    K = np.zeros((n, n))
    leak = np.zeros(n)
    for j, pname in enumerate(names):
        out = R(basis[pname].astype(complex))
        for i, qname in enumerate(names):
            K[i, j] = float(np.real(np.trace(basis[qname] @ out))) / dim
        nrm = float(np.real(np.trace(out.conj().T @ out)))
        cap = dim * float(np.sum(K[:, j] ** 2))
        leak[j] = 1.0 - cap / nrm if nrm > 1e-15 else 0.0
    return K, names, leak


def ptm_diagnostics(K, names):
    """off-diagonal L1, diagonal L1, and XL<->YL (coherent) mixing L1."""
    off = float(np.sum(np.abs(K - np.diag(np.diag(K)))))
    dia = float(np.sum(np.abs(np.diag(K))))
    xy = 0.0
    for i, ni in enumerate(names):
        _, li = split_name(ni)
        for j, nj in enumerate(names):
            _, lj = split_name(nj)
            if {li, lj} == {'XL', 'YL'}:
                xy += abs(K[i, j])
    return off, dia, xy


# ----------------------------------------------------------------------
# Plotting: one PTM per figure (no title), PDF output
# ----------------------------------------------------------------------
def plot_ptm_figure(K, names, linthresh, out_path, vmax=1.0):
    fig, ax = plt.subplots(figsize=(8, 7))
    norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(K, cmap='RdBu_r', norm=norm)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=TICK_FS)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=TICK_FS)
    #ax.set_xlabel("Input normalizer operator  P", fontsize=LABEL_FS)
    #ax.set_ylabel("Output normalizer operator  Q", fontsize=LABEL_FS)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046)
    ticks = [t for t in [-1.0, -0.1, -0.01, 0.0, 0.01, 0.1, 1.0]
             if abs(t) >= linthresh or t == 0.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:g}" for t in ticks])
    cbar.ax.tick_params(labelsize=TICK_FS)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spectrum_figure(names, clean_vec, noisy_vec, rec_vec, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(names))
    w = 0.27

    ax.bar(x - w, clean_vec, w, label='Clean', color='#2ecc71')
    ax.bar(x, noisy_vec, w, label='Noisy', color='#e74c3c')
    ax.bar(x + w, rec_vec, w, label='Std QEC', color='#3498db')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=TICK_FS)
    ax.tick_params(axis='y', labelsize=TICK_FS)
    #ax.set_xlabel("Normalizer operator  P", fontsize=LABEL_FS)
    #ax.set_ylabel(r"Spectral coefficient  $\langle P \rangle$", fontsize=LABEL_FS)
    ax.legend(fontsize=TICK_FS)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', lw=0.7)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npz', default=None,
                    help="dephasing_results_*.npz to read meta_sigma / meta_rho_corr / meta_mu")
    ap.add_argument('--sigma', type=float, default=None)
    ap.add_argument('--rho-corr', type=float, default=None)
    ap.add_argument('--mu', type=float, default=None)
    ap.add_argument('--seed', type=int, default=0, help="RNG seed for the random logical state")
    ap.add_argument('--linthresh', type=float, default=1e-2,
                    help="SymLogNorm linear threshold (default: 0.01)")
    ap.add_argument('--petz', action='store_true',
                    help="also compute the linear Petz recovery-map PTM")
    ap.add_argument('--compare', default=None,
                    help="spectral_ptm_phaseflip_L*.npz -> extra figure K_qae - K_std")
    ap.add_argument('--out-prefix', default='ptm_stdqec_phaseflip')
    args = ap.parse_args()

    # ---- noise parameters
    sigma, rho_c, mu = 0.5, 0.3, 0.3
    if args.npz is not None:
        d = np.load(args.npz)
        sigma = float(d['meta_sigma'])
        rho_c = float(d['meta_rho_corr'])
        mu    = float(d['meta_mu'])
    if args.sigma    is not None: sigma = args.sigma
    if args.rho_corr is not None: rho_c = args.rho_corr
    if args.mu       is not None: mu    = args.mu
    print(f"Noise: sigma={sigma}, rho_corr={rho_c}, mu={mu}")

    PHI = build_phi(sigma, rho_c, mu=mu)
    max_im = float(np.max(np.abs(PHI.imag)))
    print(f"max|Im(PHI)| = {max_im:.3e}  "
          f"({'NON-diagonal / coherent noise' if max_im > 1e-12 else 'diagonal / dephasing noise'})")

    basis = build_normalizer_basis()
    for name, op in basis.items():
        herm_err = np.max(np.abs(op - op.conj().T))
        assert herm_err < 1e-8, f"{name} is not Hermitian (err={herm_err})"

    # ---- PTM of the standard recovery channel
    K_std, names, leak_std = channel_ptm(std_qec, basis)
    off, dia, xy = ptm_diagnostics(K_std, names)
    print(f"\nR_std : off-diag L1={off:.4f}  diag L1={dia:.4f}  "
          f"XL<->YL mixing L1={xy:.4f}  max leakage={leak_std.max():.3e}")
    print(f"--> The standard recovery-channel PTM is "
          f"{'NON-diagonal' if off > 1e-3 else 'diagonal'} on the normalizer subspace.")

    K_petz = None
    if args.petz:
        petz = make_petz_map(PHI)
        K_petz, _, leak_petz = channel_ptm(petz, basis)
        offp, diap, xyp = ptm_diagnostics(K_petz, names)
        print(f"R_petz: off-diag L1={offp:.4f}  diag L1={diap:.4f}  "
              f"XL<->YL mixing L1={xyp:.4f}  max leakage={leak_petz.max():.3e}")

    # ---- structure check: column P = I (projection into the code space)
    nz = [(n, K_std[i, 0]) for i, n in enumerate(names) if abs(K_std[i, 0]) > 1e-9]
    print("Column P=I of K(R_std):  " + "  ".join(f"{n}={v:+.3f}" for n, v in nz))

    # ---- random logical state: clean / noisy / recovered spectra
    rng = np.random.default_rng(args.seed)
    theta = np.arccos(1 - 2 * rng.random())
    phi_a = 2 * np.pi * rng.random()
    rho_clean = state_to_dm(logical_state(theta, phi_a))
    rho_noisy = dephasing_channel(rho_clean, PHI)
    rho_rec   = std_qec(rho_noisy)

    sc = spectrum(rho_clean, basis)
    sn = spectrum(rho_noisy, basis)
    sr = spectrum(rho_rec, basis)
    clean_vec = np.array([sc[n] for n in names])
    noisy_vec = np.array([sn[n] for n in names])
    rec_vec   = np.array([sr[n] for n in names])

    lin_err = float(np.max(np.abs(K_std @ noisy_vec - rec_vec)))
    nrm = float(np.real(np.trace(rho_noisy.conj().T @ rho_noisy)))
    leak_state = 1.0 - SYS_DIM * float(np.sum(noisy_vec ** 2)) / nrm
    f_noisy = float(np.real(np.trace(rho_noisy @ rho_clean)))
    f_rec   = float(np.real(np.trace(rho_rec   @ rho_clean)))
    print(f"\nRandom state (theta={theta:.3f}, phi={phi_a:.3f}): "
          f"F_noisy={f_noisy:.4f}  F_stdQEC={f_rec:.4f}")
    print(f"Linearity check: max|K @ noisy_spectrum - recovered_spectrum| = {lin_err:.2e}")
    print(f"Leakage of rho_noisy out of span(N(S)) = {leak_state:.3e}")

    # ---- figures
    panels = [(K_std, "std")]
    if K_petz is not None:
        panels.append((K_petz, "petz"))
    if args.compare is not None:
        dq = np.load(args.compare, allow_pickle=True)
        names_q = [str(s) for s in dq['names']]
        assert names_q == names, "basis ordering mismatch between the two PTM files"
        panels.append((dq['K'] - K_std, "diff_qae_minus_std"))

    for K, tag in panels:
        plot_ptm_figure(K, names, args.linthresh,
                        f"{args.out_prefix}_ptm_{tag}.pdf")

    plot_spectrum_figure(names, clean_vec, noisy_vec, rec_vec,
                         f"{args.out_prefix}_spectrum.pdf")


if __name__ == "__main__":
    main()