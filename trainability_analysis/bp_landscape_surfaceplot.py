# bp_landscape_surfaceplot.py
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

import warnings, gc
warnings.filterwarnings("ignore")
from functools import reduce
import numpy as np
import jax
jax.config.update("jax_enable_x64", False)          # match BP experiment (float32)
try:
    jax.config.update("jax_platform_name", "gpu"); jax.devices()
except Exception:
    jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import pennylane as qml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch

print(f"  JAX backend : {jax.default_backend().upper()}")

# ================================================================
# 0. Config
# ================================================================
OPT_FILE   = "data/rep_bitflip_L2_optima.npz"
FILE_TAG   = "rep_bitflip_L2"
L_FIXED    = 2
P_NOISE    = 0.05
N_ANC      = 1
CODE_QUBITS_FALLBACK = [3, 5, 7, 9]      # used only if meta not in npz

N_SCAN_STATES = 5                         # same small set as BP probe
SPAN          = np.pi                     # scan half-width around optimum
GRID_BY_N     = {3: 41, 5: 31, 7: 25, 9: 17}
CHUNK_BY_N    = {3: 5,  5: 5,  7: 5,  9: 5}
DIR_SEED      = 1234

# spectral weights — pin the spectral loss to the same range [0,1] as the fidelity loss
WS0, WX0, WY0, WZ0 = 1/2, 1/6, 1/6, 1/6

# NORM_MODE in {"shared", "per_surface", "raw"}
#   shared       -> center-relative, single common scale (FAIR flatness/depth)
#   raw          -> no scaling, real loss values on a shared z-axis
NORM_MODE = "shared"

C64, R32 = jnp.complex64, jnp.float32
I2 = np.eye(2, dtype=np.complex64)
X_ = np.array([[0,1],[1,0]], np.complex64)
Y_ = np.array([[0,-1j],[1j,0]], np.complex64)
Z_ = np.array([[1,0],[0,-1]], np.complex64)
def kron_list(o): return reduce(np.kron, o)

# ================================================================
# 1. Code + states (identical conventions to BP experiment)
# ================================================================
def build_code(n):
    def op_on(qs, m):
        ops = [I2]*n
        for q in qs: ops[q] = m
        return kron_list(ops).astype(np.complex64)
    S  = [op_on([i, i+1], Z_) for i in range(n-1)]
    ZL = op_on([0], Z_)
    XL = kron_list([X_]*n).astype(np.complex64)
    YL = kron_list([Y_]*n).astype(np.complex64)
    k0 = np.zeros(2**n, np.complex64); k0[0]  = 1.0
    k1 = np.zeros(2**n, np.complex64); k1[-1] = 1.0
    return S, ZL, XL, YL, k0, k1

def logical_dm(k0, k1, th, ph):
    psi = (np.cos(th/2)*k0 + np.exp(1j*ph)*np.sin(th/2)*k1).astype(np.complex64)
    psi = psi.reshape(-1,1)
    return (psi @ psi.conj().T).astype(np.complex64)

def bitflip(rho, p, n):
    r = rho.copy()
    for q in range(n):
        ops=[I2]*n; ops[q]=X_; E=kron_list(ops).astype(np.complex64)
        r = (1-p)*r + p*(E @ r @ E.conj().T)
    return r.astype(np.complex64)

def fib(m):
    g=(1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/m),(2*np.pi*i/g)%(2*np.pi)) for i in range(m)]

# ================================================================
# 2. Ansatz (identical to BP experiment)
# ================================================================
def count_params(L, n_total): return (L+1)*n_total*2 + L*n_total*(n_total-1)*3

def model_b(params, L, n_total):
    idx = 0
    for q in range(n_total):
        qml.RY(params[idx], wires=q); idx+=1
        qml.RZ(params[idx], wires=q); idx+=1
    for _ in range(L):
        for c in range(n_total):
            for t in range(n_total):
                if c!=t:
                    qml.CRX(params[idx],wires=[c,t]); idx+=1
                    qml.CRY(params[idx],wires=[c,t]); idx+=1
                    qml.CRZ(params[idx],wires=[c,t]); idx+=1
        for q in range(n_total):
            qml.RY(params[idx],wires=q); idx+=1
            qml.RZ(params[idx],wires=q); idx+=1

# ================================================================
# 3. Evaluator: fidelity_loss & spectral_full loss at a point
# ================================================================
def make_evaluator(n_code, L):
    n_total = n_code + N_ANC
    sys_dim, anc_dim = 2**n_code, 2**N_ANC
    S_np, ZL_np, XL_np, YL_np, k0, k1 = build_code(n_code)
    S_STACK = jnp.array(np.stack(S_np), C64)
    ZL = jnp.array(ZL_np, C64); XL = jnp.array(XL_np, C64); YL = jnp.array(YL_np, C64)
    ANC0   = jnp.zeros((anc_dim,anc_dim), C64).at[0,0].set(1.0+0j)
    FRESH0 = jnp.zeros((sys_dim,sys_dim), C64).at[0,0].set(1.0+0j)

    def circ(p): model_b(p, L, n_total)
    enc = qml.matrix(circ, wire_order=range(n_total))

    def qae(params, rho_code):
        V = jnp.asarray(enc(params), C64); Vd = jnp.conj(V.T)
        rho = jnp.kron(rho_code.astype(C64), ANC0)
        rho = V @ rho @ Vd
        r = rho.reshape(sys_dim, anc_dim, sys_dim, anc_dim)
        rho_a = jnp.trace(r, axis1=0, axis2=2)
        rho = jnp.kron(FRESH0, rho_a)
        rho = Vd @ rho @ V
        r2 = rho.reshape(sys_dim, anc_dim, sys_dim, anc_dim)
        return jnp.trace(r2, axis1=1, axis2=3).astype(C64)

    def exps(rho):
        smean = jnp.mean(jnp.real(jnp.einsum("kij,ji->k", S_STACK, rho)))
        zl = jnp.real(jnp.einsum("ij,ji->", ZL, rho))
        xl = jnp.real(jnp.einsum("ij,ji->", XL, rho))
        yl = jnp.real(jnp.einsum("ij,ji->", YL, rho))
        return smean, zl, xl, yl

    # per-state losses (same forms as BP experiment)
    def _losses(params, rn, rc, tx, ty, tz):
        rho = qae(params, rn)
        fid_loss = (1.0 - jnp.real(jnp.einsum("ij,ji->", rho, rc)))
        sm, zl, xl, yl = exps(rho)
        #spec = (WS0*(1.0-sm) + WX0*((tx-xl)/2.0)**2
        #        + WY0*((ty-yl)/2.0)**2 + WZ0*((tz-zl)/2.0)**2)
        spec = (WS0*(1.0-sm)/2 + WX0*((tx-xl)/2.0)**2
                + WY0*((ty-yl)/2.0)**2 + WZ0*((tz-zl)/2.0)**2)
        return fid_loss.astype(R32), spec.astype(R32)

    # vmap over a chunk of states for one parameter point
    _batched = jax.jit(jax.vmap(_losses, in_axes=(None,0,0,0,0,0)))
    return dict(n_params=count_params(L,n_total), batched=_batched,
                code=(S_np,ZL_np,XL_np,YL_np,k0,k1), n_code=n_code)

def build_states(ev, p):
    S_np, ZL_np, XL_np, YL_np, k0, k1 = ev["code"]; n=ev["n_code"]
    RN=[]; RC=[]; TX=[]; TY=[]; TZ=[]
    for th, ph in fib(N_SCAN_STATES):
        rc = logical_dm(k0, k1, th, ph); rn = bitflip(rc, p, n)
        RN.append(rn); RC.append(rc)
        TZ.append(float(np.real(np.trace(ZL_np @ rc))))
        TX.append(float(np.real(np.trace(XL_np @ rc))))
        TY.append(float(np.real(np.trace(YL_np @ rc))))
    return (jnp.array(np.stack(RN), C64), jnp.array(np.stack(RC), C64),
            jnp.array(np.array(TX), R32), jnp.array(np.array(TY), R32),
            jnp.array(np.array(TZ), R32))

def eval_point(ev, params, states, chunk):
    rn, rc, tx, ty, tz = states; m = rn.shape[0]
    F=[]; S=[]
    for k in range(int(np.ceil(m/chunk))):
        sl=slice(k*chunk, min((k+1)*chunk, m))
        fl, sp = ev["batched"](params, rn[sl], rc[sl], tx[sl], ty[sl], tz[sl])
        F.append(np.asarray(fl)); S.append(np.asarray(sp))
    return float(np.concatenate(F).mean()), float(np.concatenate(S).mean())

# ================================================================
# 4. 2D landscape scan around a center
# ================================================================
def scan_landscape(ev, center, states, grid, chunk, seed):
    n_p = center.shape[0]
    rng = np.random.default_rng(seed)
    d1 = rng.standard_normal(n_p); d1 /= np.linalg.norm(d1)
    d2 = rng.standard_normal(n_p); d2 -= (d2@d1)*d1; d2 /= np.linalg.norm(d2)
    axis = np.linspace(-SPAN, SPAN, grid)
    A, B = np.meshgrid(axis, axis)
    FID = np.empty_like(A); SPEC = np.empty_like(A)
    for i in range(grid):
        for j in range(grid):
            p = jnp.asarray(center + A[i,j]*d1 + B[i,j]*d2, R32)
            FID[i,j], SPEC[i,j] = eval_point(ev, p, states, chunk)
        gc.collect()
    return A, B, FID, SPEC

# ================================================================
# 5. Load optima and scan each n
# ================================================================
opt = np.load(OPT_FILE)
try:
    L_loaded = int(opt["meta_n_layers"]); n_sweep = [int(x) for x in opt["meta_n_sweep"]]
    if L_loaded != L_FIXED:
        print(f"  WARNING: npz n_layers={L_loaded} != L_FIXED={L_FIXED}; using {L_loaded}")
        L_FIXED = L_loaded
except KeyError:
    n_sweep = CODE_QUBITS_FALLBACK
    print(f"  meta not found; assuming n_sweep={n_sweep}, L={L_FIXED}")
print(f"  Loaded {OPT_FILE}  L={L_FIXED}  n_sweep={n_sweep}  NORM_MODE={NORM_MODE}")

results = {}
for n in n_sweep:
    ev = make_evaluator(n, L_FIXED)
    center = np.asarray(opt[f"n{n}_best_params"], np.float32)
    assert center.shape[0] == ev["n_params"], f"param mismatch n={n}: {center.shape[0]} vs {ev['n_params']}"
    states = build_states(ev, P_NOISE)
    grid = GRID_BY_N[n]; chunk = CHUNK_BY_N[n]
    print(f"  n={n}: scanning {grid}x{grid} grid ...", flush=True)
    A, B, FID, SPEC = scan_landscape(ev, center, states, grid, chunk, DIR_SEED+n)
    fid_c, spec_c = eval_point(ev, jnp.asarray(center, R32), states, chunk)
    results[n] = dict(A=A, B=B, FID=FID, SPEC=SPEC, fid_c=fid_c, spec_c=spec_c)
    print(f"       center: fid_loss={fid_c:.5f}  spec_loss={spec_c:.5f}  "
          f"| FID ptp={np.ptp(FID):.3e}  SPEC ptp={np.ptp(SPEC):.3e}")
    del ev, states; gc.collect()
    if hasattr(jax, "clear_caches"): jax.clear_caches()

np.savez(f"{FILE_TAG}_surfaces.npz",
         meta_norm_mode=np.array(NORM_MODE),
         **{f"n{n}_A":results[n]['A']    for n in results},
         **{f"n{n}_B":results[n]['B']    for n in results},
         **{f"n{n}_FID":results[n]['FID']  for n in results},
         **{f"n{n}_SPEC":results[n]['SPEC'] for n in results})
print(f"  Saved surfaces -> {FILE_TAG}_surfaces.npz")


# ================================================================
# 6. Scaling helpers per NORM_MODE
# ================================================================
# With WS0+WX0+WY0+WZ0 = 1 and every term in [0,1], BOTH losses are in
# [0,1] for all n -> "raw" is the fair choice. We pin a common z-limit
# so the across-n flattening signal is preserved.
GLOBAL_ZLIM = (0.0, 1.0) if NORM_MODE == "raw" else None

def prep_surfaces(FID, SPEC, mode):
    ci = FID.shape[0] // 2
    if mode == "per_surface":
        def n01(Z):
            lo, hi = Z.min(), Z.max()
            return (Z-lo)/(hi-lo) if hi > lo else np.zeros_like(Z)
        return n01(FID), n01(SPEC), "min-max per surface [0,1]  (NOT comparable)"
    if mode == "raw":
        return FID, SPEC, "raw loss value in [0,1]  (comparable across n)"
    # "shared": kept for reference, but NOTE: per-panel scale (not cross-n fair)
    Fr = FID - FID[ci, ci]; Sr = SPEC - SPEC[ci, ci]
    scale = max(np.ptp(Fr), np.ptp(Sr), 1e-12)
    return Fr/scale, Sr/scale, "ΔLoss / per-panel scale  (across-n NOT comparable)"

# ================================================================
# 7. 4-panel overlay
# ================================================================
TITLE = f"Repetition code  p={P_NOISE}, L={L_FIXED}"

fig = plt.figure(figsize=(16, 12))
zlabel_common = ""
for idx, n in enumerate(sorted(results.keys())):
    R = results[n]; A, B = R['A'], R['B']
    FIDp, SPECp, zlabel_common = prep_surfaces(R['FID'], R['SPEC'], NORM_MODE)
    ci = A.shape[0] // 2
    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    ax.plot_surface(A, B, FIDp,  cmap=cm.Reds,  alpha=0.55,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.plot_surface(A, B, SPECp, cmap=cm.Blues, alpha=0.55,
                    linewidth=0, antialiased=True, rstride=1, cstride=1)
    ax.scatter([0],[0],[FIDp[ci,ci]],  color='darkred', s=45, marker='o')
    ax.scatter([0],[0],[SPECp[ci,ci]], color='navy',    s=45, marker='^')
    #ax.set_title(f"n={n}  (fid_loss={R['fid_c']:.4f}, spec_loss={R['spec_c']:.4f})",
                 #fontweight='bold', fontsize=10)
    ax.set_xlabel("dir 1"); ax.set_ylabel("dir 2")
    ax.set_zlabel(zlabel_common, fontsize=8)
    if GLOBAL_ZLIM is not None:               # <-- common z-axis across ALL panels
        ax.set_zlim(*GLOBAL_ZLIM)
    ax.view_init(elev=30, azim=-60)

# Title at the very top
fig.suptitle(TITLE, fontsize=15, fontweight='bold', y=0.985)

# Legend moved to the BOTTOM so it no longer covers the title
fig.legend(handles=[Patch(color='#d62728', alpha=0.6, label='Fidelity loss'),
                    Patch(color='#1f77b4', alpha=0.6, label='Spectral loss')],
           loc='lower center', ncol=2, fontsize=11, frameon=False,
           bbox_to_anchor=(0.5, 0.01))

# Reserve clean bands for title (top) and legend (bottom)
fig.subplots_adjust(top=0.93, bottom=0.07, hspace=0.15, wspace=0.10)

outname = f"{FILE_TAG}_landscapes_{NORM_MODE}.png"
fig.savefig(outname, dpi=150)
print(f"  Saved -> {outname}")
print("  Done.")