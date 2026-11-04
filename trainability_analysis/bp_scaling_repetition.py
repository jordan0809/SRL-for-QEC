# bp_scaling_repetition.py
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
_x = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _x:
    os.environ["XLA_FLAGS"] = (_x + " --xla_gpu_enable_command_buffer=").strip()

import warnings, gc, csv, time
warnings.filterwarnings("ignore")
from functools import reduce
import numpy as np
import jax
jax.config.update("jax_enable_x64", False)
try:
    jax.config.update("jax_platform_name", "gpu"); jax.devices()
except Exception:
    jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import pennylane as qml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"  JAX backend : {jax.default_backend().upper()}")

# ---- config ----
CODE_QUBITS = [3, 5, 7, 9]
L_FIXED     = 2
N_INITS     = 200
N_STATES    = 5
P_NOISE     = 0.05
N_REF       = 6
BATCH       = 5          
WS0, WX0, WY0, WZ0 = 1/2, 1/6, 1/6, 1/6   # spectral weights — pin the spectral loss to the same range [0,1] as the fidelity loss

C64, R32 = jnp.complex64, jnp.float32
I2 = np.eye(2, dtype=np.complex64)
X_ = np.array([[0,1],[1,0]], np.complex64)
Y_ = np.array([[0,-1j],[1j,0]], np.complex64)
Z_ = np.array([[1,0],[0,-1]], np.complex64)

def kron_list(o): return reduce(np.kron, o)

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

class RunningVar:
    def __init__(self, d): self.n=0; self.mean=np.zeros(d); self.M2=np.zeros(d)
    def update(self, x):
        x=np.asarray(x,float); self.n+=1
        d=x-self.mean; self.mean+=d/self.n; self.M2+=d*(x-self.mean)
    def update_batch(self, X):
        for row in np.asarray(X, float): self.update(row)
    def var(self): return self.M2/self.n if self.n else np.zeros_like(self.mean)

def make_probe(n_code, L):
    n_total = n_code + 1
    sys_dim, anc_dim = 2**n_code, 2
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
        # mean over stabilizers -> cancels the k(n)~n-1 term-count (slope-safe)
        smean = jnp.mean(jnp.real(jnp.einsum("kij,ji->k", S_STACK, rho)))
        zl = jnp.real(jnp.einsum("ij,ji->", ZL, rho))
        xl = jnp.real(jnp.einsum("ij,ji->", XL, rho))
        yl = jnp.real(jnp.einsum("ij,ji->", YL, rho))
        return smean, zl, xl, yl

    def C_fidelity(params, rn, rc, tx, ty, tz):
        rho = qae(params, rn)
        return (1.0 - jnp.real(jnp.einsum("ij,ji->", rho, rc))).astype(R32)

    def _spec(params, rn, tx, ty, tz, wS, wX, wY, wZ):
        rho = qae(params, rn); sm, zl, xl, yl = exps(rho)
        return (wS*(1.0-sm)/2 + wX*((tx-xl)/2.0)**2
                + wY*((ty-yl)/2.0)**2 + wZ*((tz-zl)/2.0)**2).astype(R32)

    def C_spectral_full(params, rn, rc, tx, ty, tz):
        return _spec(params, rn, tx, ty, tz, WS0, WX0, WY0, WZ0)
    def C_stab_Z(params, rn, rc, tx, ty, tz):
        return _spec(params, rn, tx, ty, tz, 1.0, 0.0, 0.0, 1.0)
    def C_logical_global(params, rn, rc, tx, ty, tz):
        return _spec(params, rn, tx, ty, tz, 0.0, 1.0, 1.0, 0.0)

    n_p = count_params(L, n_total)
    ref = np.arange(min(N_REF, n_p))
    basis = jnp.array(np.eye(n_p, dtype=np.float32)[ref], R32)

    # forward-mode JVP (memory-light: no tape), vmapped over a BATCH of inits.
    def make_jvp(fn):
        def one(p, e, rn, rc, tx, ty, tz):
            _, tang = jax.jvp(lambda pp: fn(pp, rn, rc, tx, ty, tz), (p,), (e,))
            return tang.astype(R32)
        batched = jax.vmap(one, in_axes=(0, None, None, None, None, None, None))
        return jax.jit(batched)

    fns = {"fidelity":      make_jvp(C_fidelity),
           "spectral_full": make_jvp(C_spectral_full),
           "SZ_local":      make_jvp(C_stab_Z),
           "XY_global":     make_jvp(C_logical_global)}
    return n_p, ref, basis, fns

def fib(n):
    g=(1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/n),(2*np.pi*i/g)%(2*np.pi)) for i in range(n)]

MODES = ["fidelity", "spectral_full"]
res = {"n_code": [], "n_params": []}
for m in MODES:
    res[m+"_var"] = []

for n_code in CODE_QUBITS:
    S_np, ZL_np, XL_np, YL_np, k0, k1 = build_code(n_code)
    states = []
    for th, ph in fib(N_STATES):
        rc = logical_dm(k0, k1, th, ph)
        rn = bitflip(rc, P_NOISE, n_code)
        tz = float(np.real(np.trace(ZL_np @ rc)))
        tx = float(np.real(np.trace(XL_np @ rc)))
        ty = float(np.real(np.trace(YL_np @ rc)))
        states.append((jnp.array(rn, C64), jnp.array(rc, C64),
                       jnp.float32(tx), jnp.float32(ty), jnp.float32(tz)))

    n_p, ref, basis, gf = make_probe(n_code, L_FIXED)
    rng = np.random.default_rng(1000 + n_code)

    # ---- CLEANER: per-state Var over random inits, then average across states ----
    # For each mode we accumulate, per state, the component-wise variance over the
    # N_INITS random inits; then we mean over states and over the ref components.
    # This isolates the random-init gradient variance (the BP quantity) from the
    # between-state variation that pooling would otherwise inject into the slope.
    per_state_var = {m: [] for m in MODES}   # list of length N_STATES, each = var vector (len ref)
    t0 = time.time()
    for (rn, rc, tx, ty, tz) in states:
        rv = {m: RunningVar(len(ref)) for m in MODES}   # fresh per state
        done = 0
        while done < N_INITS:
            b = min(BATCH, N_INITS - done); done += b
            pbatch = jnp.asarray(
                rng.uniform(-np.pi, np.pi, (b, n_p)).astype(np.float32), R32)
            for m in MODES:
                grad_cols = np.empty((b, len(ref)), np.float64)
                for ri in range(len(ref)):
                    tang_b = gf[m](pbatch, basis[ri], rn, rc, tx, ty, tz)
                    grad_cols[:, ri] = np.asarray(jax.device_get(tang_b))
                rv[m].update_batch(grad_cols)
        for m in MODES:
            per_state_var[m].append(rv[m].var())        # var over inits for THIS state

    res["n_code"].append(n_code); res["n_params"].append(n_p)
    line = f"  n={n_code} (n_p={n_p}): "
    for m in MODES:
        # mean over states, then mean over ref components
        v = float(np.mean(np.mean(np.stack(per_state_var[m], axis=0), axis=0)))
        res[m+"_var"].append(v)
        line += f"{m}={v:.2e}  "
    print(line + f"({time.time()-t0:.1f}s)")
    del gf, basis, per_state_var; gc.collect()
    if hasattr(jax, "clear_caches"): jax.clear_caches()

# ---------- scaling analysis: exponential slope + exp-vs-poly R^2 ----------
n = np.array(res["n_code"], float)

def slope_r2(y):
    """Returns (slope_a, shrink_e^a, R2_exponential, R2_polynomial)."""
    y = np.array(y, float); mk = (y > 0) & np.isfinite(y)
    if mk.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    nn, ly = n[mk], np.log(y[mk])
    a, b = np.polyfit(nn, ly, 1)                          # semilog -> exponential
    r2e = 1 - np.sum((ly-(a*nn+b))**2)/np.sum((ly-ly.mean())**2)
    lp, kq = np.polyfit(np.log(nn), ly, 1)                # log-log -> polynomial
    r2p = 1 - np.sum((ly-(lp*np.log(nn)+kq))**2)/np.sum((ly-ly.mean())**2)
    return a, np.exp(a), r2e, r2p

stats = {}
for m in MODES:
    aR, eR, r2eR, r2pR = slope_r2(res[m+"_var"])
    stats[m] = dict(slope_raw=aR, shrink_raw=eR, r2exp_raw=r2eR, r2poly_raw=r2pR)

# ---------- CSV: per-n table + integrated slope/R^2 summary ----------
with open("bp_scaling_repetition.csv", "w", newline="") as f:
    w = csv.writer(f)
    # section 1: per-n data
    w.writerow(["n_code", "n_params"] + [m+"_var" for m in MODES])
    for i in range(len(n)):
        w.writerow([res["n_code"][i], res["n_params"][i]]
                   + [f"{res[m+'_var'][i]:.6e}" for m in MODES])
    # section 2: scaling summary
    w.writerow([])
    w.writerow(["=== SCALING SUMMARY (Var ~ exp(a*n)) ==="])
    w.writerow(["metric"] + MODES)
    def srow(label, key): w.writerow([label] + [f"{stats[m][key]:+.4f}" for m in MODES])
    srow("slope_a_raw",         "slope_raw")
    srow("shrink_per_qubit_raw","shrink_raw")
    srow("R2_exponential_raw",  "r2exp_raw")
    srow("R2_polynomial_raw",   "r2poly_raw")

# console markdown summary
print("\n  | mode | slope a (raw) | e^a | R2_exp | R2_poly | verdict |")
print("  |---|---|---|---|---|---|")
for m in MODES:
    s = stats[m]
    verdict = "exponential" if s["r2exp_raw"] >= s["r2poly_raw"] else "polynomial"
    print(f"  | {m} | {s['slope_raw']:+.2f} | {s['shrink_raw']:.2f} | "
          f"{s['r2exp_raw']:.3f} | {s['r2poly_raw']:.3f} | {verdict} |")

# ---------- plots: raw semilog + slope bars ----------
cmap = {"fidelity":"#e74c3c","spectral_full":"#2ecc71"}
mark = {"fidelity":"s-","spectral_full":"o-"}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

for m in MODES:
    y = np.array(res[m+"_var"], float)
    a1.semilogy(n, y, mark[m], color=cmap[m], lw=2, ms=8,
                label=f"{m}  a={stats[m]['slope_raw']:+.2f}")
    mk = (y > 0) & np.isfinite(y)
    if mk.sum() > 1:                              # exponential fit line
        a, b = np.polyfit(n[mk], np.log(y[mk]), 1)
        a1.semilogy(n, np.exp(a*n+b), color=cmap[m], lw=1, ls=":", alpha=0.5)
a1.set_xlabel("code qubits n"); a1.set_xticks(n)
a1.set_ylabel("Var[∂C]  (per-state, avg over states)")
a1.set_title("Var[∂C] decay")
a1.legend(fontsize=8); a1.grid(True, alpha=0.3, which="both")

x = np.arange(len(MODES)); wbar = 0.55
a2.bar(x, [stats[m]["slope_raw"] for m in MODES], wbar,
       color=[cmap[m] for m in MODES])
a2.set_xticks(x); a2.set_xticklabels(MODES, rotation=20, ha="right")
a2.set_ylabel("decay slope a")
a2.set_title("Exponential decay slope per loss")
a2.axhline(0, color="k", lw=0.8); a2.grid(True, alpha=0.3, axis="y")

plt.tight_layout(); fig.savefig("bp_scaling_repetition.png", dpi=150, bbox_inches="tight")
print("\n  Saved : bp_scaling_repetition.csv / .png")
print("  Story: raw slope = scale-fair BP evidence; R2_exp vs R2_poly = exponential vs polynomial decay.")