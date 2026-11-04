"""
Spectral QEC — training only. *(for generating barren plateaus surface plot))
Bit-flip repetition code [[n,1,1]] for n in {3,5,7,9}, bit-flip noise p=0.05.
Fixed n_layers=2 . Saves ONLY optimal params.
"""
import warnings; warnings.filterwarnings("ignore")
import os, gc, time
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
import optax, pennylane as qml, numpy as np
from functools import reduce

print(f"  JAX backend : {jax.default_backend().upper()}")

# ================================================================
# 1. Config
# ================================================================
N_CODE_SWEEP = [3, 5, 7, 9]
P_BITFLIP    = 0.05
N_ANC        = 1
N_LAYERS     = 2                      # FIXED (must match surface-plot ansatz)

N_TRAIN_STATES = 20                 

# per-n memory knobs (n=9 -> 10 qubits, 1024-dim; keep batch tiny + remat)
BATCH_BY_N = {3: 5, 5: 5, 7: 3, 9: 1}
CHUNK_BY_N = {3: 100, 5: 50, 7: 25, 9: 10}
REMAT_BY_N = {3: False, 5: False, 7: True, 9: True}

LR                 = 0.05
N_INIT_ROUNDS      = 12
INIT_EPOCHS        = 20
MAIN_EPOCHS        = 180
TOTAL_EPOCHS       = INIT_EPOCHS + MAIN_EPOCHS
WEIGHT_UPDATE_FREQ = 5

W_STAB_HIGH = 5.0;  W_STAB_LOW = 0.5
W_ZL_LOW = 5.0; W_ZL_HIGH = 20.0
W_XL_LOW = 5.0; W_XL_HIGH = 20.0
W_YL_LOW = 5.0; W_YL_HIGH = 20.0
STAB_SAT_LOW = 0.70; STAB_SAT_HIGH = 0.99; MAX_W_DELTA = 1.0
W_STAB_BASE = 1.0; W_ZL_BASE = 10.0; W_XL_BASE = 10.0; W_YL_BASE = 10.0

FILE_TAG = "rep_bitflip_L2"

# ================================================================
# 2. Pauli primitives
# ================================================================
I2 = np.eye(2, dtype=complex)
X_ = np.array([[0,1],[1,0]], complex)
Y_ = np.array([[0,-1j],[1j,0]], complex)
Z_ = np.array([[1,0],[0,-1]], complex)
def kron_list(ops): return reduce(np.kron, ops)

# ================================================================
# 3. Build code operators (numpy) + JAX stacks
# ================================================================
def build_code(n):
    N_TOTAL = n + 1; SYS_DIM = 2**n; ANC_DIM = 2**N_ANC
    stabs = []
    for i in range(n-1):
        ops=[I2]*n; ops[i]=Z_; ops[i+1]=Z_; stabs.append(kron_list(ops))
    XL = kron_list([X_]*n); YL = kron_list([Y_]*n); ZL = kron_list([Z_]*n)
    ket0 = np.zeros(SYS_DIM, complex); ket0[0]=1.0
    ket1 = np.zeros(SYS_DIM, complex); ket1[-1]=1.0
    X_emb = [kron_list([X_ if q==i else I2 for i in range(n)]) for q in range(n)]
    return dict(n=n, N_TOTAL=N_TOTAL, SYS_DIM=SYS_DIM, ANC_DIM=ANC_DIM,
                stabs=stabs, XL=XL, YL=YL, ZL=ZL, ket0=ket0, ket1=ket1, X_emb=X_emb)

def logical_state(C, th, ph):
    return np.cos(th/2)*C['ket0'] + np.exp(1j*ph)*np.sin(th/2)*C['ket1']
def state_to_dm(psi):
    psi = psi.reshape(-1,1); return psi @ psi.conj().T

# ================================================================
# 4. Ansatz
# ================================================================
def count_params(n_layers, n_total):
    return (n_layers+1)*n_total*2 + n_layers*n_total*(n_total-1)*3

def model_b_ansatz(params, n_layers, n_total):
    idx = 0
    for q in range(n_total):
        qml.RY(params[idx], wires=q); idx+=1
        qml.RZ(params[idx], wires=q); idx+=1
    for _ in range(n_layers):
        for c in range(n_total):
            for t in range(n_total):
                if c!=t:
                    qml.CRX(params[idx], wires=[c,t]); idx+=1
                    qml.CRY(params[idx], wires=[c,t]); idx+=1
                    qml.CRZ(params[idx], wires=[c,t]); idx+=1
        for q in range(n_total):
            qml.RY(params[idx], wires=q); idx+=1
            qml.RZ(params[idx], wires=q); idx+=1
    return idx

# ================================================================
# 5. Per-code JAX components (memory-hardened)
# ================================================================
def make_components(C, n_layers, use_remat):
    n_total=C['N_TOTAL']; SYS_DIM=C['SYS_DIM']; ANC_DIM=C['ANC_DIM']
    n_p = count_params(n_layers, n_total); N_STAB = len(C['stabs'])

    S_STACK = jnp.array(np.stack(C['stabs']))                       # (n-1,S,S)
    L_STACK = jnp.array(np.stack([C['ZL'], C['XL'], C['YL']]))      # (3,S,S)
    X_EMB   = jnp.array(np.stack(C['X_emb']))                       # (n,S,S)
    ANC_0   = jnp.zeros((ANC_DIM,ANC_DIM), complex).at[0,0].set(1.0)
    FRESH_0 = jnp.zeros((SYS_DIM,SYS_DIM), complex).at[0,0].set(1.0)

    def expect(stack, rho): return jnp.real(jnp.einsum('kij,ji->k', stack, rho))
    def pt_sys(r):
        r=r.reshape(SYS_DIM,ANC_DIM,SYS_DIM,ANC_DIM); return jnp.trace(r,axis1=0,axis2=2)
    def pt_anc(r):
        r=r.reshape(SYS_DIM,ANC_DIM,SYS_DIM,ANC_DIM); return jnp.trace(r,axis1=1,axis2=3)

    # bit-flip channel on a single code DM (chunked at call site)
    @jax.jit
    def depol_single(rho, p):
        res = rho
        for q in range(C['n']):
            Xq = X_EMB[q]
            res = (1.0-p)*res + p*(Xq @ res @ Xq)
        return res

    def circuit(params): model_b_ansatz(params, n_layers, n_total)
    encoder_fn = qml.matrix(circuit, wire_order=range(n_total))

    def _fwd(V, Vd, rho_code):
        rho = jnp.kron(rho_code, ANC_0)
        rho = V @ rho @ Vd
        rho_a = pt_sys(rho)
        rho = jnp.kron(FRESH_0, rho_a)
        rho = Vd @ rho @ V
        return pt_anc(rho)
    _vmapped = jax.vmap(_fwd, in_axes=(None,None,0))

    @jax.jit
    def qae_batch(params, rho_chunk):
        V = encoder_fn(params); Vd = jnp.conj(V.T)
        return _vmapped(V, Vd, rho_chunk)

    def _loss_core(p, rho_b, tgt_b, ws, wz, wx, wy):
        V = encoder_fn(p); Vd = jnp.conj(V.T)
        outs = _vmapped(V, Vd, rho_b)
        def per(rho, tgt):
            s = expect(S_STACK, rho); l = expect(L_STACK, rho)   # l=[zl,xl,yl]
            return (ws*(-jnp.sum(s)) + wz*(tgt[0]-l[0])**2
                    + wx*(tgt[1]-l[1])**2 + wy*(tgt[2]-l[2])**2)
        return jnp.mean(jax.vmap(per)(outs, tgt_b))
    loss_core = jax.checkpoint(_loss_core) if use_remat else _loss_core

    @jax.jit
    def batch_loss_and_grad(params, rho_b, tgt_b, ws, wz, wx, wy):
        return jax.value_and_grad(loss_core)(params, rho_b, tgt_b, ws, wz, wx, wy)

    @jax.jit
    def batch_stab_avg(params, rho_chunk):
        out = qae_batch(params, rho_chunk)
        per = lambda r: jnp.sum(expect(S_STACK, r)) / N_STAB
        return jnp.mean(jax.vmap(per)(out))

    @jax.jit
    def batch_baseline_loss(params, rho_chunk, tgt_chunk):
        out = qae_batch(params, rho_chunk)
        def per(rho, tgt):
            s = expect(S_STACK, rho); l = expect(L_STACK, rho)
            return (W_STAB_BASE*(-jnp.sum(s)) + W_ZL_BASE*(tgt[0]-l[0])**2
                    + W_XL_BASE*(tgt[1]-l[1])**2 + W_YL_BASE*(tgt[2]-l[2])**2)
        return jax.vmap(per)(out, tgt_chunk)

    return dict(n_params=n_p, depol_single=depol_single,
                batch_loss_and_grad=batch_loss_and_grad,
                batch_stab_avg=batch_stab_avg, batch_baseline_loss=batch_baseline_loss)

# ================================================================
# 6. Dataset (training only)
# ================================================================
def fibonacci_sphere(n):
    g=(1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/n),(2*np.pi*i/g)%(2*np.pi)) for i in range(n)]

def make_train_data(C, comps, p, chunk):
    angles = fibonacci_sphere(N_TRAIN_STATES)
    clean = np.stack([state_to_dm(logical_state(C, th, ph)) for th,ph in angles])
    # bit-flip noise, chunked on device
    pj = jnp.float64(p); f = jax.vmap(lambda r: comps['depol_single'](r, pj))
    noisy = np.empty_like(clean)
    for k in range(int(np.ceil(len(clean)/chunk))):
        sl = slice(k*chunk, min((k+1)*chunk, len(clean)))
        noisy[sl] = np.array(f(jnp.array(clean[sl])))
    tgt = np.array([[float(np.real(np.trace(C['ZL']@clean[i]))),
                     float(np.real(np.trace(C['XL']@clean[i]))),
                     float(np.real(np.trace(C['YL']@clean[i])))] for i in range(len(clean))])
    return jnp.array(noisy), jnp.array(tgt)

# ================================================================
# 7. Adaptive weights + epoch + chunked probes
# ================================================================
def compute_target_weights(avg_stab):
    sat = float(np.clip((avg_stab-STAB_SAT_LOW)/(STAB_SAT_HIGH-STAB_SAT_LOW),0,1))
    return (W_STAB_HIGH-(W_STAB_HIGH-W_STAB_LOW)*sat,
            W_ZL_LOW+(W_ZL_HIGH-W_ZL_LOW)*sat,
            W_XL_LOW+(W_XL_HIGH-W_XL_LOW)*sat,
            W_YL_LOW+(W_YL_HIGH-W_YL_LOW)*sat)
def step_weight(c,t): return c+float(np.clip(t-c,-MAX_W_DELTA,MAX_W_DELTA))

def stab_mean_chunked(bsa, params, rn, chunk):
    n=rn.shape[0]; tot=0.0; cnt=0
    for k in range(int(np.ceil(n/chunk))):
        sl=slice(k*chunk,min((k+1)*chunk,n)); w=sl.stop-sl.start
        tot+=float(bsa(params, rn[sl]))*w; cnt+=w
    return tot/max(cnt,1)

def baseline_loss_chunked(bbl, params, rn, tgt, chunk):
    n=rn.shape[0]; acc=[]
    for k in range(int(np.ceil(n/chunk))):
        sl=slice(k*chunk,min((k+1)*chunk,n))
        acc.append(np.array(bbl(params, rn[sl], tgt[sl])))
    return float(np.concatenate(acc).mean())

def run_epoch(params, opt_state, optimizer, ws,wz,wx,wy, rng, blg, rn, tgt, batch):
    jw = tuple(map(jnp.float64,(ws,wz,wx,wy)))
    steps=int(np.ceil(N_TRAIN_STATES/batch)); rng,sk=jax.random.split(rng)
    idx=np.array(jax.random.permutation(sk, N_TRAIN_STATES)); ep=0.0; nb=0
    for st in range(steps):
        b=idx[st*batch:(st+1)*batch]
        if len(b)==0: continue
        lv,gv=blg(params, rn[b], tgt[b], *jw)
        gv=jnp.where(jnp.isnan(gv),0.0,gv); lv=jnp.where(jnp.isnan(lv),0.0,lv)
        upd,opt_state=optimizer.update(gv,opt_state,params)
        params=optax.apply_updates(params,upd); ep+=float(lv); nb+=1
    return params, opt_state, ep/max(nb,1), rng

# ================================================================
# 8. Train one code (fixed n_layers), selection by baseline spectral loss
# ================================================================
def train_code(C):
    n=C['n']; batch=BATCH_BY_N[n]; chunk=CHUNK_BY_N[n]; use_remat=REMAT_BY_N[n]
    comps = make_components(C, N_LAYERS, use_remat)
    blg=comps['batch_loss_and_grad']; bsa=comps['batch_stab_avg']; bbl=comps['batch_baseline_loss']
    n_p=comps['n_params']
    rn, tgt = make_train_data(C, comps, P_BITFLIP, chunk)
    print(f"\n{'='*60}\n  n={n}  N_TOTAL={C['N_TOTAL']}  params={n_p}"
          f"  batch={batch}  remat={use_remat}\n{'='*60}")

    # warm-up JIT
    _p0=jnp.zeros(n_p)
    blg(_p0, rn[:batch], tgt[:batch], *(jnp.float64(1.0),)*4)

    spe=int(np.ceil(N_TRAIN_STATES/batch))
    optimizer=optax.chain(optax.clip_by_global_norm(1.0),
        optax.adam(optax.cosine_decay_schedule(LR, TOTAL_EPOCHS*spe)))
    rng=jax.random.PRNGKey(42+n)

    best_loss=np.inf; best_p=best_opt=None
    best_w=(W_STAB_HIGH,W_ZL_LOW,W_XL_LOW,W_YL_LOW); best_rng=rng

    def phase1(tp, rr):
        to=optimizer.init(tp); ws,wz,wx,wy=W_STAB_HIGH,W_ZL_LOW,W_XL_LOW,W_YL_LOW
        for ep in range(INIT_EPOCHS):
            if ep%WEIGHT_UPDATE_FREQ==0:
                tw=compute_target_weights(stab_mean_chunked(bsa,tp,rn,chunk))
                ws,wz,wx,wy=(step_weight(ws,tw[0]),step_weight(wz,tw[1]),
                             step_weight(wx,tw[2]),step_weight(wy,tw[3]))
            tp,to,_,rr=run_epoch(tp,to,optimizer,ws,wz,wx,wy,rr,blg,rn,tgt,batch)
        return tp,to,(ws,wz,wx,wy),rr

    for rd in range(N_INIT_ROUNDS):
        rng,sk=jax.random.split(rng); s=rd%4
        if   s==0: tp=jax.random.uniform(sk,(n_p,),minval=-np.pi,maxval=np.pi)
        elif s==1: tp=jax.random.normal(sk,(n_p,))*0.15
        elif s==2: tp=jax.random.normal(sk,(n_p,))*1.0
        else:      tp=jax.random.uniform(sk,(n_p,),minval=-0.1,maxval=0.1)
        rng,rr=jax.random.split(rng)
        tp,to,w,rr=phase1(tp,rr)
        L=baseline_loss_chunked(bbl,tp,rn,tgt,chunk); star=""
        if L<best_loss:
            best_loss,best_p,best_opt,best_w,best_rng=L,tp,to,w,rr; star=" *"
        print(f"  round {rd:2d}  spec_loss={L:.4f}{star}")
        if star=="": del to
        gc.collect()
    print(f"  best warm-up spec_loss={best_loss:.4f}")

    params,opt_state=best_p,best_opt; ws,wz,wx,wy=best_w; rng_key=best_rng
    for ep in range(MAIN_EPOCHS):
        t0=time.time()
        if ep%WEIGHT_UPDATE_FREQ==0:
            tw=compute_target_weights(stab_mean_chunked(bsa,params,rn,chunk))
            ws,wz,wx,wy=(step_weight(ws,tw[0]),step_weight(wz,tw[1]),
                         step_weight(wx,tw[2]),step_weight(wy,tw[3]))
        params,opt_state,_,rng_key=run_epoch(params,opt_state,optimizer,
            ws,wz,wx,wy,rng_key,blg,rn,tgt,batch)
        if ep%20==0 or ep==MAIN_EPOCHS-1:
            L=baseline_loss_chunked(bbl,params,rn,tgt,chunk)
            print(f"  ep {INIT_EPOCHS+ep:3d}  spec_loss={L:.4f}  {time.time()-t0:.1f}s")
        if ep%20==0: gc.collect()

    final=np.array(params)
    del rn, tgt, comps, blg, bsa, bbl; gc.collect()
    return dict(n=n, n_total=C['N_TOTAL'], n_layers=N_LAYERS,
                params=final, n_params=n_p)

# ================================================================
# 9. Sweep over n + save optima only
# ================================================================
save_dict = {'meta_p': np.array(P_BITFLIP),
             'meta_n_sweep': np.array(N_CODE_SWEEP),
             'meta_n_layers': np.array(N_LAYERS)}
for n in N_CODE_SWEEP:
    C = build_code(n)
    res = train_code(C)
    save_dict[f"n{n}_best_params"] = res['params']
    save_dict[f"n{n}_n_layers"]    = np.array(res['n_layers'])
    save_dict[f"n{n}_n_total"]     = np.array(res['n_total'])
    del C, res; gc.collect()

np.savez(f"{FILE_TAG}_optima.npz", **save_dict)
print(f"\n  Saved optima -> {FILE_TAG}_optima.npz")
print("  Keys:", [k for k in save_dict if k.startswith('n')])
print("  Done.")