# bp_scaling_rep_add11.py  — run n=11 (direct density-matrix engine).
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.65")
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
_x = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_enable_command_buffer=" not in _x:
    os.environ["XLA_FLAGS"] = (_x + " --xla_gpu_enable_command_buffer=").strip()

import warnings, gc, csv, time, shutil
warnings.filterwarnings("ignore")
from functools import reduce
from datetime import datetime, timedelta
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

# ---------------- config ----------------
N_CODE_NEW  = 11
L_FIXED     = 2
N_INITS     = 200         # set to e.g. 4 for a quick smoke test first
N_STATES    = 5
P_NOISE     = 0.05
N_REF       = 6
BATCH_RNG   = 5          
REF_CHUNK   = 3           

LOG_EVERY   = 20          # print a progress+ETA line every N inits
VERIFY      = True        # cross-check the new engine vs the old dense method on n=3
WS0, WX0, WY0, WZ0 = 1/2, 1/6, 1/6, 1/6

CSV_IN   = "data/bp_scaling_repetition.csv"   # source (n=3..9 data) 
CSV_OUT  = "bp_scaling_repetition.csv"        # NEW file, written into the current dir
PNG_PATH = "bp_scaling_repetition_with_n11.png"

C64, R32 = jnp.complex64, jnp.float32
I2 = np.eye(2, dtype=np.complex64)
X_ = np.array([[0,1],[1,0]], np.complex64)
Y_ = np.array([[0,-1j],[1j,0]], np.complex64)
Z_ = np.array([[1,0],[0,-1]], np.complex64)
MODES = ["fidelity", "spectral_full"]    

def kron_list(o): return reduce(np.kron, o)

def build_code(n):
    def op_on(qs, m):
        ops=[I2]*n
        for q in qs: ops[q]=m
        return kron_list(ops).astype(np.complex64)
    S=[op_on([i,i+1],Z_) for i in range(n-1)]
    ZL=op_on([0],Z_); XL=kron_list([X_]*n).astype(np.complex64); YL=kron_list([Y_]*n).astype(np.complex64)
    k0=np.zeros(2**n,np.complex64); k0[0]=1.0
    k1=np.zeros(2**n,np.complex64); k1[-1]=1.0
    return S,ZL,XL,YL,k0,k1

def logical_dm(k0,k1,th,ph):
    psi=(np.cos(th/2)*k0 + np.exp(1j*ph)*np.sin(th/2)*k1).astype(np.complex64).reshape(-1,1)
    return (psi@psi.conj().T).astype(np.complex64)

def bitflip(rho,p,n):
    r=rho.copy()
    for q in range(n):
        ops=[I2]*n; ops[q]=X_; E=kron_list(ops).astype(np.complex64)
        r=(1-p)*r + p*(E@r@E.conj().T)
    return r.astype(np.complex64)

def count_params(L,n_total): return (L+1)*n_total*2 + L*n_total*(n_total-1)*3

def fib(n):
    g=(1+np.sqrt(5))/2
    return [(np.arccos(1-2*(i+0.5)/n),(2*np.pi*i/g)%(2*np.pi)) for i in range(n)]

# original circuit (used ONLY by the verification block)
def model_b(params,L,n_total):
    idx=0
    for q in range(n_total):
        qml.RY(params[idx],wires=q); idx+=1
        qml.RZ(params[idx],wires=q); idx+=1
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

# ---------- low-memory engine: apply gates straight to rho, never build the 2^n x 2^n unitary ----------
def gate_list(L,n_total):
    g=[]; idx=0
    for q in range(n_total):
        g.append(("RY",(q,),idx)); idx+=1
        g.append(("RZ",(q,),idx)); idx+=1
    for _ in range(L):
        for c in range(n_total):
            for t in range(n_total):
                if c!=t:
                    g.append(("CRX",(c,t),idx)); idx+=1
                    g.append(("CRY",(c,t),idx)); idx+=1
                    g.append(("CRZ",(c,t),idx)); idx+=1
        for q in range(n_total):
            g.append(("RY",(q,),idx)); idx+=1
            g.append(("RZ",(q,),idx)); idx+=1
    return g

def _ry(th):
    h=0.5*th; c=jnp.cos(h); s=jnp.sin(h)
    return jnp.stack([c,-s,s,c]).astype(C64).reshape(2,2)
def _rz(th):
    h=0.5*th; e=jnp.exp(-1j*h); z=jnp.zeros_like(e)
    return jnp.stack([e,z,z,jnp.conj(e)]).reshape(2,2).astype(C64)
def _rx(th):
    h=0.5*th; c=jnp.cos(h).astype(C64); s=jnp.sin(h).astype(C64)
    return jnp.stack([c,-1j*s,-1j*s,c]).reshape(2,2).astype(C64)
def controlled(u2):
    z=jnp.zeros((2,2),C64)
    top=jnp.concatenate([jnp.eye(2,dtype=C64),z],axis=1)
    bot=jnp.concatenate([z,u2],axis=1)
    return jnp.concatenate([top,bot],axis=0)

def apply_1q(R,U,w,n):                       # rho -> U_w rho U_w^dagger  (tensor of rank 2n)
    nax=2*n; sub=list(range(nax)); NEW=nax
    R=jnp.einsum(U,[NEW,w],R,sub,[NEW if a==w else a for a in sub])
    Uc=jnp.conj(U)
    R=jnp.einsum(Uc,[NEW,n+w],R,sub,[NEW if a==(n+w) else a for a in sub])
    return R

def apply_2q(R,U4,c,t,n):
    U=U4.reshape(2,2,2,2); Uc=jnp.conj(U)
    nax=2*n; sub=list(range(nax)); A,B=nax,nax+1
    R=jnp.einsum(U,[A,B,c,t],R,sub,[A if a==c else (B if a==t else a) for a in sub])
    R=jnp.einsum(Uc,[A,B,n+c,n+t],R,sub,[A if a==(n+c) else (B if a==(n+t) else a) for a in sub])
    return R

def apply_circuit(R,params,gates,n,adjoint):
    seq=gates[::-1] if adjoint else gates    # V^dagger = reversed order, negated angles
    for kind,wires,i in seq:
        th=-params[i] if adjoint else params[i]
        if   kind=="RY":  R=apply_1q(R,_ry(th),wires[0],n)
        elif kind=="RZ":  R=apply_1q(R,_rz(th),wires[0],n)
        elif kind=="CRX": R=apply_2q(R,controlled(_rx(th)),wires[0],wires[1],n)
        elif kind=="CRY": R=apply_2q(R,controlled(_ry(th)),wires[0],wires[1],n)
        elif kind=="CRZ": R=apply_2q(R,controlled(_rz(th)),wires[0],wires[1],n)
    return R

def make_probe(n_code,L):
    n_total=n_code+1; sys_dim=2**n_code; anc_dim=2; D=2**n_total
    S_np,ZL_np,XL_np,YL_np,k0,k1=build_code(n_code)
    S_STACK=jnp.array(np.stack(S_np),C64)
    ZL=jnp.array(ZL_np,C64); XL=jnp.array(XL_np,C64); YL=jnp.array(YL_np,C64)
    ANC0=jnp.zeros((anc_dim,anc_dim),C64).at[0,0].set(1.0+0j)
    FRESH0=jnp.zeros((sys_dim,sys_dim),C64).at[0,0].set(1.0+0j)
    gates=gate_list(L,n_total); tshape=[2]*(2*n_total)

    def qae(params,rho_code):
        R=jnp.kron(rho_code.astype(C64),ANC0).reshape(tshape)
        R=apply_circuit(R,params,gates,n_total,False)                  # encode
        rho=R.reshape(D,D).reshape(sys_dim,anc_dim,sys_dim,anc_dim)
        rho_a=jnp.trace(rho,axis1=0,axis2=2)                           # keep ancilla
        R=jnp.kron(FRESH0,rho_a).reshape(tshape)                       # reset system to |0>
        R=apply_circuit(R,params,gates,n_total,True)                   # decode = V^dagger . V
        rho2=R.reshape(D,D).reshape(sys_dim,anc_dim,sys_dim,anc_dim)
        return jnp.trace(rho2,axis1=1,axis2=3).astype(C64)             # keep system

    def exps(rho):
        sm=jnp.mean(jnp.real(jnp.einsum("kij,ji->k",S_STACK,rho)))
        zl=jnp.real(jnp.einsum("ij,ji->",ZL,rho))
        xl=jnp.real(jnp.einsum("ij,ji->",XL,rho))
        yl=jnp.real(jnp.einsum("ij,ji->",YL,rho))
        return sm,zl,xl,yl

    def all_losses(params,rn,rc,tx,ty,tz):                             # 2 losses share one qae call
        rho=qae(params,rn)
        fid=1.0-jnp.real(jnp.einsum("ij,ji->",rho,rc))
        sm,zl,xl,yl=exps(rho)                                          # all four are still used by spectral_full
        spectral_full=WS0*(1.0-sm)/2 + WX0*((tx-xl)/2.0)**2 + WY0*((ty-yl)/2.0)**2 + WZ0*((tz-zl)/2.0)**2
        return jnp.stack([fid, spectral_full]).astype(R32)             # order MUST match MODES

    n_p=count_params(L,n_total)
    ref=np.arange(min(N_REF,n_p))
    basis=jnp.array(np.eye(n_p,dtype=np.float32)[ref],R32)

    def jvp_dir(p,e,rn,rc,tx,ty,tz):
        _,tang=jax.jvp(lambda pp: all_losses(pp,rn,rc,tx,ty,tz),(p,),(e,))
        return tang                                                    # (2,)
    jvp_refs=jax.jit(jax.vmap(jvp_dir,in_axes=(None,0,None,None,None,None,None)))
    return dict(n_total=n_total,n_p=n_p,ref=ref,basis=basis,qae=qae,jvp_refs=jvp_refs)

# ---------------- verification: new engine vs old dense method ----------------
def dense_qae_builder(n_code,L):
    n_total=n_code+1; sys_d=2**n_code; anc_d=2
    def circ(p): model_b(p,L,n_total)
    enc=qml.matrix(circ,wire_order=range(n_total))
    ANC0=np.zeros((anc_d,anc_d),np.complex64); ANC0[0,0]=1
    FRESH=np.zeros((sys_d,sys_d),np.complex64); FRESH[0,0]=1
    def dqae(p,rho_code):
        V=np.asarray(enc(p),np.complex64); Vd=V.conj().T
        rho=np.kron(rho_code,ANC0); rho=V@rho@Vd
        r=rho.reshape(sys_d,anc_d,sys_d,anc_d); rho_a=np.trace(r,axis1=0,axis2=2)
        rho=np.kron(FRESH,rho_a); rho=Vd@rho@V
        r2=rho.reshape(sys_d,anc_d,sys_d,anc_d); return np.trace(r2,axis1=1,axis2=3)
    return dqae

def verify_consistency(n_code_v=3,L=L_FIXED,tol=2e-4):
    pr=make_probe(n_code_v,L); dqae=dense_qae_builder(n_code_v,L)
    S,ZL,XL,YL,k0,k1=build_code(n_code_v)
    rc=logical_dm(k0,k1,0.7,1.1); rn=bitflip(rc,P_NOISE,n_code_v)
    p=np.random.default_rng(0).uniform(-np.pi,np.pi,pr["n_p"]).astype(np.float32)
    rm=np.asarray(pr["qae"](jnp.asarray(p,R32),jnp.asarray(rn,C64)))
    rd=dqae(p,rn); err=float(np.max(np.abs(rm-rd)))
    print(f"  [verify] n={n_code_v}: max|new-dense| = {err:.2e}  ->  {'OK' if err<tol else 'MISMATCH!'}")
    return err<tol

class RunningVar:
    def __init__(self,d): self.n=0; self.mean=np.zeros(d); self.M2=np.zeros(d)
    def update(self,x):
        x=np.asarray(x,float); self.n+=1
        d=x-self.mean; self.mean+=d/self.n; self.M2+=d*(x-self.mean)
    def var(self): return self.M2/self.n if self.n else np.zeros_like(self.mean)

def _fmt(sec):
    sec = int(max(0, sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"

# ---------------- run the new n point ----------------
def run_new_point(n_code):
    print(f"\n  === running n={n_code} (direct density-matrix engine) ===")
    S,ZL,XL,YL,k0,k1=build_code(n_code)
    states=[]
    for th,ph in fib(N_STATES):
        rc=logical_dm(k0,k1,th,ph); rn=bitflip(rc,P_NOISE,n_code)
        tz=float(np.real(np.trace(ZL@rc))); tx=float(np.real(np.trace(XL@rc))); ty=float(np.real(np.trace(YL@rc)))
        states.append((jnp.asarray(rn,C64),jnp.asarray(rc,C64),
                       jnp.float32(tx),jnp.float32(ty),jnp.float32(tz)))
    pr=make_probe(n_code,L_FIXED)
    n_p=pr["n_p"]; ref=pr["ref"]; basis=pr["basis"]; jvp_refs=pr["jvp_refs"]
    print(f"  n_params={n_p}, dim=2^{pr['n_total']}={2**pr['n_total']}, refs={len(ref)}, ref_chunk={REF_CHUNK}")
    rng=np.random.default_rng(1000+n_code)

    # --- FIX: one variance accumulator PER STATE (over inits), per mode ---
    #   gvar[m][si] accumulates the len(ref)-vector of gradient components
    #   across the N_INITS random initializations of state si only.
    gvar={m:[RunningVar(len(ref)) for _ in range(N_STATES)] for m in MODES}

    total_inits = N_STATES * N_INITS
    done_inits  = 0
    per_init    = []                     # steady-state times (excludes first/compiled init)
    t_start     = time.time()
    print(f"  total work: {N_STATES} states x {N_INITS} inits = {total_inits} inits "
          f"({len(ref)} JVPs each). First init includes one-time JIT compile.\n", flush=True)

    for si,(rn,rc,tx,ty,tz) in enumerate(states):
        t_state=time.time(); print(f"  --- state {si+1}/{N_STATES} ---", flush=True)
        done=0
        while done<N_INITS:
            b=min(BATCH_RNG,N_INITS-done)
            pblock=rng.uniform(-np.pi,np.pi,(b,n_p)).astype(np.float32)   # same RNG stream as your original
            for j in range(b):
                t_i=time.time()
                p=jnp.asarray(pblock[j],R32); cols=[]
                for s in range(0,len(ref),REF_CHUNK):
                    e=basis[s:s+REF_CHUNK]
                    cols.append(np.asarray(jax.device_get(jvp_refs(p,e,rn,rc,tx,ty,tz))))
                T=np.concatenate(cols,axis=0)               # (n_ref, 2)
                for mi,m in enumerate(MODES): gvar[m][si].update(T[:,mi])   # <-- per-state update
                dt=time.time()-t_i; done_inits+=1

                if done_inits==1:
                    print(f"    first init (incl. JIT compile): {_fmt(dt)}", flush=True)
                else:
                    per_init.append(dt)
                    if done_inits==2 or done_inits%LOG_EVERY==0 or done_inits==total_inits:
                        avg=float(np.mean(per_init[-50:]))
                        eta=(total_inits-done_inits)*avg
                        fin=datetime.now()+timedelta(seconds=eta)
                        print(f"    {done_inits:>4}/{total_inits} ({100*done_inits/total_inits:5.1f}%) "
                              f"| {avg*1000:6.0f} ms/init | elapsed {_fmt(time.time()-t_start)} "
                              f"| ETA {_fmt(eta)} -> ~{fin:%a %H:%M:%S}", flush=True)
            done+=b
        print(f"    state {si+1} done ({_fmt(time.time()-t_state)})", flush=True)

    # --- FIX: E_state[ Var_init(g) ] ---
    #   per state: variance over inits (len(ref)-vector) -> mean over components (scalar)
    #   then average those per-state scalars over the N_STATES states.
    out={}
    per_state_report={m:[] for m in MODES}
    for m in MODES:
        per_state_scalar=[float(np.mean(gvar[m][si].var())) for si in range(N_STATES)]
        per_state_report[m]=per_state_scalar
        out[m]=float(np.mean(per_state_scalar))

    print("\n  per-state Var_init (mean over components):")
    for m in MODES:
        vals="  ".join(f"{v:.2e}" for v in per_state_report[m])
        print(f"    {m:14s}: [{vals}]  -> mean {out[m]:.2e}")
    print("\n  RESULT  "+"  ".join(f"{m}={out[m]:.2e}" for m in MODES)
          +f"  | total {_fmt(time.time()-t_start)}", flush=True)
    del pr, jvp_refs; gc.collect()
    if hasattr(jax,"clear_caches"): jax.clear_caches()
    return n_p, out

# ---------------- CSV merge + scaling + plot ----------------
def load_existing(path):
    data={}
    if not os.path.exists(path):
        print(f"  [warn] {path} not found; creating new file (scaling needs >=2 points)."); return data
    rows=list(csv.reader(open(path,newline="")))
    if not rows: return data
    header=rows[0]
    try:
        col={name:i for i,name in enumerate(header)}
        vcols={m:col[m+"_var"] for m in MODES}; ip=col["n_params"]
    except KeyError:
        print("  [warn] unexpected header; creating fresh."); return data
    for row in rows[1:]:
        if not row or not row[0].strip(): break
        c0=row[0].strip()
        if not c0.lstrip("-").isdigit(): break
        try: data[int(c0)]={"n_params":int(row[ip]),**{m:float(row[vcols[m]]) for m in MODES}}
        except (ValueError,IndexError): break
    return data

def slope_r2(n,y):
    y=np.asarray(y,float); mk=(y>0)&np.isfinite(y)
    if mk.sum()<2: return (np.nan,np.nan,np.nan,np.nan)
    nn,ly=n[mk],np.log(y[mk])
    a,b=np.polyfit(nn,ly,1)
    r2e=1-np.sum((ly-(a*nn+b))**2)/np.sum((ly-ly.mean())**2)
    lp,kq=np.polyfit(np.log(nn),ly,1)
    r2p=1-np.sum((ly-(lp*np.log(nn)+kq))**2)/np.sum((ly-ly.mean())**2)
    return a,np.exp(a),r2e,r2p

def write_csv(path,data):
    ns=np.array(sorted(data),float)
    stats={m:slope_r2(ns,[data[int(nc)][m] for nc in ns]) for m in MODES}
    with open(path,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["n_code","n_params"]+[m+"_var" for m in MODES])
        for nc in sorted(data):
            w.writerow([nc,data[nc]["n_params"]]+[f"{data[nc][m]:.6e}" for m in MODES])
        w.writerow([]); w.writerow(["=== SCALING SUMMARY (Var ~ exp(a*n)) ==="])
        w.writerow(["metric"]+MODES)
        def srow(lab,idx): w.writerow([lab]+[f"{stats[m][idx]:+.4f}" for m in MODES])
        srow("slope_a_raw",0); srow("shrink_per_qubit_raw",1)
        srow("R2_exponential_raw",2); srow("R2_polynomial_raw",3)
    return stats

def make_plot(path,data,stats):
    ns=np.array(sorted(data),float)
    cmap={"fidelity":"#e74c3c","spectral_full":"#2ecc71"}
    mark={"fidelity":"s-","spectral_full":"o-"}
    fig,(a1,a2)=plt.subplots(1,2,figsize=(14,6))
    for m in MODES:
        y=np.array([data[int(nc)][m] for nc in ns],float)
        a1.semilogy(ns,y,mark[m],color=cmap[m],lw=2,ms=8,label=f"{m}  a={stats[m][0]:+.2f}")
        mk=(y>0)&np.isfinite(y)
        if mk.sum()>1:
            a,b=np.polyfit(ns[mk],np.log(y[mk]),1)
            a1.semilogy(ns,np.exp(a*ns+b),color=cmap[m],lw=1,ls=":",alpha=0.5)
    a1.set_xlabel("code qubits n"); a1.set_xticks(ns); a1.set_ylabel("Var[∂C]")
    a1.set_title("Var[∂C] decay (incl. n=11)"); a1.legend(fontsize=8); a1.grid(True,alpha=0.3,which="both")
    x=np.arange(len(MODES))
    a2.bar(x,[stats[m][0] for m in MODES],0.55,color=[cmap[m] for m in MODES])
    a2.set_xticks(x); a2.set_xticklabels(MODES,rotation=20,ha="right")
    a2.set_ylabel("decay slope a"); a2.set_title("Exponential decay slope per loss")
    a2.axhline(0,color="k",lw=0.8); a2.grid(True,alpha=0.3,axis="y")
    plt.tight_layout(); fig.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)

# ---------------- main ----------------
if VERIFY:
    if not verify_consistency(3,L_FIXED):
        print("  [verify] WARNING: engines disagree; n=11 may be inconsistent with n=3..9.")

try:
    n_p_new,out_new=run_new_point(N_CODE_NEW)
except Exception as ex:
    if "RESOURCE_EXHAUSTED" in str(ex) or "out of memory" in str(ex).lower():
        print("  [OOM] lower REF_CHUNK (try 2, then 1) and rerun.")
    raise

data=load_existing(CSV_IN)                       # reads the 4-col file, keeps only fidelity+spectral_full
if not data:
    print(f"  [warn] no rows read from {CSV_IN}; new CSV will hold only n={N_CODE_NEW}.")
if os.path.exists(CSV_OUT):                       # protect any previous run in the current dir
    shutil.copyfile(CSV_OUT,CSV_OUT+".bak"); print(f"  backup: {CSV_OUT}.bak")
data[N_CODE_NEW]={"n_params":n_p_new,**out_new}
stats=write_csv(CSV_OUT,data)                     # writes the NEW 2-col file here
make_plot(PNG_PATH,data,stats)

print("\n  | mode | slope a | e^a | R2_exp | R2_poly | verdict |")
print("  |---|---|---|---|---|---|")
for m in MODES:
    a,e,r2e,r2p=stats[m]
    verdict="exponential" if (np.isfinite(r2e) and np.isfinite(r2p) and r2e>=r2p) else "polynomial"
    print(f"  | {m} | {a:+.2f} | {e:.2f} | {r2e:.3f} | {r2p:.3f} | {verdict} |")
print(f"\n  Source CSV (untouched) : {CSV_IN}")
print(f"  New CSV written        : {CSV_OUT}")
print(f"  New plot               : {PNG_PATH}")