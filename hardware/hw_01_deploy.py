"""hw_01_deploy.py

NOTHING IS INJECTED. The external channel is the device's own idle: one
`delay` per storage interval, two intervals per shot, mu = 0.

FOUR BASELINES (variants), all sharing the SAME wall clock per shot:
    bare        1 physical qubit    + idle + idle
    uncorr      logical (encoded)   + idle + idle + SOFTWARE ideal recovery
    qae_*       logical             + idle + QAE unitary + idle + SOFTWARE
                                       ideal recovery on the QAE output
    qec         logical             + idle + REAL mid-circuit syndrome
                                       measurement + conditional Z + idle +
                                       SOFTWARE ideal recovery on the residual
"""
import argparse, ast, csv, os
import numpy as np
import qae_hw_common as C

NAN = float("nan")
REF_A, REF_B = -1, -2                     # pad-free SPAM anchors: first, last


# ------------------------------------------------------------ channel algebra
def sig_eq_from(lam):
    """lam = exp(-sigma^2) over two equal intervals. nan if not invertible."""
    if not np.isfinite(lam) or lam <= 0.0 or lam > 1.0:
        return NAN
    return float(np.sqrt(-np.log(lam)))


def sig_deph_from(lam, lz):
    """lam = exp(-sigma_deph^2)*sqrt(lz). nan if the T1 share exceeds lam."""
    if not (np.isfinite(lam) and np.isfinite(lz)) or lam <= 0.0 or lz <= 0.0:
        return NAN
    x = lam / np.sqrt(lz)
    if x <= 0.0 or x > 1.0:
        return NAN
    return float(np.sqrt(-np.log(x)))


def gamma_from(lz):
    """lz > 1 gives gamma < 0; that is REPORTED, not hidden."""
    if not np.isfinite(lz) or lz < 0.0:
        return NAN
    return float(1.0 - np.sqrt(lz))


def ratio(cur, ref):
    """per-slot ratio, averaged over slots afterwards. cur (n,ns), ref (ns,)."""
    cur = np.asarray(cur, float); ref = np.asarray(ref, float)
    ok = np.isfinite(ref) & (np.abs(ref) > 1e-6)
    if not ok.any():
        return np.full(cur.shape[0], NAN)
    return (cur[:, ok] / ref[ok]).mean(1)


def bare_bloch(ev, key, slots):
    """PER-SLOT (A_perp, Omega, Lambda_z, offset). Never average components
    across qubits before taking the modulus: different detunings make the
    vector sum shrink and re-phase, which fakes a non-monotone decay."""
    A, Om, Lz, Of = [], [], [], []
    for q in slots:
        cx = ev[(key, "bare", "X+", "X", q)][1]
        cy = ev[(key, "bare", "X+", "Y", q)][1]
        dx = ev[(key, "bare", "Y+", "X", q)][1]
        dy = ev[(key, "bare", "Y+", "Y", q)][1]
        co, si = 0.5 * (cx + dy), 0.5 * (cy - dx)
        A.append(float(np.hypot(co, si)))
        Om.append(float(np.arctan2(si, co)))
        zp = ev[(key, "bare", "Z+", "Z", q)][1]
        zm = ev[(key, "bare", "Z-", "Z", q)][1]
        Lz.append(0.5 * (zp - zm)); Of.append(0.5 * (zp + zm))
    return (np.array(A), np.array(Om), np.array(Lz), np.array(Of))


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="fake:fez")
    p.add_argument("--tag", default="dev")
    p.add_argument("--outdir", default=".",
                   help="directory to write hw_deploy_<tag>.npz/.csv into "
                        "(created if missing); keeps repeated runs "
                        "organized for hw_05_aggregate.py")
    p.add_argument("--params-hw", default="")
    p.add_argument("--params-clean", default="")
    p.add_argument("--params-hw-multi", default="",
                   help="hw_params_multipoint_*.npz -- per-tau trained "
                        "thetas for qae_hw, one match per --tau-sweep value")
    p.add_argument("--params-clean-multi", default="",
                   help="hw_params_multipoint_*.npz -- per-tau trained "
                        "thetas for qae_clean, one match per --tau-sweep value")
    p.add_argument("--targets", default="")
    p.add_argument("--variants", default="qae_clean,uncorr,bare",
                   help="comma list; add 'qec' for the real hardware-QEC "
                        "baseline (measure + if_test + Z)")
    p.add_argument("--tau-sweep", default="0,5,10,15,28,40")   # tau/2 per interval
    p.add_argument("--match-qae-idle", type=int, default=1,
                   help="pad every arm to the same wall clock, matched to "
                        "the longer of the QAE and/or std-QEC mid-block")
    p.add_argument("--pad-bare", type=int, default=1,
                   help="1 = pad the bare arm too (time-matched P); "
                        "0 = bare arm unpadded (clean channel, P not matched)")
    p.add_argument("--fb-latency-us", type=float, default=0.5,
                   help="classical feedback round-trip added to the std-QEC "
                        "block duration ESTIMATE (if_test has no duration in "
                        "the ideal gate-timeline model)")
    p.add_argument("--qec-mode", default="parallel",
                   choices=["serial", "parallel"],
                   help="'serial' = ONE ancilla reused sequentially for S1 "
                        "then S2 (original implementation, 4-qubit layout). "
                        "'parallel' [DEFAULT] = TWO ancillas measuring S1 "
                        "and S2 CONCURRENTLY (requires --anc2-qubit, "
                        "5-qubit layout for this arm). The two modes have "
                        "DIFFERENT native mid-block durations -- a "
                        "--qec-dur-us measured for one mode does NOT apply "
                        "to the other.")
    p.add_argument("--qec-dur-us", type=float, default=None,
                   help="override the estimated std-QEC mid-block duration "
                        "with a value you already measured on hardware FOR "
                        "THE CURRENT --qec-mode. Workflow: run once with "
                        "--variants qec (no --yes needed -- the native "
                        "duration is printed and the run aborts before "
                        "submission), read the printed value, then rerun "
                        "the full sweep with this set so the pad is exact, "
                        "not estimated. A value measured for 'serial' does "
                        "NOT apply to 'parallel' and vice versa.")
    p.add_argument("--anc2-qubit", type=int, default=None,
                   help="physical qubit for the SECOND qec ancilla, used "
                        "ONLY when --qec-mode parallel (the default) and "
                        "'qec' is in --variants. --layout's 4th entry is "
                        "anc1 (shared with qae); this is anc2. REQUIRED in "
                        "that case, ignored (with a note) if --qec-mode "
                        "serial. Pick a physical qubit connected to BOTH "
                        "qubits playing the roles of q1 and q2 in --layout "
                        "(layout[1] and layout[2]), and not already used "
                        "in --layout.")
    p.add_argument("--spam-ref", type=int, default=-1,
                   help="pad-free reference block: -1 auto (on when pad>0), 0, 1")
    p.add_argument("--drift-check", type=int, default=1,
                   help="repeat the reference block at the END of the run")
    p.add_argument("--shuffle", type=int, default=1,
                   help="randomise submission order so drift cannot alias into tau")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--bare-slots", type=int, default=3)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--rep-delay", type=float, default=100.0)
    p.add_argument("--rocal", type=int, default=1)
    p.add_argument("--layout", default="")
    p.add_argument("--L", type=int, default=0)
    p.add_argument("--yes", action="store_true")
    return p.parse_args()


def _load_multipoint(path):
    zm = np.load(path, allow_pickle=True)
    taus_m = np.asarray(zm["tau_half"], float)
    n_pts = int(zm["n_points"])
    mp = {float(taus_m[i]): np.asarray(zm[f"P{i}_theta"], float)
          for i in range(n_pts)}
    L = int(zm["L"])
    o = ast.literal_eval(str(zm["opts"]))
    rho_c = float(zm["rho_c"])
    mu_by_tau = {float(taus_m[i]): float(zm[f"P{i}_mu"]) for i in range(n_pts)}
    return mp, L, o, mu_by_tau, rho_c


def main():
    a = cli()
    VAR = [v for v in a.variants.split(",") if v]
    th, th_multi = {}, {}
    o, form, L, meta = dict(C.DEF_OPTS), "selfinv", a.L or C.L_HW, {}

    multi_paths = {"qae_hw": a.params_hw_multi, "qae_clean": a.params_clean_multi}
    single_paths = {"qae_hw": a.params_hw, "qae_clean": a.params_clean}
    single_keys = {"qae_hw": ("theta_best", "theta"), "qae_clean": ("theta",)}

    for tag in ("qae_hw", "qae_clean"):
        if tag not in VAR:
            continue
        mpath, spath = multi_paths[tag], single_paths[tag]
        if mpath:
            mp, L_m, o_m, mu_by_tau, rho_m = _load_multipoint(mpath)
            th_multi[tag] = mp
            L = a.L or L_m
            o, form = o_m, "selfinv"
            meta["mu_range"] = (min(mu_by_tau.values()), max(mu_by_tau.values()))
            meta["rho_c"] = rho_m
            print(f"  {tag}: loaded {len(mp)} multi-point thetas from "
                  f"{mpath}, tau/2 = {sorted(mp)}")
        elif spath:
            z = np.load(spath, allow_pickle=True)
            th[tag] = np.asarray(
                z[next(k for k in single_keys[tag] if k in z.files)], float)
            o = ast.literal_eval(str(z["opts"]))
            form = str(z["form"]); L = a.L or int(z["L"])
            for k in ("mu", "sigma", "rho_c", "gamma"):
                if k in z.files: meta[k] = float(z[k])
        else:
            VAR.remove(tag)

    cfg = C.Cfg(L=L, o=o, form=form, idle_impl="delay", mu=0.0,
               qec_mode=a.qec_mode)
    P = C.n_params(L, o, "hw")
    run = C.Runner(a.backend, rep_delay_us=a.rep_delay)
    try:    gran = int(run.tb.target.granularity) or 16
    except Exception: gran = 16
    print(f"  backend {run.name} ({'REAL' if run.real else 'local'}); "
          f"variants {VAR}; L={L}")
    print(f"  dt = {run.dt:.6g} s, granularity = {gran} dt "
          f"({gran*run.dt*1e9:.2f} ns); mu injected = 0 (nothing is injected)")
    if "sigma" in meta:
        print(f"  warm start trained at sigma={meta.get('sigma')}, "
              f"gamma={meta.get('gamma')}, rho_c={meta.get('rho_c')}, "
              f"mu={meta.get('mu')}")
    elif meta:
        mlo, mhi = meta.get("mu_range", (float("nan"), float("nan")))
        print(f"  multi-point warm starts trained at rho_c={meta.get('rho_c')}, "
              f"mu in [{mlo:+.3f}, {mhi:+.3f}] rad (sigma/gamma/mu all differ "
              f"per tau/2, see per-point thetas above)")
    if "qec" in VAR:
        print(f"  'qec' = REAL mid-circuit syndrome measurement + "
              f"conditional Z correction (dynamic circuit); mode="
              f"'{a.qec_mode}'; requires backend dynamic-circuits support")

    layout = ([int(x) for x in a.layout.split(",")] if a.layout
              else C.pick_layout(run, 15.0, L))
    pm = run.pass_manager(layout if run.kind != "aer" else None)

    # ---- qec ancilla setup: 'serial' reuses the main 4-qubit layout/pm;
    # 'parallel' needs a 5th physical qubit and its OWN pass manager.
    pm_qec = pm
    if "qec" in VAR:
        if a.qec_mode == "parallel":
            if a.anc2_qubit is None:
                raise SystemExit(
                    "--anc2-qubit is required when --qec-mode parallel "
                    "(the default) and 'qec' is in --variants: the std-QEC "
                    "arm measures S1/S2 with TWO ancillas (--layout's 4th "
                    "entry is anc1, --anc2-qubit is anc2). Pick a free "
                    "physical qubit connected to BOTH qubits playing the "
                    "roles of q1 and q2 in --layout (layout[1], layout[2]). "
                    "Or pass --qec-mode serial to use the original "
                    "single-ancilla implementation instead (no --anc2-qubit "
                    "needed).")
            if a.anc2_qubit in layout:
                raise SystemExit(f"--anc2-qubit {a.anc2_qubit} is already "
                                 f"used in --layout {layout}; pick a "
                                 f"different physical qubit")
            layout_qec = layout + [a.anc2_qubit]
            pm_qec = run.pass_manager(layout_qec if run.kind != "aer" else None)
            print(f"  qec ancillas (parallel mode): anc1 (S1) = "
                  f"layout[3]={layout[3]}, anc2 (S2) = {a.anc2_qubit}  "
                  f"(5-qubit layout {layout_qec})")
        else:
            if a.anc2_qubit is not None:
                print(f"  NOTE --anc2-qubit {a.anc2_qubit} given but "
                      f"--qec-mode serial does not use a second ancilla; "
                      f"ignored")
            print(f"  qec ancilla (serial mode): anc = layout[3]="
                  f"{layout[3]} (reused sequentially for S1 then S2, "
                  f"4-qubit layout {layout})")

    tv = C.ParameterVector("t", P)
    fv = None

    sweep = [float(x) for x in a.tau_sweep.split(",")]
    dts = [run.idle_dt(x) for x in sweep]
    xlab = "tau/2 [us]"
    n = len(dts)
    T1, T2, Tphi = run.coherence(layout[:3])            # the bare/code qubits
    T1a, T2a, Tphia = run.coherence(layout)             # all four
    if T1 == T1:
        print(f"  code qubits {layout[:3]}: T1={T1:.1f} T2={T2:.1f} "
              f"T_phi={Tphi:.1f} us   (T2 already contains T1)")
        print(f"  all 4 qubits {layout}: T1={T1a:.1f} T2={T2a:.1f} "
              f"T_phi={Tphia:.1f} us")

    # ---- matched baseline idle, SNAPPED to the timing grid.
    # ALL non-ref arms must occupy the SAME wall clock: uncorr/bare get a
    # plain delay `mid`; qae gets `mid` MINUS its own native mid-block
    # duration, appended as extra_dt after that block; qec gets `mid` MINUS
    # its own native mid-block duration, SPLIT IN HALF and inserted before
    # (extra_qec_pre) and after (extra_qec_post) that block.
    mid = 0
    extra_qae = 0
    extra_qec_pre = extra_qec_post = 0
    dur, raws = {}, {}
    have_qae = any(v.startswith("qae") for v in VAR)
    have_qec = "qec" in VAR
    if a.match_qae_idle and (have_qae or have_qec):
        st = C.axis_states()[0]
        b = pm.run(C.build_circuit(cfg, st, "XXX", "uncorr", dts[-1], 0, tv, fv))
        if have_qae:
            q = pm.run(C.build_circuit(cfg, st, "XXX", "qae", dts[-1], 0, tv, fv))
            raws["qae"] = max(0.0, (run.circuit_seconds(q)
                                    - run.circuit_seconds(b)) / run.dt)
            dur["qae"] = int(round(raws["qae"] / gran)) * gran
        if have_qec:
            if a.qec_dur_us is not None:
                raws["qec"] = a.qec_dur_us * 1e-6 / run.dt
            else:
                z = pm_qec.run(C.build_circuit(cfg, st, "XXX", "qec",
                                               dts[-1], 0, tv, fv))
                raws["qec"] = C.qec_native_seconds(
                    run, z, b, a.fb_latency_us) / run.dt
            dur["qec"] = int(round(raws["qec"] / gran)) * gran
        mid = max(dur.values()) if dur else 0
        extra_qae = max(0, mid - dur.get("qae", 0))
        extra_qec_total = max(0, mid - dur.get("qec", 0))
        k_total = extra_qec_total // gran     # exact: mid & dur already are
        k_pre = k_total // 2                  # multiples of gran
        extra_qec_pre = k_pre * gran
        extra_qec_post = extra_qec_total - extra_qec_pre
    pad_us = mid * run.dt * 1e6
    if mid:
        print(f"\n  matched baseline pad = {mid} dt = {pad_us:.3f} us "
              f"(max native mid-block over {list(dur)})")
        for k in dur:
            if k == "qec":
                extra = extra_qec_pre + extra_qec_post
                detail = (f"  [pre {extra_qec_pre*run.dt*1e6:.3f} us + "
                         f"post {extra_qec_post*run.dt*1e6:.3f} us]")
            else:
                extra = extra_qae
                detail = ""
            tag = "  [--qec-dur-us override]" if (
                k == "qec" and a.qec_dur_us is not None) else ""
            print(f"    {k:>4}: native {dur[k]} dt = "
                  f"{dur[k]*run.dt*1e6:.3f} us (raw {raws[k]:.1f} dt){tag}"
                  f"  -> +{extra*run.dt*1e6:.3f} us appended to match{detail}")
        print(f"  inserted ONCE mid-storage -> total idle per shot = tau + "
              f"{pad_us:.3f} us; equivalent per interval t_eff = tau/2 + "
              f"{pad_us/2:.3f} us")
        print(f"  bare arm padded: {'yes (time-matched P)' if a.pad_bare else 'no'}")
        if have_qec and a.qec_dur_us is None:
            print(f"  NOTE std-QEC duration is an ESTIMATE for --qec-mode "
                  f"'{a.qec_mode}': if_test has no duration in the ideal "
                  f"gate-timeline model, so a fixed feedback-latency guess "
                  f"(--fb-latency-us {a.fb_latency_us:g}) was added on top "
                  f"of the real reset/H/CX/measure timings. Probe once "
                  f"with --variants qec, read the printed native duration, "
                  f"and rerun the full sweep with --qec-dur-us to pin the "
                  f"pad down exactly. This estimate does NOT carry over to "
                  f"the other --qec-mode.")

    use_ref = ("bare" in VAR) and (a.spam_ref == 1 or
                                   (a.spam_ref < 0 and mid > 0 and a.pad_bare))
    if use_ref:
        print("  pad-free SPAM reference block ON -> absolute sigma/gamma "
              "without any T_phi model")

    # ---- circuits
    states = C.axis_states()
    slots = list(C.CODE[:max(1, min(3, a.bare_slots))])
    ns = len(slots)

    def bare_tasks(key, idle, pad):
        out = []
        for s in states:
            axes = ("X", "Y") if s[3] in ("X", "Y") else (s[3],)
            for ax in axes:
                for sl in slots:
                    qc = C.build_circuit(cfg, s, ax, "bare", idle, pad,
                                         tv, fv, slot=sl)
                    out.append(((key, "bare", s[0], ax, sl), pm.run(qc), 1, sl))
        return out

    tasks = []
    if use_ref:
        tasks += bare_tasks(REF_A, 0, 0)
    for i, d in enumerate(dts):
        for v in VAR:
            base = "qae" if v.startswith("qae") else v
            if base == "bare":
                tasks += bare_tasks(i, d, mid if a.pad_bare else 0)
            elif base == "qec":
                for s in states:
                    for g in C.AX_SET[s[3]]:
                        qc = C.build_circuit(cfg, s, g, "qec", d, 0, tv, fv,
                                             extra_dt=extra_qec_post,
                                             extra_dt_pre=extra_qec_pre)
                        tasks.append(((i, v, s[0], g, -1), pm_qec.run(qc), 3, -1))
            else:
                for s in states:
                    for g in C.AX_SET[s[3]]:
                        ed = extra_qae if base == "qae" else 0
                        qc = C.build_circuit(cfg, s, g, base, d, mid, tv, fv,
                                             extra_dt=ed)
                        tasks.append(((i, v, s[0], g, -1), pm.run(qc), 3, -1))
    if use_ref and a.drift_check:
        tasks += bare_tasks(REF_B, 0, 0)

    if a.shuffle:                       # anchors stay first/last, rest randomised
        head = [t for t in tasks if t[0][0] == REF_A]
        tail = [t for t in tasks if t[0][0] == REF_B]
        body = [t for t in tasks if t[0][0] not in (REF_A, REF_B)]
        idx = np.random.default_rng(a.seed).permutation(len(body))
        tasks = head + [body[k] for k in idx] + tail

    est = run.estimate([t[1] for t in tasks], [1] * len(tasks), a.shots)
    print(f"\n  {len(tasks)} circuits, estimated {est/60:.1f} QPU min")
    if run.real and not a.yes:
        raise SystemExit("add --yes to submit to hardware")
    run.open()

    eps = np.zeros((C.NQ, 2))
    if a.rocal and run.kind != "aer":
        r = run.run([(pm.run(q),) for q in C.ro_circuits()], 4096, "rocal")
        eps = C.ro_eps(
            C.pub_probs(r[0].data.c, C.NQ).reshape(-1, 1 << C.NQ).mean(0),
            C.pub_probs(r[1].data.c, C.NQ).reshape(-1, 1 << C.NQ).mean(0))
        print("  readout errors:\n", np.round(eps, 4))

    def theta_for(tag, tau_val):
        if tag in th_multi:
            mp = th_multi[tag]
            best_t = min(mp, key=lambda k: abs(k - tau_val))
            if abs(best_t - tau_val) > 1e-6:
                print(f"    NOTE {tag}: no exact multi-point match for "
                      f"tau/2={tau_val}; using nearest trained point "
                      f"tau/2={best_t}")
            return mp[best_t]
        return th.get(tag, np.zeros(P))

    def tau_val_of(k):
        i = k[0]
        if isinstance(i, int) and 0 <= i < n:
            return sweep[i]
        return None

    pubs = []
    for k, isa, nb, sl in tasks:
        tag = k[1]
        tv_val = tau_val_of(k)
        if tag in th_multi and tv_val is not None:
            theta_row = theta_for(tag, tv_val)
        else:
            theta_row = th.get(tag, np.zeros(P))
        pubs.append((isa, C.param_array(isa, np.asarray(theta_row)[None, :],
                                        None, tv, fv)))
    res = run.run(pubs, a.shots, "deploy")
    run.close()

    ev = {}
    for (k, isa, nb, sl), r in zip(tasks, res):
        pr = C.pub_probs(r.data.c, nb).reshape(-1, 1 << nb).mean(0)
        ev[k] = C.parities(C.ro_unfold(pr[None, :],
                                       C.CODE if nb == 3 else (sl,), eps), nb)[0]

    # ---- P per arm
    Pm = {v: np.zeros(n) for v in VAR}
    Ps_all = {v: np.zeros((n, 6)) for v in VAR}
    spec = {v: np.zeros((n, 5)) for v in VAR if v != "bare"}
    for i in range(n):
        for v in VAR:
            for j, s in enumerate(states):
                if v == "bare":
                    dv = np.mean([ev[(i, v, s[0], s[3], q)][1] for q in slots])
                else:
                    dv = C.decode({g: ev[(i, v, s[0], g, -1)]
                                   for g in C.AX_SET[s[3]]})[s[3]]
                Ps_all[v][i, j] = 0.5 * (1 + s[4] * dv)
            Pm[v][i] = Ps_all[v][i].mean()
            if v != "bare":
                dd = C.decode({"XXX": ev[(i, v, "Z+", "XXX", -1)],
                               "ZZZ": ev[(i, v, "X+", "ZZZ", -1)],
                               "YYY": ev[(i, v, "Y+", "YYY", -1)]})
                spec[v][i] = [dd["S1"], dd["S2"], dd["XXX"], dd["YYY"], dd["ZZZ"]]

    P_ref = NAN
    if use_ref:
        P_ref = float(np.mean([0.5 * (1 + s[4] * np.mean(
            [ev[(REF_A, "bare", s[0], s[3], q)][1] for q in slots]))
            for s in states]))

    # ---- channel extraction (bare arm, PER SLOT then averaged)
    have = "bare" in VAR
    Aq = np.full((n, ns), NAN); Omq = np.full((n, ns), NAN)
    Lzq = np.full((n, ns), NAN); Ofq = np.full((n, ns), NAN)
    Aref = Lref = Oref = Ofref = None
    lam = lz = np.full(n, NAN); offc = np.full(n, NAN); Om = np.full(n, NAN)
    basis, drift = "none", NAN
    if have:
        for i in range(n):
            Aq[i], Omq[i], Lzq[i], Ofq[i] = bare_bloch(ev, i, slots)
        if use_ref:
            Aref, Oref, Lref, Ofref = bare_bloch(ev, REF_A, slots)
            Omq = Omq - Oref[None, :]
        if n > 1:
            Omq = np.unwrap(Omq, axis=0)
        Om = Omq.mean(1)
        if use_ref:
            lam, lz = ratio(Aq, Aref), ratio(Lzq, Lref)
            offc, basis = Ofq.mean(1) - Ofref.mean(), "absolute"
            if a.drift_check:
                Ad, _, Ld, _ = bare_bloch(ev, REF_B, slots)
                drift = float(np.mean(Ad / np.where(np.abs(Aref) > 1e-6,
                                                    Aref, np.nan)))
        elif sweep[0] == 0.0:
            lam, lz = ratio(Aq, Aq[0]), ratio(Lzq, Lzq[0])
            offc = Ofq.mean(1) - Ofq[0].mean()
            basis = "absolute" if mid == 0 or not a.pad_bare else "increment"
        else:
            lam, lz = Aq.mean(1), Lzq.mean(1)
            offc, basis = Ofq.mean(1), "raw"

    mu_hw = 0.5 * Om
    sig_eq = np.array([sig_eq_from(x) for x in lam])
    sig_d = np.array([sig_deph_from(x, y) for x, y in zip(lam, lz)])
    gam = np.array([gamma_from(y) for y in lz])
    t_eff = np.asarray(sweep, float) + (0.5 * pad_us if a.pad_bare else 0.0)
    tmod = t_eff if basis == "absolute" else np.asarray(sweep, float)
    s_mod = np.array([C.noise_at_tau(t, T1, Tphi)[0] for t in tmod])
    g_mod = np.array([C.noise_at_tau(t, T1, Tphi)[1] for t in tmod])

    # ---- print
    print(f"\n  channel basis: {basis.upper()}"
          + ("   (pad divides out of every ratio)" if basis == "increment" else ""))
    if use_ref:
        print(f"  P_bare at zero storage (SPAM only) = {P_ref:.4f}")
        if np.isfinite(drift):
            flag = "OK" if abs(drift - 1) < 0.03 else "DRIFT / NOT REPRODUCIBLE"
            print(f"  drift check: A_perp(end)/A_perp(start) = {drift:.4f}  [{flag}]")
    hdr = (f"\n  {xlab:>10} {'t_eff':>8} " + " ".join(f"{v:>10}" for v in VAR))
    if have:
        hdr += ("   lam_perp  sigma_eq sigma_deph  gamma/int   offset   mu_hw"
                "  sig_mod  gam_mod")
    print(hdr)
    for i in range(n):
        line = (f"  {sweep[i]:10.2f} {t_eff[i]:8.2f} "
                + " ".join(f"{Pm[v][i]:10.4f}" for v in VAR))
        if have:
            line += (f"   {lam[i]:8.4f}  {sig_eq[i]:8.3f}  {sig_d[i]:9.3f}"
                     f"  {gam[i]:9.4f}  {offc[i]:+7.4f}  {mu_hw[i]:+6.3f}"
                     f"  {s_mod[i]:7.3f}  {g_mod[i]:7.4f}")
        print(line)

    if have:
        print("\n  sigma_eq   = TOTAL transverse decay as an equivalent "
              "per-interval sigma (CONTAINS T1)")
        print("  sigma_deph = the part the phase-flip code can actually correct")
        print("  gamma/int  = amplitude damping per interval, from the "
              "LONGITUDINAL decay")
        print("  sig_mod/gam_mod = calibration prediction at t_eff "
              "(sqrt(2t/T_phi), 1-exp(-t/T1))")
        print("  nan = the inversion has NO physical solution at that point "
              "(never silently 0)")
        bad = np.isfinite(lam) & (lam > 1.0)
        if bad.any():
            print(f"\n  WARNING lam_perp > 1 at tau/2 = "
                  f"{[sweep[i] for i in np.where(bad)[0]]}: the transverse "
                  "signal GREW.\n  That is impossible for a decay -> the "
                  "reference point or the run itself drifted.\n  Re-run with "
                  "--spam-ref 1 --drift-check 1 --shuffle 1 before using these "
                  "numbers.")
        if np.isnan(sig_d).any() and not np.isnan(lam).all():
            print("  NOTE some sigma_deph are nan: there the measured "
                  "transverse decay is WEAKER\n  than amplitude damping alone "
                  "implies, so no dephasing share can be extracted.")
        if np.nanmax(gam) > 0.05:
            print("\n  NOTE gamma > 0.05: X_j commutes with both stabilizers, "
                  "so every T1\n  jump is a SILENT logical error and 3 code "
                  "qubits give 3x the rate.\n  Expect P_uncorr < P_bare. This "
                  "is predicted, not a training failure.")
        if np.nanmin(gam) < -0.01:
            print("  WARNING gamma < 0: the longitudinal reference is wrong "
                  "(drift or bad SPAM ref).")
        if np.nanmax(np.abs(mu_hw)) > 0.1:
            print("  WARNING |mu_hw| > 0.1 rad: real coherent bias. Report it; "
                  "do NOT inject mu.")
        spread = np.nanmean(Omq.std(1)) if n > 1 else NAN
        if np.isfinite(spread):
            print(f"  per-qubit phase spread (std over {ns} slots, mean over "
                  f"tau) = {spread:.3f} rad")
        Tt = 2.0 * tmod                       # total idle time actually seen
        m = np.isfinite(lam) & (lam > 0.05) & (lam <= 1.0) & (Tt > 0)
        if m.sum() >= 2:
            e = np.polyfit(Tt[m], np.log(lam[m]), 1)
            g = np.polyfit(Tt[m] ** 2, np.log(lam[m]), 1)
            print(f"\n  T2*_fit (exponential) = {-1/e[0]:.1f} us   |   "
                  f"T2*_fit (gaussian) = {np.sqrt(-1/g[0]):.1f} us   "
                  f"(fit vs TOTAL idle = tau + pad)")
        mz = np.isfinite(lz) & (lz > 0.05) & (Tt > 0)
        if mz.sum() >= 2:
            e1 = np.polyfit(Tt[mz], np.log(lz[mz]), 1)
            print(f"  T1_fit (longitudinal) = {-1/e1[0]:.1f} us   compare "
                  f"{T1:.1f} us from calibration")

    # ---- targets overlay, matched at the EFFECTIVE time
    if a.targets:
        t = np.load(a.targets, allow_pickle=True)
        tg = np.asarray(t["tau_half"], float)
        print(f"\n  theory matched at t_eff (pad included: "
              f"{'yes' if a.pad_bare and mid else 'no'})")
        print(f"  {'tau/2':>7} {'t_eff':>7} {'t_thy':>7} | {'P_bare':>9} "
              f"{'thy':>9} | {'P_unc':>9} {'thy':>9} | {'P_qae':>9} {'thy':>9}")
        vq = ("qae_hw" if "qae_hw" in VAR
              else ("qae_clean" if "qae_clean" in VAR else None))
        for i in range(n):
            j = int(np.argmin(np.abs(tg - tmod[i])))
            print(f"  {sweep[i]:7.1f} {t_eff[i]:7.2f} {tg[j]:7.2f} | "
                  f"{Pm.get('bare', [NAN]*n)[i]:9.4f} "
                  f"{float(t['P_bare'][j]):9.4f} | "
                  f"{Pm.get('uncorr', [NAN]*n)[i]:9.4f} "
                  f"{float(t['P_unc'][j]):9.4f} | "
                  f"{(Pm[vq][i] if vq else NAN):9.4f} "
                  f"{float(t['P_ideal'][j]):9.4f}")

    # ---- save
    os.makedirs(a.outdir, exist_ok=True)
    out_npz = os.path.join(a.outdir, f"hw_deploy_{a.tag}.npz")
    out_csv = os.path.join(a.outdir, f"hw_deploy_{a.tag}.csv")
    C.save(out_npz, sweep=np.array(sweep), t_eff=t_eff,
           variants=np.array(VAR), lam=lam, Aperp=Aq.mean(1), Aperp_slot=Aq,
           Om=Om, Om_slot=Omq, mu_hw=mu_hw, sigma_eq=sig_eq, sigma_deph=sig_d,
           gamma_hw=gam, lz=lz, Lz=Lzq.mean(1), Lz_slot=Lzq, offset=offc,
           sigma_model=s_mod, gamma_model=g_mod, sigma_basis=basis,
           P_bare_spam=P_ref, drift=drift, mid_dt=mid, pad_us=pad_us,
           dt=run.dt, granularity=gran, pad_bare=int(a.pad_bare),
           pad_placement="qae_after__qec_split_pre_post",
           qec_mode=a.qec_mode,
           anc2_qubit=(a.anc2_qubit if a.anc2_qubit is not None else -1),
           slots=np.array(slots),
           spec_labels=np.array(["S1", "S2", "XXX", "YYY", "ZZZ"]),
           layout=layout, L=L, eps=eps, xlabel=xlab,
           T1=T1, T2=T2, Tphi=Tphi, T1_all=T1a, T2_all=T2a, Tphi_all=Tphia,
           dur_qae_dt=dur.get("qae", 0), dur_qec_dt=dur.get("qec", 0),
           extra_qae_dt=extra_qae,
           extra_qec_dt=extra_qec_pre + extra_qec_post,
           extra_qec_pre_dt=extra_qec_pre,
           extra_qec_post_dt=extra_qec_post,
           qec_dur_us_cli=(a.qec_dur_us if a.qec_dur_us is not None else np.nan),
           fb_latency_us=a.fb_latency_us, multi_theta=int(bool(th_multi)),
           cfg=str(vars(a)),
           **{f"P_{v}": Pm[v] for v in VAR},
           **{f"Pstates_{v}": Ps_all[v] for v in VAR},
           **{f"spec_{v}": spec[v] for v in spec})
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([xlab, "t_eff_us", "lam_perp", "sigma_eq", "sigma_deph",
                    "gamma_hw", "offset", "Om_rad", "mu_hw", "sigma_model",
                    "gamma_model"] + [f"P_{v}" for v in VAR]
                   + ["gain_vs_uncorr", "gain_vs_bare", "r_vs_uncorr", "C_p2q"])
        vref = ("qae_hw" if "qae_hw" in VAR else
                ("qae_clean" if "qae_clean" in VAR else VAR[0]))
        for i in range(n):
            g1 = Pm[vref][i] - Pm["uncorr"][i] if "uncorr" in VAR else NAN
            g2b = Pm[vref][i] - Pm["bare"][i] if "bare" in VAR else NAN
            if "uncorr" in VAR and Pm["uncorr"][i] > 0.5 and Pm[vref][i] > 0.5:
                rr = (Pm[vref][i] - 0.5) / (Pm["uncorr"][i] - 0.5)
                cp = -np.log(rr)
            else:
                rr = cp = NAN
            w.writerow([sweep[i], f"{t_eff[i]:.4f}", f"{lam[i]:.5f}",
                        f"{sig_eq[i]:.4f}", f"{sig_d[i]:.4f}", f"{gam[i]:.5f}",
                        f"{offc[i]:.5f}", f"{Om[i]:.5f}", f"{mu_hw[i]:.5f}",
                        f"{s_mod[i]:.4f}", f"{g_mod[i]:.5f}"]
                       + [f"{Pm[v][i]:.5f}" for v in VAR]
                       + [f"{g1:+.5f}", f"{g2b:+.5f}", f"{rr:.5f}", f"{cp:.5f}"])

    # ---- r and the internal-noise measurement
    if "uncorr" in VAR and any(v.startswith("qae") for v in VAR):
        v = "qae_hw" if "qae_hw" in VAR else "qae_clean"
        print(f"\n  {xlab:>10} {'P_qae':>9} {'P_unc':>9} {'r':>8} {'C*p2q':>8}")
        rs = []
        for i in range(n):
            if Pm["uncorr"][i] <= 0.5 or Pm[v][i] <= 0.5:
                print(f"  {sweep[i]:10.2f} {Pm[v][i]:9.4f} "
                      f"{Pm['uncorr'][i]:9.4f} {'--':>8} {'--':>8}   "
                      "(a P is at/below 0.5: Eq. 28 undefined)")
                continue
            rr = (Pm[v][i] - 0.5) / (Pm["uncorr"][i] - 0.5)
            rs.append(rr)
            print(f"  {sweep[i]:10.2f} {Pm[v][i]:9.4f} {Pm['uncorr'][i]:9.4f} "
                  f"{rr:8.4f} {-np.log(rr):8.4f}")
        if len(rs) > 1:
            cp = -np.log(np.asarray(rs))
            ok = cp.std() < 0.03
            print(f"  C*p2q = {cp.mean():.4f} +- {cp.std():.4f}   "
                  + ("tau-INDEPENDENT" if ok else
                     "tau-DEPENDENT -> drift, leakage, or a bad reference"))
            print("  divide by C_count (hw_00_dryrun --stage estimate) and the "
                  "device p2q to get eta")
        elif len(rs) <= 1:
            print("  too few usable points for C*p2q")
    if "qec" in VAR and "uncorr" in VAR:
        print(f"\n  {xlab:>10} {'P_qec':>9} {'P_unc':>9} {'gain':>8}")
        for i in range(n):
            print(f"  {sweep[i]:10.2f} {Pm['qec'][i]:9.4f} "
                  f"{Pm['uncorr'][i]:9.4f} {Pm['qec'][i]-Pm['uncorr'][i]:+8.4f}")
    print(f"\n  QPU used ~{run.usage:.0f} s in {run.jobs} jobs")
    print(f"  wrote {out_npz}")
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    main()