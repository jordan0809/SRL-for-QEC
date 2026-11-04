"""hw_00_dryrun.py 

Stages
  scan        validate ansatz conversion + channel + convention against the
              trainer npz (SINGLE operating point), write a hardware
              warm-start file
  scan-multi  same validation, but for a FAMILY of operating points from
              train_qae_multipoint.py, one per tau/2. Finds the wiring/
              reset/phase convention ONCE (using the highest-sigma point,
              most discriminating), then applies and validates it for
              EVERY point, writing one combined locked-parameter file.
  targets     build hw_targets_combined.npz: P_bare / P_unc / P_ideal and the
              break-even p2q for a list of device idle times.

  smoke       end-to-end plumbing test on a LOCAL simulator (needs qiskit-aer)
  estimate    pad, dt, gate counts, C_count, circuit count and QPU budget
"""
import argparse, ast, itertools, math
import numpy as np
import qae_hw_common as C


# --------------------------------------------------------------------- cli
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["scan", "scan-multi", "targets", "smoke", "estimate"])
    # scan
    p.add_argument("--npz", default="dephasing_results_layerwise_combined.npz")
    p.add_argument("--L", type=int, default=C.L_HW)
    p.add_argument("--out", default="hw_params_clean_combined_L2.npz")
    # scan-multi
    p.add_argument("--npz-multi", default="dephasing_multipoint_L2.npz")
    p.add_argument("--out-multi", default="hw_params_multipoint_L2.npz")
    # targets / smoke / estimate
    p.add_argument("--params", default="hw_params_clean_combined_L2.npz")
    p.add_argument("--backend", default="fake:fez")
    p.add_argument("--t1", type=float, default=0.0)      # us; 0 = read backend
    p.add_argument("--t2", type=float, default=0.0)      # us
    p.add_argument("--tphi", type=float, default=0.0)    # us; overrides t1/t2
    p.add_argument("--tau", default="0,5,10,15,20,28,40")
    p.add_argument("--pad-us", type=float, default=0.0,
                   help="matched baseline idle per shot [us]; the baselines "
                        "are evaluated at tau/2 + pad/2")
    p.add_argument("--mu", type=float, default=None,
                   help="coherent bias per interval; default = trained value. "
                        "Hardware injects nothing, so use 0 or --mu-per-us")
    p.add_argument("--mu-per-us", type=float, default=None,
                   help="device bias per us of interval time; mu = k * t")
    p.add_argument("--C-fit", type=float, default=107.0)
    p.add_argument("--eta", type=float, default=0.25)
    p.add_argument("--targets-out", default="hw_targets_combined.npz")
    p.add_argument("--targets", default="")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--n-angle", type=int, default=256)
    p.add_argument("--idle-us", type=float, default=15.0)
    p.add_argument("--rep-delay", type=float, default=100.0)
    p.add_argument("--layout", default="")
    p.add_argument("--variants", default="qae_clean,uncorr,bare")
    p.add_argument("--bare-slots", type=int, default=3)
    return p.parse_args()


def _meta(z, key, default):
    for k in (key, "meta_" + key):
        if k in z.files:
            return float(np.asarray(z[k]).ravel()[0])
    return default


# ------------------------------------------------------------------ unit tests
def t1_abc(n=200, seed=0):
    from qiskit.circuit.library import UnitaryGate
    rng = np.random.default_rng(seed); worst = 0.0
    for _ in range(n):
        a = rng.normal(size=3) * 2.0
        U = C._rz(a[2]) @ C._ry(a[1]) @ C._rx(a[0])
        qc = C.QuantumCircuit(2); C._cu(qc, 0, 1, C.abc_params(U))
        qr = C.QuantumCircuit(2); qr.append(UnitaryGate(U).control(1), [0, 1])
        worst = max(worst, float(np.abs(C.Operator(qc).data
                                        - C.Operator(qr).data).max()))
    return worst


def t2_decoder():
    """all single-Z errors on the 6 axis states must decode back exactly."""
    worst = 0.0
    Zs = [C.pauli_op(l) for l in ("ZII", "IZI", "IIZ")]
    for _, th, ph, ax, _ in C.axis_states():
        r0 = C.code_state(th, ph, 1)
        want = C.exact_spectrum(r0)
        for Z in [np.eye(8, dtype=complex)] + Zs:
            r = Z @ r0 @ Z.conj().T
            ev = {}
            for g in C.ALL_SET:
                e = np.empty(8)
                for m in range(8):
                    lab = "".join(g[i] if (m >> i) & 1 else "I" for i in range(3))
                    e[m] = float(np.real(np.trace(C.pauli_op(lab) @ r)))
                ev[g] = e
            got = C.decode(ev)
            for k in ("Z", "X", "Y"):
                worst = max(worst, abs(got[k] - want[k]))
    return worst


def t3_ansatz(Ls=(1, 2), seed=1):
    rng = np.random.default_rng(seed); worst = 0.0
    for L in Ls:
        sim = rng.normal(size=C.n_params(L, C.DEF_OPTS, "sim")) * 1.3
        worst = max(worst, C.check_conversion(sim, L, C.DEF_OPTS))
    return worst


def _run_self_tests():
    e1 = t1_abc()
    print(f"[t1] ABC controlled-SU(2): max|dU| = {e1:.3e}",
          "OK" if e1 < 1e-10 else "*** FAIL ***")
    e2 = t2_decoder()
    print(f"[t2] majority-vote decoder, all single-Z errors: max|d| = {e2:.3e}",
          "OK" if e2 < 1e-10 else "*** FAIL ***")
    e3 = t3_ansatz()
    print(f"[t3] sim ansatz == ABC ansatz (L=1,2): max|dU| = {e3:.3e}",
          "OK" if e3 < 1e-10 else "*** FAIL ***")
    if max(e1, e2, e3) > 1e-10:
        raise SystemExit("  library self-test failed; stop.")


# ---------------------------------------------------------------- stage scan
def stage_scan(a):
    _run_self_tests()

    z = np.load(a.npz, allow_pickle=True)
    mu    = _meta(z, "mu", 0.0)
    sigma = _meta(z, "sigma", 0.0)
    rhoc  = _meta(z, "rho_corr", 0.0)
    gamma = _meta(z, "gamma", 0.0)
    layers = sorted(int(k[1:k.index("_")]) for k in z.files
                    if k.startswith("L") and k.endswith("_params"))
    L = a.L if a.L in layers else layers[0]
    th_t = np.asarray(z["meta_test_thetas"], float)
    ph_t = np.asarray(z["meta_test_phis"], float)
    print(f"\n  npz: mu={mu}  sigma={sigma}  rho_c={rhoc}  gamma={gamma}"
          f"  |  {th_t.size} test states  |  layers {layers}  |  using L={L}")
    if gamma > 0:
        print("  channel = AD(gamma) o Dephase(mu,sigma,rho_c)  [combined]")

    # ---- [t4] baselines of the memory protocol
    Pb_f = C.P_bare_theory(mu, sigma, gamma)
    Pb_e = C.exact_run("bare", mu, sigma, rhoc, gamma=gamma)[0]
    Pu_e = C.P_unc_exact(mu, sigma, rhoc, gamma)
    Pu_f = C.P_unc_theory(mu, sigma, rhoc, gamma)
    print(f"\n[t4] memory baselines at mu={mu}, sigma={sigma}, rho_c={rhoc}, "
          f"gamma={gamma}")
    print(f"     P_bare = {Pb_e:.6f}  (closed form {Pb_f:.6f}, "
          f"d={abs(Pb_e-Pb_f):.2e})")
    print(f"     P_unc  = {Pu_e:.6f}  (exact, numeric)"
          + (f"   closed form(gamma=0) {Pu_f:.6f}" if gamma == 0 else
             "   [dephasing-only closed form not applicable]"))
    if abs(Pb_e - Pb_f) > 1e-9:
        raise SystemExit("  P_bare closed form disagrees with the exact run; stop.")
    if gamma == 0 and abs(Pu_e - Pu_f) > 1e-8:
        raise SystemExit("  P_unc closed form disagrees with the exact run; stop.")

    # ---- [A] channel check
    print("\n[A] noise channel vs 'f_noisy_test' (validates the mask, mu, "
          "sigma, rho_c, gamma and the state parameterisation)")
    stored = np.asarray(z["f_noisy_test"], float)
    got = C.fid_noisy_batch(th_t, ph_t, mu, sigma, rhoc, gamma)
    rmsA = float(np.sqrt(np.mean((got - stored) ** 2)))
    okA = rmsA < 1e-6
    print(f"    RMS = {rmsA:.3e}   mean {got.mean():.8f} vs stored "
          f"{stored.mean():.8f}   {'OK' if okA else '*** MISMATCH ***'}")
    if not okA:
        print("    -> the noise channel does not match. Stop and report this "
              "line.\n    (a large POSITIVE mean difference usually means "
              "gamma is missing somewhere)")
        raise SystemExit(1)

    # ---- [B] convention scan
    key = f"L{L}_f_qae_test"
    tgt = np.asarray(z[key], float)
    sim = np.asarray(z[f"L{L}_params"], float).ravel()
    print(f"\n[B] QAE convention scan against '{key}' "
          f"(1 noise interval + QAE, fidelity with the clean state)")
    print(f"    L{L}_params: {sim.size} params (this layout needs "
          f"{C.n_params(L, C.DEF_OPTS, 'sim')})")
    print(f"    target mean fidelity = {tgt.mean():.8f}\n")
    rows = []
    for anc, reset, phase in itertools.product((3, 2, 1, 0), ("code", "anc"),
                                               (1, -1)):
        o = dict(anc=anc, reset=reset, phase=phase)
        try:
            hw = C.convert_sim_params(sim, L, o)
            U = C.Operator(C.build_u(hw, L, o, "hw")).data
            S = C.map_superop(U, o, "selfinv")
            f = C.fid_qae_batch(th_t, ph_t, mu, sigma, rhoc, S, phase, gamma)
            rows.append((float(np.sqrt(np.mean((f - tgt) ** 2))),
                         float(f.mean()), o))
        except Exception:
            rows.append((9.9, float("nan"), o))
    rows.sort(key=lambda r: r[0])
    print("    rank   RMS vs stored     mean F  form      ancilla-wire  reset   "
          "phase")
    for i, (r, m, o) in enumerate(rows[:8]):
        print(f"    {i:>4}   {r:>13.3e}  {m:>9.6f}  selfinv   wire {o['anc']}  "
              f"      {o['reset']:<5}  {o['phase']:+d}")
    best_rms, best_mean, best_o = rows[0]
    if best_rms > 1e-5:
        print("\n    *** no candidate matches. Do NOT continue. Paste this table.")
        raise SystemExit(1)
    print(f"\n    -> convention {best_o}, RMS {best_rms:.3e}  OK")

    # ---- [C] every depth with the SAME convention
    print("\n[C] same convention at every available depth:")
    allok = True
    for nl in layers:
        s = np.asarray(z[f"L{nl}_params"], float).ravel()
        t = np.asarray(z[f"L{nl}_f_qae_test"], float)
        hw = C.convert_sim_params(s, nl, best_o)
        U = C.Operator(C.build_u(hw, nl, best_o, "hw")).data
        S = C.map_superop(U, best_o, "selfinv")
        f = C.fid_qae_batch(th_t, ph_t, mu, sigma, rhoc, S, best_o["phase"], gamma)
        r = float(np.sqrt(np.mean((f - t) ** 2)))
        ok = r < 1e-5; allok &= ok
        print(f"    L={nl}: RMS {r:.3e}   mean {f.mean():.6f} vs {t.mean():.6f}"
              f"  {'OK' if ok else 'FAIL'}")
    if not allok:
        raise SystemExit("  [C] failed: the deployed circuit is not the trained "
                         "one. STOP.")

    # ---- write the warm start
    hw = C.convert_sim_params(sim, L, best_o)
    conv = C.check_conversion(sim, L, best_o)
    Pm, Fc, sp = C.exact_run("qae", mu, sigma, rhoc, theta=hw, L=L, o=best_o,
                             form="selfinv", gamma=gamma)
    print(f"\n  writing: L={L}, form=selfinv, opts={best_o}")
    print(f"  sim->hw conversion ||dU|| = {conv:.2e}   "
          f"({sim.size} logical -> {hw.size} physical angles)")
    print(f"  6-axis quantum-memory P = {Pm:.6f}   F_corr = {Fc:.6f}   "
          f"<S> = {0.5*(sp['S1']+sp['S2']):.4f}")
    print(f"  reference: P_bare = {Pb_e:.6f}   P_unc = {Pu_e:.6f}   "
          f"benefit vs unc = {Pm-Pu_e:+.6f}")
    C.save(a.out, theta=hw, sim=sim, L=L, form="selfinv", opts=best_o,
           rms=best_rms, rms_A=rmsA, conv=conv,
           mu=mu, sigma=sigma, rho_c=rhoc, gamma=gamma,
           P_mem=Pm, F_corr=Fc, P_bare=Pb_e, P_unc=Pu_e, src=a.npz)


# ----------------------------------------------------------- stage scan-multi
def stage_scan_multi(a):
    _run_self_tests()

    z = np.load(a.npz_multi, allow_pickle=True)
    taus = np.asarray(z["tau_half"], float)
    L = a.L if a.L else int(z["L"])
    rhoc = float(z["rho_corr"])
    src = str(z["noise_source"]) if "noise_source" in z.files else "unknown"
    th_t = np.asarray(z["meta_test_thetas"], float)
    ph_t = np.asarray(z["meta_test_phis"], float)
    n_pts = len(taus)
    print(f"\n  {n_pts} operating points, L={L}, rho_c={rhoc}")
    print(f"  noise source: {src}")
    for i in range(n_pts):
        print(f"    P{i}: tau/2={float(z[f'P{i}_tau_half']):6.2f} us  "
              f"mu={float(z[f'P{i}_mu']):+.4f}  "
              f"sigma={float(z[f'P{i}_sigma']):.4f}  "
              f"gamma={float(z[f'P{i}_gamma']):.4f}")

    j = int(np.argmax([float(z[f"P{i}_sigma"]) for i in range(n_pts)]))
    mu_j = float(z[f"P{j}_mu"])
    sigma_j, gamma_j = float(z[f"P{j}_sigma"]), float(z[f"P{j}_gamma"])
    sim_j = np.asarray(z[f"P{j}_params"], float).ravel()
    tgt_j = np.asarray(z[f"P{j}_f_qae_test"], float)
    print(f"\n[B] convention scan using P{j} (tau/2={taus[j]}, mu={mu_j:+.4f}, "
          f"sigma={sigma_j:.4f}, gamma={gamma_j:.4f})")
    rows = []
    for anc, reset, phase in itertools.product((3, 2, 1, 0), ("code", "anc"),
                                               (1, -1)):
        o = dict(anc=anc, reset=reset, phase=phase)
        try:
            hw = C.convert_sim_params(sim_j, L, o)
            U = C.Operator(C.build_u(hw, L, o, "hw")).data
            S = C.map_superop(U, o, "selfinv")
            f = C.fid_qae_batch(th_t, ph_t, mu_j, sigma_j, rhoc, S, phase, gamma_j)
            rows.append((float(np.sqrt(np.mean((f - tgt_j) ** 2))),
                         float(f.mean()), o))
        except Exception:
            rows.append((9.9, float("nan"), o))
    rows.sort(key=lambda r: r[0])
    best_rms, best_mean, best_o = rows[0]
    print(f"    best: {best_o}  RMS={best_rms:.3e}  mean F={best_mean:.6f}")
    if best_rms > 1e-5:
        print("\n    *** no candidate matches. Do NOT continue.")
        raise SystemExit(1)

    print(f"\n[C] locking all {n_pts} points with convention {best_o}:")
    out = dict(tau_half=taus, rho_c=rhoc, L=L, form="selfinv",
               opts=str(best_o), n_points=n_pts, noise_source=src)
    allok = True
    for i in range(n_pts):
        mu_i = float(z[f"P{i}_mu"])
        sig_i, gam_i = float(z[f"P{i}_sigma"]), float(z[f"P{i}_gamma"])
        sim_i = np.asarray(z[f"P{i}_params"], float).ravel()
        tgt_i = np.asarray(z[f"P{i}_f_qae_test"], float)
        hw_i = C.convert_sim_params(sim_i, L, best_o)
        U = C.Operator(C.build_u(hw_i, L, best_o, "hw")).data
        S = C.map_superop(U, best_o, "selfinv")
        f = C.fid_qae_batch(th_t, ph_t, mu_i, sig_i, rhoc, S, best_o["phase"], gam_i)
        r = float(np.sqrt(np.mean((f - tgt_i) ** 2)))
        ok = r < 1e-5; allok &= ok
        Pm, Fc, sp = C.exact_run("qae", mu_i, sig_i, rhoc, theta=hw_i, L=L,
                                 o=best_o, form="selfinv", gamma=gam_i)
        Pb = C.exact_run("bare", mu_i, sig_i, rhoc, gamma=gam_i)[0]
        Pu = C.P_unc_exact(mu_i, sig_i, rhoc, gam_i)
        print(f"    P{i} tau/2={taus[i]:6.2f}: RMS={r:.3e}  "
              f"{'OK' if ok else 'FAIL'}   P_mem={Pm:.4f}  P_bare={Pb:.4f}  "
              f"P_unc={Pu:.4f}")
        out[f"P{i}_tau_half"] = taus[i]
        out[f"P{i}_mu"] = mu_i
        out[f"P{i}_sigma"] = sig_i
        out[f"P{i}_gamma"] = gam_i
        out[f"P{i}_theta"] = hw_i
        out[f"P{i}_rms"] = r
        out[f"P{i}_P_mem"] = Pm
        out[f"P{i}_P_bare"] = Pb
        out[f"P{i}_P_unc"] = Pu
    if not allok:
        raise SystemExit("  [C] failed for at least one point. STOP.")
    C.save(a.out_multi, **out)
    print(f"\n  wrote {a.out_multi}  ({n_pts} locked operating points)")

# ------------------------------------------------------------- stage targets
def stage_targets(a):
    z = np.load(a.params, allow_pickle=True)
    L = int(z["L"]); o = ast.literal_eval(str(z["opts"]))
    th = np.asarray(z["theta"], float)
    mu_tr, rhoc = float(z["mu"]), float(z["rho_c"])
    sig_tr, gam_tr = float(z["sigma"]), float(z["gamma"])

    T1, T2, Tphi = a.t1, a.t2, a.tphi
    if Tphi <= 0:
        if T1 <= 0 or T2 <= 0:
            run = C.Runner(a.backend, rep_delay_us=a.rep_delay)
            lay = ([int(x) for x in a.layout.split(",")] if a.layout
                   else C.pick_layout(run, a.idle_us, L))
            T1, T2, Tphi = run.coherence(lay)
            print(f"  coherence from {run.name}, qubits {lay}: "
                  f"T1={T1:.1f} T2={T2:.1f} us")
        else:
            Tphi = C.tphi_from(T1, T2)
    if not (np.isfinite(T1) and np.isfinite(Tphi) and T1 > 0 and Tphi > 0):
        raise SystemExit("  need finite positive T1 and T_phi; pass --t1/--t2 "
                         "or --tphi explicitly.")
    print(f"  T1={T1:.1f} us  T2={T2:.1f} us  T_phi={Tphi:.1f} us   "
          f"(1/T2 = 1/(2T1) + 1/T_phi; T2 ALREADY contains T1)")
    print(f"  trained at sigma={sig_tr}, gamma={gam_tr}, rho_c={rhoc}, "
          f"mu={mu_tr}")

    pad = max(0.0, float(a.pad_us))
    print(f"  matched pad = {pad:.3f} us per shot -> baselines are evaluated "
          f"at tau/2 + {pad/2:.3f} us, the QAE arm at tau/2")
    if pad == 0.0:
        print("  (pad = 0: pass --pad-us from 'hw_00_dryrun.py --stage "
              "estimate' to match a padded deployment)")

    if a.mu_per_us is not None:
        print(f"  coherent bias: mu(t) = {a.mu_per_us:+g} * t  (device's own, "
              "nothing injected)")
    else:
        mu_c = mu_tr if a.mu is None else a.mu
        print(f"  coherent bias: mu = {mu_c:+g} (constant)"
              + ("   <-- trained value; hardware injects NOTHING, consider "
                 "--mu 0" if a.mu is None else ""))

    def mu_at(t):
        if a.mu_per_us is not None:
            return a.mu_per_us * t
        return mu_tr if a.mu is None else a.mu

    tau_match = sig_tr ** 2 * Tphi / 2.0
    print(f"  the training point sigma={sig_tr} is reached at "
          f"tau/2 = {tau_match:.1f} us, where gamma = "
          f"{1-math.exp(-tau_match/T1):.4f} (trained {gam_tr})")

    chk_e = C.P_unc_exact(0.0, sig_tr, rhoc, 0.0)
    chk_f = C.P_unc_theory(0.0, sig_tr, rhoc, 0.0)
    print(f"  sanity: at mu=0, gamma=0 exact P_unc {chk_e:.9f} vs closed form "
          f"{chk_f:.9f}  d={abs(chk_e-chk_f):.2e}")

    taus = [float(x) for x in a.tau.split(",")]
    keys = ("tau_half", "t_base", "t_qae", "sigma", "gamma", "sigma_qae",
            "gamma_qae", "mu_base", "mu_qae", "P_bare", "P_unc", "P_ideal",
            "B_unc", "B_bare", "p2q_unc", "p2q_bare")
    out = {k: [] for k in keys}
    print(f"\n  {'tau/2':>7} {'t_base':>7} {'sig_b':>7} {'gam_b':>7} "
          f"{'sig_q':>7} {'P_bare':>9} {'P_unc':>9} {'P_ideal':>9} "
          f"{'B_unc':>9} {'p2q_crit':>10}")
    for t in taus:
        tb, tq = t + 0.5 * pad, t
        sb, gb = C.noise_at_tau(tb, T1, Tphi)
        sq, gq = C.noise_at_tau(tq, T1, Tphi)
        mb, mq = mu_at(tb), mu_at(tq)
        pb = C.P_bare_theory(mb, sb, gb)
        pu = C.exact_run("uncorr", mb, sb, rhoc, gamma=gb)[0]
        pi = C.P_ideal_exact(th, mq, sq, rhoc, gq, L, o, "selfinv")
        cu = C.p2q_crit(pi, pu, a.C_fit); cb = C.p2q_crit(pi, pb, a.C_fit)
        for k, v in zip(keys, (t, tb, tq, sb, gb, sq, gq, mb, mq,
                               pb, pu, pi, pi - pu, pi - pb, cu, cb)):
            out[k].append(v)
        print(f"  {t:7.1f} {tb:7.2f} {sb:7.3f} {gb:7.4f} {sq:7.3f} "
              f"{pb:9.6f} {pu:9.6f} {pi:9.6f} {pi-pu:+9.5f} {cu:10.2e}")

    print(f"\n  p2q_crit above uses C = {a.C_fit:g} (the fitted simulator "
          f"weight).\n  With eta = {a.eta} and C_count from --stage estimate, "
          f"use C_eff = eta*C_count.")
    print("  NOTE P_unc < P_bare is EXPECTED at gamma > 0: X_j commutes with "
          "both\n  stabilizers, so every T1 jump is a silent logical error and "
          "3 qubits\n  give 3x the rate. This is physics, not a bug.")
    C.save(a.targets_out, T1=T1, T2=T2, Tphi=Tphi, rho_c=rhoc, L=L,
           mu_train=mu_tr, sigma_train=sig_tr, gamma_train=gam_tr,
           tau_match=tau_match, pad_us=pad, C_fit=a.C_fit, eta=a.eta,
           mu_per_us=(np.nan if a.mu_per_us is None else a.mu_per_us),
           **{k: np.array(v, float) for k, v in out.items()})


# --------------------------------------------------------------- stage smoke
def stage_smoke(a):
    """Local-simulator plumbing test. Uses idle_impl='rz' with sampled angles,
    which reproduces the DEPHASING part exactly (gamma = 0). It only proves
    prep / decode / readout wiring, and never runs on hardware."""
    z = np.load(a.params, allow_pickle=True)
    L = int(z["L"]); o = ast.literal_eval(str(z["opts"]))
    th = np.asarray(z["theta"], float)
    rhoc, sig = float(z["rho_c"]), float(z["sigma"])
    cfg = C.Cfg(L=L, o=o, form="selfinv", idle_impl="rz", mu=0.0,
                sigma=sig, rho=rhoc, n_angle=a.n_angle)
    run = C.Runner("aer:fez", rep_delay_us=None)
    pm = run.pass_manager(None)
    tv = C.ParameterVector("t", C.n_params(L, o, "hw"))
    fv = C.ParameterVector("f", 6)
    ang = C.sample_angles(cfg, a.n_angle, 7)
    states = C.axis_states()
    print(f"  local ideal simulator, sigma={sig}, rho_c={rhoc}, gamma=0, "
          f"n_angle={a.n_angle}")

    tasks, ev = [], {}
    for v in ("qae", "uncorr", "bare"):
        for s in states:
            if v == "bare":
                axes = ("X", "Y") if s[3] in ("X", "Y") else (s[3],)
                for ax in axes:
                    qc = C.build_circuit(cfg, s, ax, "bare", 0, 0, tv, fv,
                                         slot=C.CODE[0])
                    tasks.append(((v, s[0], ax), pm.run(qc), 1))
            else:
                for g in C.AX_SET[s[3]]:
                    qc = C.build_circuit(cfg, s, g, v, 0, 0, tv, fv)
                    tasks.append(((v, s[0], g), pm.run(qc), 3))
    pubs = [(isa, C.param_array(isa, th[None, :] if k[0] == "qae"
                                else np.zeros((1, len(tv))), ang, tv, fv))
            for k, isa, nb in tasks]
    res = run.run(pubs, a.shots, "smoke")
    for (k, isa, nb), r in zip(tasks, res):
        pr = C.pub_probs(r.data.c, nb).reshape(-1, 1 << nb).mean(0)
        ev[k] = C.parities(pr[None, :], nb)[0]

    print(f"\n  {'arm':>10} {'measured':>10} {'exact':>10} {'|d|':>8}")
    ok = True
    for v in ("qae", "uncorr", "bare"):
        Ps = []
        for s in states:
            if v == "bare":
                Ps.append(0.5 * (1 + s[4] * ev[(v, s[0], s[3])][1]))
            else:
                Ps.append(0.5 * (1 + s[4] * C.decode(
                    {g: ev[(v, s[0], g)] for g in C.AX_SET[s[3]]})[s[3]]))
        got = float(np.mean(Ps))
        exact = C.exact_run(v, 0.0, sig, rhoc, theta=th, L=L, o=o,
                            form="selfinv", gamma=0.0)[0]
        d = abs(got - exact); ok &= d < 0.03
        print(f"  {v:>10} {got:10.4f} {exact:10.4f} {d:8.4f}")
    print("\n  PASS" if ok else "\n  FAIL: deviation > 0.03, check the wiring")


# ------------------------------------------------------------ stage estimate
def _n_settings(variants, n_slots):
    """exact number of circuits per tau, mirroring hw_01_deploy.py."""
    per = {}
    for v in variants:
        k = 0
        for _, _, _, ax, _ in C.axis_states():
            if v == "bare":
                k += (2 if ax in ("X", "Y") else 1) * n_slots
            else:
                k += len(C.AX_SET[ax])
        per[v] = k
    return per


def stage_estimate(a):
    z = np.load(a.params, allow_pickle=True)
    L = int(z["L"]); o = ast.literal_eval(str(z["opts"]))
    run = C.Runner(a.backend, rep_delay_us=a.rep_delay)
    lay = ([int(x) for x in a.layout.split(",")] if a.layout
           else C.pick_layout(run, a.idle_us, L))
    pm = run.pass_manager(lay if run.kind != "aer" else None)
    tv = C.ParameterVector("t", C.n_params(L, o, "hw"))
    cfg = C.Cfg(L=L, o=o, form=str(z["form"]))
    try:
        gran = int(run.tb.target.granularity) or 16
    except Exception:
        gran = 16
    d = run.idle_dt(a.idle_us)
    T1, T2, Tphi = run.coherence(lay[:3])
    p2q = run.median_p2q()
    print(f"  backend {run.name}  layout {lay}  L={L}  idle {a.idle_us} us")
    print(f"  dt = {run.dt:.6g} s   granularity = {gran} dt "
          f"({gran*run.dt*1e9:.2f} ns)")
    print(f"  code qubits {lay[:3]}: T1={T1:.1f} T2={T2:.1f} T_phi={Tphi:.1f} "
          f"us   median p2q = {p2q:.2e}")

    # ---- the matched pad, exactly as hw_01_deploy.py computes it
    st = C.axis_states()[0]
    q = pm.run(C.build_circuit(cfg, st, "XXX", "qae", d, 0, tv, None))
    b = pm.run(C.build_circuit(cfg, st, "XXX", "uncorr", d, 0, tv, None))
    raw = max(0.0, (run.circuit_seconds(q) - run.circuit_seconds(b)) / run.dt)
    mid = int(round(raw / gran)) * gran
    pad_us = mid * run.dt * 1e6
    print(f"\n  MATCHED PAD = {mid} dt = {pad_us:.3f} us   "
          f"(raw {raw:.1f} dt, snapped to {gran} dt)")
    print(f"  -> the baselines idle tau + {pad_us:.3f} us, the QAE arm idles "
          f"tau.\n     Pass --pad-us {pad_us:.3f} to --stage targets.")

    s_i, g_i = C.noise_at_tau(a.idle_us, T1, Tphi)
    s_b, g_b = C.noise_at_tau(a.idle_us + 0.5 * pad_us, T1, Tphi)
    print(f"  predicted at tau/2 = {a.idle_us} us:")
    print(f"     QAE arm      sigma_deph = {s_i:.3f}   gamma = {g_i:.4f}")
    print(f"     baselines    sigma_deph = {s_b:.3f}   gamma = {g_b:.4f}"
          f"   (pad included)")
    if pad_us > 0.5 * T2:
        print(f"  WARNING the pad ({pad_us:.1f} us) is comparable to T2 "
              f"({T2:.1f} us):\n  the baselines are largely dephased before "
              "the sweep even starts. Reduce L\n  or accept that only the "
              "smallest tau points carry information.")

    # ---- gate counts and durations, baselines WITH the pad
    print(f"\n  {'arm':>8} {'N2q':>5} {'N1q':>6} {'N1q_phys':>9} "
          f"{'C_count':>8} {'C_phys':>7} {'duration':>10}")
    tot = {}
    for v, setting in (("qae", "XXX"), ("uncorr", "XXX"), ("bare", "X")):
        qc = C.build_circuit(cfg, st, setting, v, d,
                             0 if v == "qae" else mid, tv, None, slot=0)
        isa = pm.run(qc)
        n2, n1, n1p = C.gate_counts(isa)
        tot[v] = (n2, n1, n1p, n2 + n1 / 10.0, n2 + n1p / 10.0,
                  run.circuit_seconds(isa))
        print(f"  {v:>8} {n2:5d} {n1:6d} {n1p:9d} {tot[v][3]:8.1f} "
              f"{tot[v][4]:7.1f} {tot[v][5]*1e6:9.2f} us")
    cc = tot["qae"][3]
    print(f"\n  C_count (paper convention, N2q + N1q/10) = {cc:.1f}")
    print(f"  C_eff  = eta * C_count = {a.eta} * {cc:.1f} = {a.eta*cc:.1f}")
    dur = [tot[v][5] * 1e6 for v in tot]
    print(f"  duration spread across arms = {max(dur)-min(dur):.2f} us "
          f"(0 would be perfect time matching)")

    # ---- exact circuit count and budget
    variants = [v for v in a.variants.split(",") if v]
    n_slots = max(1, min(3, a.bare_slots))
    per = _n_settings(variants, n_slots)
    n_tau = len(a.tau.split(","))
    print(f"\n  circuits per tau: "
          + ", ".join(f"{v}={per[v]}" for v in variants)
          + f"  -> {sum(per.values())} per tau, "
            f"{n_tau*sum(per.values())} total (+2 readout cal)")
    rep = (a.rep_delay or 0.0) * 1e-6
    sec = 0.0
    for v in variants:
        base = "qae" if v.startswith("qae") else v
        sec += per[v] * n_tau * a.shots * (tot[base][5] + rep)
    sec += 2 * 4096 * (2e-6 + rep)
    print(f"  budget at {a.shots} shots, rep_delay {a.rep_delay} us: "
          f"{sec:.0f} s = {sec/60:.1f} QPU min")

    if a.targets:
        t = np.load(a.targets, allow_pickle=True)
        print(f"\n  {'tau/2':>7} {'p2q_crit(C_fit)':>16} {'p2q_crit(C_eff)':>16}"
              f" {'device p2q':>11} {'verdict':>9}")
        for i, tt in enumerate(t["tau_half"]):
            pi, pu = float(t["P_ideal"][i]), float(t["P_unc"][i])
            c1 = C.p2q_crit(pi, pu, float(t["C_fit"]))
            c2 = C.p2q_crit(pi, pu, a.eta * cc)
            print(f"  {tt:7.1f} {c1:16.2e} {c2:16.2e} {p2q:11.2e} "
                  f"{'WIN' if p2q < c2 else 'above':>9}")


if __name__ == "__main__":
    args = cli()
    {"scan": stage_scan, "scan-multi": stage_scan_multi,
     "targets": stage_targets, "smoke": stage_smoke,
     "estimate": stage_estimate}[args.stage](args)