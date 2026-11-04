"""hw_02_aggregate.py -- mean/std across repeated hw_01_deploy runs, plus plots.

Usage (recommended, folder-based):
    python hw_02_aggregate.py --dir runs/r1_multipoint --tag r1_avg \
        --drift-tol 0.03

Usage (explicit file list, alternative):
    python hw_02_aggregate.py --deploys hw_deploy_r1_1.npz,hw_deploy_r1_2.npz \
        --tag r1_avg --drift-tol 0.03
"""
import argparse
import glob
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"qae_hw":    ("QPU-trained SRL",    "C4", "o"),
         "qae_clean": ("SRL", "C2", "o"),
         "uncorr":    ("Uncorrected logical", "C3", "s"),
         "qec":       ("Standard QEC", "C0", "D"),
         "bare":      ("Bare physical", "C1", "^")}


def load(path):
    return dict(np.load(path, allow_pickle=True))

def save_fidelity_csv(path, sweep, variants, p_mean, p_std):
    """Save the data used for the mean-fidelity plot.

    Columns:
        tau_half_us,
        P_<variant>_mean,
        P_<variant>_std
    """
    header = ["tau_half_us"]

    for variant in variants:
        header.extend([
            f"P_{variant}_mean",
            f"P_{variant}_std",
        ])

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for i, tau in enumerate(sweep):
            row = [float(tau)]

            for variant in variants:
                row.extend([
                    float(p_mean[variant][i]),
                    float(p_std[variant][i]),
                ])

            writer.writerow(row)


def save_noise_csv(path, sweep, noise_keys, noise_mean, noise_std):
    """Save the MEASURED noise-channel quantities (sigma_eq, sigma_deph,
    gamma_hw, mu_hw, t_eff): mean/std ACROSS the aggregated hardware
    sessions, reconstructed from the bare-arm Bloch vector (see
    hw_02_deploy.py's bare_bloch() + sig_eq_from/sig_deph_from/gamma_from,
    and mu_hw = 0.5*Om, the measured coherent phase bias). NOT a
    theoretical model fit.

    Columns:
        tau_half_us,
        <key>_mean,
        <key>_std
    for each key in noise_keys (sigma_eq, sigma_deph, gamma_hw, mu_hw, t_eff).
    """
    header = ["tau_half_us"]

    for key in noise_keys:
        header.extend([
            f"{key}_mean",
            f"{key}_std",
        ])

    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for i, tau in enumerate(sweep):
            row = [float(tau)]

            for key in noise_keys:
                row.extend([
                    float(noise_mean[key][i]),
                    float(noise_std[key][i]),
                ])

            writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="",
                   help="directory containing hw_deploy_*.npz repeats "
                        "(use instead of --deploys)")
    p.add_argument("--pattern", default="hw_deploy_*.npz",
                   help="glob pattern within --dir")
    p.add_argument("--deploys", default="",
                   help="explicit comma list of .npz paths (alternative to "
                        "--dir); SAME tau-sweep/layout required across all")
    p.add_argument("--tag", default="agg")
    p.add_argument("--outdir", default=".",
                   help="directory to write hw_agg_<tag>.* into")
    p.add_argument("--drift-tol", type=float, default=0.03,
                   help="drop runs whose |drift-1| exceeds this")
    p.add_argument("--dpi", type=int, default=170)
    a = p.parse_args()

    if a.dir:
        paths = sorted(glob.glob(os.path.join(a.dir, a.pattern)))
        if not paths:
            raise SystemExit(f"no files matching {a.pattern!r} in {a.dir}")
    elif a.deploys:
        paths = a.deploys.split(",")
    else:
        raise SystemExit("pass either --dir <folder> or --deploys <list>")

    print(f"  found {len(paths)} candidate files:")
    for pth in paths:
        print(f"    {pth}")

    runs = []
    for path in paths:
        d = load(path)
        drift = float(np.asarray(d.get("drift", np.nan)).ravel()[0])
        if np.isfinite(drift) and abs(drift - 1) > a.drift_tol:
            print(f"  DROPPED {path}: drift={drift:.4f} exceeds tol")
            continue
        runs.append((path, d))
    if not runs:
        raise SystemExit("no usable runs (all failed drift check)")

    ref_sweep = np.asarray(runs[0][1]["sweep"], float)
    for path, d in runs[1:]:
        if not np.allclose(np.asarray(d["sweep"], float), ref_sweep):
            raise SystemExit(f"{path}: tau-sweep does not match reference "
                             f"run {runs[0][0]}; cannot align")

    variants = [v.decode() if isinstance(v, bytes) else str(v)
                for v in np.asarray(runs[0][1]["variants"]).ravel()]
    n = ref_sweep.size

    print(f"\n  aggregating {len(runs)} / {len(paths)} runs "
          f"(dropped {len(paths)-len(runs)} for drift)")

    # ---- real, per-session stability check (the meaningful one --
    # NOT the static calibration-snapshot T1/T2, which is usually
    # identical across runs submitted within the same calibration
    # window and tells you nothing about stability DURING execution)
    print(f"  {'run':>40} {'T1':>7} {'T2':>7} {'Tphi':>8} {'drift':>7}")
    for path, d in runs:
        t1 = float(np.asarray(d["T1"]).ravel()[0])
        t2 = float(np.asarray(d["T2"]).ravel()[0])
        tphi = float(np.asarray(d["Tphi"]).ravel()[0])
        dr = float(np.asarray(d.get("drift", np.nan)).ravel()[0])
        print(f"  {path:>40} {t1:7.1f} {t2:7.1f} {tphi:8.1f} {dr:7.4f}")

    T1s = [float(np.asarray(d["T1"]).ravel()[0]) for _, d in runs]
    T2s = [float(np.asarray(d["T2"]).ravel()[0]) for _, d in runs]
    Tphis = [float(np.asarray(d["Tphi"]).ravel()[0]) for _, d in runs]
    drifts = [float(np.asarray(d.get("drift", np.nan)).ravel()[0]) for _, d in runs]
    print(f"\n  T1   = {np.mean(T1s):.1f} +- {np.std(T1s):.1f} us  "
          f"(n={len(T1s)}; calibration snapshot, not a live measurement)")
    print(f"  T2   = {np.mean(T2s):.1f} +- {np.std(T2s):.1f} us")
    print(f"  Tphi = {np.mean(Tphis):.1f} +- {np.std(Tphis):.1f} us")
    finite_dr = [x for x in drifts if np.isfinite(x)]
    if finite_dr:
        print(f"  drift (live, per-run SPAM start/end ratio) = "
              f"{np.mean(finite_dr):.4f} +- {np.std(finite_dr):.4f}  "
              f"(n={len(finite_dr)}; THIS is the real stability check)")

    out = dict(sweep=ref_sweep, variants=np.array(variants),
               n_runs=len(runs), source_files=np.array(paths),
               T1_mean=np.mean(T1s), T1_std=np.std(T1s),
               T2_mean=np.mean(T2s), T2_std=np.std(T2s),
               Tphi_mean=np.mean(Tphis), Tphi_std=np.std(Tphis),
               T1_all_runs=np.array(T1s), T2_all_runs=np.array(T2s),
               Tphi_all_runs=np.array(Tphis), drift_all_runs=np.array(drifts))

    print(f"\n  {'tau/2':>7} " + " ".join(f"{'P_'+v+'_mean':>14} {'P_'+v+'_std':>12}"
                                          for v in variants))
    P_mean, P_std = {}, {}
    for v in variants:
        key = f"P_{v}"
        stack = np.stack([np.asarray(d[key], float) for _, d in runs
                          if key in d])
        P_mean[v] = stack.mean(0)
        P_std[v] = stack.std(0)
        out[f"{key}_mean"] = P_mean[v]
        out[f"{key}_std"] = P_std[v]
    for i in range(n):
        row = f"  {ref_sweep[i]:7.2f} "
        row += " ".join(f"{P_mean[v][i]:14.4f} {P_std[v][i]:12.4f}"
                        for v in variants)
        print(row)

   
    noise_keys = ("sigma_eq", "sigma_deph", "gamma_hw", "mu_hw", "t_eff")
    noise_mean, noise_std = {}, {}
    print(f"\n  {'tau/2':>7} " + " ".join(
        f"{k+'_mean':>14} {k+'_std':>12}" for k in noise_keys))
    for key in noise_keys:
        stack = np.stack([np.asarray(d[key], float) for _, d in runs
                          if key in d])
        noise_mean[key] = np.nanmean(stack, axis=0)
        noise_std[key] = np.nanstd(stack, axis=0)
        out[f"{key}_mean"] = noise_mean[key]
        out[f"{key}_std"] = noise_std[key]
    for i in range(n):
        row = f"  {ref_sweep[i]:7.2f} "
        row += " ".join(f"{noise_mean[k][i]:14.4f} {noise_std[k][i]:12.4f}"
                        for k in noise_keys)
        print(row)

    os.makedirs(a.outdir, exist_ok=True)
    out_npz = os.path.join(a.outdir, f"hw_agg_{a.tag}.npz")
    np.savez_compressed(out_npz, **out)
    print(f"\n  saved {out_npz}")


    out_p_csv = os.path.join(a.outdir, f"hw_agg_{a.tag}_P.csv")
    save_fidelity_csv(
        path=out_p_csv,
        sweep=ref_sweep,
        variants=variants,
        p_mean=P_mean,
        p_std=P_std,
    )
    print(f"  saved {out_p_csv}")

    out_noise_csv = os.path.join(a.outdir, f"hw_agg_{a.tag}_noise.csv")
    save_noise_csv(
        path=out_noise_csv,
        sweep=ref_sweep,
        noise_keys=noise_keys,
        noise_mean=noise_mean,
        noise_std=noise_std,
    )
    print(f"  saved {out_noise_csv}")

    # ================================================================
    # PLOT 1 -- P vs tau/2, mean +- std band, one curve per arm
    # ================================================================
    x = ref_sweep
    f1, ax = plt.subplots(figsize=(7.4, 5.1))
    for v in variants:
        lab, col, mk = STYLE.get(v, (v, "C7", "d"))
        ax.plot(x, P_mean[v], mk + "-", color=col, ms=6, lw=1.8,
                label=f"{lab}")
        ax.fill_between(x, P_mean[v] - P_std[v], P_mean[v] + P_std[v],
                        color=col, alpha=0.18)
    ax.axhline(0.5, color="k", lw=0.9, ls=":")
    ax.set_xlabel(r"$\tau/2$ [$\mu$s]",fontsize=15)
    ax.set_ylabel(r"$\langle F \rangle$",fontsize=15)
    #ax.set_title(f"averaged over {len(runs)} independent hardware sessions"
    #            f"\n(shaded = across-run std, i.e. session-to-session "
    #            f"variation, not shot noise)", fontsize=10)
    
    ax.grid(alpha=0.25)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(frameon=False, fontsize=12)
    f1.tight_layout()
    out_p_png = os.path.join(a.outdir, f"hw_agg_{a.tag}_P.png")
    out_p_pdf = os.path.join(a.outdir, f"hw_agg_{a.tag}_P.pdf")
    f1.savefig(out_p_png, dpi=a.dpi)
    f1.savefig(out_p_pdf)
    plt.close(f1)

    # ================================================================
    # PLOT 2 -- MEASURED sigma_deph / sigma_eq / gamma_hw vs tau/2,
    # mean +- std ACROSS THE 10 RUNS. No theoretical model curves
    # ================================================================
    f2, ax2 = plt.subplots(figsize=(7.4, 5.1))
    ax2b = ax2.twinx()

    sd_m, sd_s = noise_mean["sigma_deph"], noise_std["sigma_deph"]
    se_m, se_s = noise_mean["sigma_eq"], noise_std["sigma_eq"]
    g_m, g_s = noise_mean["gamma_hw"], noise_std["gamma_hw"]

    ok_sd = np.isfinite(sd_m)
    ok_se = np.isfinite(se_m)
    ok_g = np.isfinite(g_m)

    l1 = ax2.errorbar(x[ok_sd], sd_m[ok_sd], yerr=sd_s[ok_sd], fmt="o-",
                      color="C0", ms=6, capsize=3, lw=1.6,
                      label=r"$\sigma_{\rm deph}$")
    l2 = ax2.errorbar(x[ok_se], se_m[ok_se], yerr=se_s[ok_se], fmt="s--",
                      color="C1", ms=6, capsize=3, lw=1.4, alpha=0.85,
                      label=r"$\sigma_{\rm eq}$")
    l3 = ax2b.errorbar(x[ok_g], g_m[ok_g], yerr=g_s[ok_g], fmt="^-",
                       color="C5", ms=6, capsize=3, lw=1.6,
                       label=r"$\gamma$")

    ax2.set_xlabel(r"$\tau/2$ [$\mu$s]",fontsize=15)
    ax2.set_ylabel(r"$\sigma$",fontsize=15)
    ax2b.set_ylabel(r"$\gamma$", color="C5",fontsize=15)
    ax2b.tick_params(axis="y", labelcolor="C5")
    #ax2.set_title(f"MEASURED device noise per delay setting\n"
    #              f"mean $\\pm$ std across {len(runs)} independent hardware "
    #              f"sessions (bare-arm reconstruction, no model fit)",
    #              fontsize=10)
    
    ax2.grid(alpha=0.25)
    handles = [l1, l2, l3]
    labels = [h.get_label() for h in handles]
    ax2.legend(handles, labels, frameon=False, fontsize=12, loc="upper left")
    ax2.tick_params(axis='both', labelsize=12)
    ax2b.tick_params(axis='y', labelsize=12)
    f2.tight_layout()
    out_n_png = os.path.join(a.outdir, f"hw_agg_{a.tag}_noise.png")
    out_n_pdf = os.path.join(a.outdir, f"hw_agg_{a.tag}_noise.pdf")
    f2.savefig(out_n_png, dpi=a.dpi)
    f2.savefig(out_n_pdf)
    plt.close(f2)

    print(f"  wrote {out_p_png} / {out_p_pdf}")
    print(f"  wrote {out_n_png} / {out_n_pdf}")


if __name__ == "__main__":
    main()