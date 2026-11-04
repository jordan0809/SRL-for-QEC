"""
Replot script for the Steane combined-noise spectral-QEC experiment
"""

import os, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =====================================================================
# CONFIG  —  edit everything here
# =====================================================================
CONFIG = dict(

    # ---------- INPUT FILES (explicit names; change the folder later) ----------
    files = dict(
        results  = "data/steane_combined_results_layerwise.npz",
        grid_npz = "data/steane_combined_grid_layerwise.npz",
        grid_csv = "data/steane_combined_grid_layerwise.csv",   # fallback if npz missing
        cuts_npz = "data/steane_combined_cuts_layerwise.npz",
        cuts_csv = "data/steane_combined_cuts_layerwise.csv",   # fallback if npz missing
    ),

    # ---------- OUTPUT FILES (base names, no extension) ----------
    out_files = dict(
        fig1 = "steane_combined_fidelity_layerwise",
        fig2 = "steane_combined_grid_heatmaps_layerwise",
        fig3 = "steane_combined_cuts_layerwise",
    ),

    # ---------- input / output folders & format ----------
    in_dir    = ".",                # used only for bare file names above
    out_dir   = ".",
    out_suffix= "",          # appended to output file names ("" = overwrite)
    # fmt       = "png",              # png | pdf | svg
    fmt       = "pdf",              # png | pdf | svg
    dpi       = 200,

    # ---------- LEGEND / SERIES LABELS ----------
    labels = dict(
        qae   = "Spectral QAE",     # "{L}" is replaced by the best layer count
        petz  = "Petz map",
        std   = "std QEC",
        noisy = "No correction",
        qae_layer = "L={nl} ({np}p)",           # per-layer curve label, Fig.1
        qae_layer_fid = "L={nl} ({np}p)  F={f:.4f}",
    ),
    # append the mean fidelity to the baseline hlines in Fig.1 panel (a)?
    append_value_to_baseline_labels = True,

    # ---------- TITLES ----------
    titles = dict(
        fig1_suptitle = ("Spectral QAE — Steane [[7,1,3]] combined noise "
                         "(p={p}, σ={sigma}, ρ={rho}, μ={mu}, γ={gamma})"),
        fig1_a = "Training fidelity vs epoch (±1σ)",
        fig1_b = "Baseline loss vs epoch",
        fig1_c = "Test fidelity vs layer count",
        fig1_d = "Logical spectrum error vs layer count",

        fig2_suptitle = ("Steane [[7,1,3]] combined noise (best L={nl}, trained @ p={p}, σ={sigma}, γ={gamma})"),
        # one title template per column key
        # fig2_col = dict(
        #     diff_std   = "F({qae}) − F({std})",
        #     diff_petz  = "F({qae}) − F({petz})",
        #     diff_noisy = "F({qae}) − F({noisy})",
        #     qae = "{qae}", petz = "{petz}", std = "{std}", noisy = "{noisy}",
        # ),
        fig2_col = dict(
            # diff_std   = "vs. std QEC",
            # diff_petz  = "vs. Petz map",
            # diff_noisy = "vs. no correction",
            diff_std   = r"vs. $\bf{{std\ QEC}}$",
            diff_petz  = r"vs. $\bf{{Petz\ map}}$",
            diff_noisy = r"vs. $\bf{{no\ correction}}$",
            qae = "{qae}", petz = "{petz}", std = "{std}", noisy = "{noisy}",
        ),
        # fig2_panel = "γ={gamma:g} — {col}",       # row+col title
        fig2_panel = "γ={gamma:g} {col}",           # row+col title: gamma value + vs. baseline
        fig2_stats = "⟨Δ⟩={mean:+.4f}, win {win}/{tot}",   # 2nd title line, see grid.panel_stats

        fig3_suptitle = ("Steane [[7,1,3]] combined noise — 1D cuts "
                         "(best L={nl}, other parameters at the training point)"),
        fig3_panel = "vary {axis}",
    ),

    # ---------- AXIS LABELS ----------
    axis = dict(
        epoch   = "Global epoch",
        fid     = "Mean fidelity",
        fid_tr  = "Avg fidelity (training set)",
        loss    = "Fixed-weight baseline spectral loss",
        specerr = "Mean |target − recovered|",
        # p       = "depolarizing $p$",
        # sigma   = "dephasing $\\sigma$",
        p       = "$p$",
        sigma   = "$\\sigma$",
        gamma   = "amplitude damping $\\gamma$",
        cbar_f  = "mean fidelity",
        cbar_d  = "$\\Delta F$",
    ),

    # ---------- COLORS ----------
    colors = dict(qae='#2ecc71', petz='#f39c12', std='#3498db', noisy='#e74c3c',
                  zl='#e74c3c', xl='#3498db', yl='#9b59b6',
                  layers=['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']),
    linestyles = dict(qae='o-', petz='d-.', std='s--', noisy='^:'),

    # ---------- WHICH SERIES TO SHOW ----------
    show = dict(
        fig1_std=True,  fig1_petz=True,  fig1_noisy=True,
        fig3_std=True,  fig3_petz=True,  fig3_noisy=True,
        fig1_petz_spec_lines=True,      # Petz ΔZL/ΔXL hlines in panel (d)
        errorbars=True,                 # Fig.1 (c) bar error bars
        bands=True,                     # Fig.1 (a)/(b) ±1σ shading
        fig3_bands=False,               # <-- variance band on the 1D cuts: OFF
    ),

    # ---------- FIG.2 heat maps ----------
    grid = dict(
        # column keys: 'diff_std' | 'diff_petz' | 'diff_noisy'
        #              (absolute maps still available: 'qae','petz','std','noisy')
        columns=('diff_std', 'diff_petz', 'diff_noisy'),
        cmap='viridis',              # for absolute-fidelity columns
        cmap_diff='RdBu_r',          # for difference columns
        diff_scale='column',         # 'column' | 'panel' | 'shared'
        shared_scale=True,           # shared vmin/vmax across absolute columns
        # annotate=False, annot_fs=7, annot_fmt="{:.3f}", annot_fmt_diff="{:+.3f}",
        annotate=True, annot_fs=11, annot_fmt="{:.3f}", annot_fmt_diff="{:+.3f}",
        diff_headroom=1.5,           # >1 lightens diff colors so red/blue aren't too dark
        hatch_negative=False,         # mark cells where the QAE loses
        note_colorkey = False,       # add a note to the suptitle about the color key
        # panel_stats=True,            # add mean Δ + win count under each panel title
        panel_stats=False,           # add mean Δ + win count under each panel title
        gammas=None,                 # None = all rows; e.g. [0.05, 0.30] to subset
        mark_training_point=True,
        panel_w=5.3, panel_h=4.4,
    ),

    # ---------- FONT SIZES ----------
    fs = dict(suptitle=12, title=11, label=11, tick=10, legend=9, annot=8),

    # ---------- misc ----------
    grid_alpha = 0.3,
    figs = (1, 2, 3),
)
# =====================================================================


# ---------------------------------------------------------------------
# CLI overrides
# ---------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir");  ap.add_argument("--out-dir")
    ap.add_argument("--out-suffix"); ap.add_argument("--fmt")
    ap.add_argument("--dpi", type=int)
    ap.add_argument("--figs", help="comma list, e.g. 1,3")
    ap.add_argument("--results", help="explicit path to the results .npz")
    ap.add_argument("--grid",    help="explicit path to the grid .npz or .csv")
    ap.add_argument("--cuts",    help="explicit path to the cuts .npz or .csv")
    a = ap.parse_args()

    for k in ("fmt", "out_suffix"):
        if getattr(a, k) is not None: CONFIG[k] = getattr(a, k)
    if a.in_dir:  CONFIG["in_dir"]  = a.in_dir
    if a.out_dir: CONFIG["out_dir"] = a.out_dir
    if a.dpi:     CONFIG["dpi"]     = a.dpi
    if a.figs:    CONFIG["figs"]    = tuple(int(s) for s in a.figs.split(","))

    if a.results: CONFIG["files"]["results"] = a.results
    for kind, val in (("grid", a.grid), ("cuts", a.cuts)):
        if not val: continue
        ext = os.path.splitext(val)[1].lower()
        CONFIG["files"][f"{kind}_npz"] = val if ext != ".csv" else None
        CONFIG["files"][f"{kind}_csv"] = val if ext == ".csv" else None


# ---------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------
def _expand(name):
    return os.path.expanduser(os.path.expandvars(name))

def inpath(key):
    """Resolve CONFIG['files'][key] -> full path (or None if disabled)."""
    name = CONFIG["files"].get(key)
    if not name: return None
    name = _expand(name)
    if os.path.isabs(name) or os.path.dirname(name):
        return name                                   # folder given -> use as is
    return os.path.join(CONFIG["in_dir"], name)       # bare name -> in_dir

def outpath(key):
    """Resolve CONFIG['out_files'][key] -> full path incl. suffix + extension."""
    name = _expand(CONFIG["out_files"][key])
    base, ext = os.path.splitext(name)
    if ext.lower() in (".png", ".pdf", ".svg"):
        fname = f"{base}{CONFIG['out_suffix']}{ext}"
    else:
        fname = f"{name}{CONFIG['out_suffix']}.{CONFIG['fmt']}"
    if not (os.path.isabs(fname) or os.path.dirname(fname)):
        fname = os.path.join(CONFIG["out_dir"], fname)
    d = os.path.dirname(fname)
    if d: os.makedirs(d, exist_ok=True)
    return fname

def _f(x):
    """scalar from a 0-d numpy array"""
    return float(np.asarray(x).reshape(-1)[0])

def fmt_g(x):
    return f"{x:g}"

def save(fig, key):
    fp = outpath(key)
    fig.savefig(fp, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  saved : {fp}")

def lbl(key, **kw):
    return CONFIG["labels"][key].format(**kw)


# ---------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------
def load_results():
    f = inpath("results")
    if f is None:
        print("  [skip] results file disabled (files['results'] = None)"); return None
    if not os.path.exists(f):
        print(f"  [skip] {f} not found"); return None
    print(f"  loading: {f}")
    z = np.load(f, allow_pickle=True)
    layers = [int(v) for v in np.atleast_1d(z["meta_layers"])]
    R = dict(
        layers=layers,
        best_nl=int(_f(z["best_n_layers"])),
        f_noisy=np.asarray(z["f_noisy_test"]),
        f_std=np.asarray(z["f_std_test"]),
        f_petz=np.asarray(z["f_petz_test"]),
        meta=dict(p=_f(z["meta_p"]), sigma=_f(z["meta_sigma"]),
                  rho=_f(z["meta_rho_corr"]), mu=_f(z["meta_mu"]),
                  gamma=_f(z["meta_gamma"])),
        per={}, hist={},
    )
    for k, dst in [("petz_dzl_test", "pz_dzl"), ("petz_dxl_test", "pz_dxl"),
                   ("petz_dyl_test", "pz_dyl")]:
        R[dst] = np.asarray(z[k]) if k in z.files else None
    for nl in layers:
        R["per"][nl] = dict(
            n_params=len(np.asarray(z[f"L{nl}_params"])),
            f_qae=np.asarray(z[f"L{nl}_f_qae_test"]),
            dzl=np.asarray(z[f"L{nl}_dzl_test"]),
            dxl=np.asarray(z[f"L{nl}_dxl_test"]),
            dyl=np.asarray(z[f"L{nl}_dyl_test"]),
        )
        hk = [k.split(f"L{nl}_hist_")[1] for k in z.files if k.startswith(f"L{nl}_hist_")]
        R["hist"][nl] = {k: np.asarray(z[f"L{nl}_hist_{k}"]) for k in hk}
    return R


def load_grid():
    fz, fc = inpath("grid_npz"), inpath("grid_csv")

    if fz and os.path.exists(fz):
        print(f"  loading: {fz}")
        z = np.load(fz, allow_pickle=True)
        return dict(p=np.atleast_1d(z["p_grid"]).astype(float),
                    sigma=np.atleast_1d(z["sigma_grid"]).astype(float),
                    gamma=np.atleast_1d(z["gamma_grid"]).astype(float),
                    qae=np.asarray(z["F_qae"]), petz=np.asarray(z["F_petz"]),
                    std=np.asarray(z["F_std"]), noisy=np.asarray(z["F_noisy"]),
                    best_nl=int(_f(z["best_n_layers"])))

    if not (fc and os.path.exists(fc)):
        print(f"  [skip] grid data not found ({fz or '-'} / {fc or '-'})"); return None

    print(f"  loading: {fc}  (CSV fallback)")
    rows = list(csv.DictReader(open(fc)))
    g = np.array(sorted({float(r["gamma"]) for r in rows}))
    s = np.array(sorted({float(r["sigma"]) for r in rows}))
    p = np.array(sorted({float(r["p"])     for r in rows}))
    sh = (len(g), len(s), len(p))
    M = {k: np.full(sh, np.nan) for k in ("qae", "petz", "std", "noisy")}
    col = dict(qae="F_qae", petz="F_petz", std="F_stdQEC", noisy="F_noisy")
    for r in rows:
        i = int(np.argmin(abs(g - float(r["gamma"]))))
        j = int(np.argmin(abs(s - float(r["sigma"]))))
        k = int(np.argmin(abs(p - float(r["p"]))))
        for key, c in col.items(): M[key][i, j, k] = float(r[c])
    return dict(p=p, sigma=s, gamma=g, best_nl=None, **M)


def load_cuts():
    fz, fc = inpath("cuts_npz"), inpath("cuts_csv")

    if fz and os.path.exists(fz):
        print(f"  loading: {fz}")
        z = np.load(fz, allow_pickle=True)
        out = {}
        for ax in ("p", "sigma", "gamma"):
            if f"{ax}_values" not in z.files: continue
            out[ax] = dict(v=np.asarray(z[f"{ax}_values"]),
                           qae=np.asarray(z[f"{ax}_F_qae"]),
                           qae_std=np.asarray(z[f"{ax}_F_qae_std"]),
                           petz=np.asarray(z[f"{ax}_F_petz"]),
                           std=np.asarray(z[f"{ax}_F_std"]),
                           noisy=np.asarray(z[f"{ax}_F_noisy"]))
        return out

    if not (fc and os.path.exists(fc)):
        print(f"  [skip] cuts data not found ({fz or '-'} / {fc or '-'})"); return None

    print(f"  loading: {fc}  (CSV fallback)")
    rows = list(csv.DictReader(open(fc)))
    out = {}
    for ax in ("p", "sigma", "gamma"):
        rr = sorted([r for r in rows if r["cut_axis"] == ax], key=lambda r: float(r["value"]))
        if not rr: continue
        out[ax] = dict(v=np.array([float(r["value"]) for r in rr]),
                       qae=np.array([float(r["F_qae"]) for r in rr]),
                       qae_std=np.array([float(r["F_qae_std"]) for r in rr]),
                       petz=np.array([float(r["F_petz"]) for r in rr]),
                       std=np.array([float(r["F_stdQEC"]) for r in rr]),
                       noisy=np.array([float(r["F_noisy"]) for r in rr]))
    return out


# ---------------------------------------------------------------------
# FIGURE 1 — layer sweep
# ---------------------------------------------------------------------
def fig1(R):
    C, T, A, CO, S, FS = (CONFIG, CONFIG["titles"], CONFIG["axis"],
                          CONFIG["colors"], CONFIG["show"], CONFIG["fs"])
    L = R["layers"]; ga = C["grid_alpha"]
    def bl(key, arr):
        t = CONFIG["labels"][key]
        return f"{t} {arr.mean():.4f}" if C["append_value_to_baseline_labels"] else t

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # (a) training fidelity
    ax = axes[0, 0]
    for i, nl in enumerate(L):
        h = R["hist"][nl]; c = CO["layers"][i % len(CO["layers"])]
        if "fid" not in h: continue
        ax.plot(h["epoch"], h["fid"], color=c, lw=2,
                ls='-' if i % 2 == 0 else '--',
                label=lbl("qae_layer_fid", nl=nl, np=R["per"][nl]["n_params"],
                          f=R["per"][nl]["f_qae"].mean()))
        if S["bands"] and "fid_std" in h:
            ax.fill_between(h["epoch"], h["fid"]-h["fid_std"], h["fid"]+h["fid_std"],
                            color=c, alpha=0.12)
    if S["fig1_std"]:
        ax.axhline(R["f_std"].mean(), color=CO["std"], ls='--', lw=1.5, label=bl("std", R["f_std"]))
    if S["fig1_petz"]:
        ax.axhline(R["f_petz"].mean(), color=CO["petz"], ls='-.', lw=1.5, label=bl("petz", R["f_petz"]))
    if S["fig1_noisy"]:
        ax.axhline(R["f_noisy"].mean(), color='gray', ls=':', lw=1.5, label=bl("noisy", R["f_noisy"]))
    ax.set_xlabel(A["epoch"], fontsize=FS["label"]); ax.set_ylabel(A["fid_tr"], fontsize=FS["label"])
    ax.set_title(T["fig1_a"], fontweight='bold', fontsize=FS["title"])
    ax.legend(fontsize=FS["legend"]-1, loc='lower right'); ax.grid(True, alpha=ga)

    # (b) loss
    ax = axes[0, 1]
    for i, nl in enumerate(L):
        h = R["hist"][nl]; c = CO["layers"][i % len(CO["layers"])]
        if "loss" not in h: continue
        ax.plot(h["epoch"], h["loss"], color=c, lw=2, ls='-' if i % 2 == 0 else '--',
                label=lbl("qae_layer", nl=nl, np=R["per"][nl]["n_params"]))
        if S["bands"] and "loss_std" in h:
            ax.fill_between(h["epoch"], h["loss"]-h["loss_std"], h["loss"]+h["loss_std"],
                            color=c, alpha=0.12)
    ax.set_xlabel(A["epoch"], fontsize=FS["label"]); ax.set_ylabel(A["loss"], fontsize=FS["label"])
    ax.set_title(T["fig1_b"], fontweight='bold', fontsize=FS["title"])
    ax.legend(fontsize=FS["legend"]-1); ax.grid(True, alpha=ga)

    # (c) test fidelity bars
    ax = axes[1, 0]; x = np.arange(len(L))
    series = []
    if S["fig1_noisy"]: series.append((CONFIG["labels"]["noisy"], [R["f_noisy"].mean()]*len(L), CO["noisy"], None))
    if S["fig1_std"]:   series.append((CONFIG["labels"]["std"],   [R["f_std"].mean()]*len(L),   CO["std"],   None))
    if S["fig1_petz"]:  series.append((CONFIG["labels"]["petz"],  [R["f_petz"].mean()]*len(L),  CO["petz"],  None))
    series.append((CONFIG["labels"]["qae"].format(L=R["best_nl"]),
                   [R["per"][nl]["f_qae"].mean() for nl in L], CO["qae"],
                   [R["per"][nl]["f_qae"].std() for nl in L]))
    n = len(series); bw = 0.8/n
    for k, (name, vals, col, err) in enumerate(series):
        off = (k - (n-1)/2) * bw
        b = ax.bar(x+off, vals, bw, label=name, color=col, alpha=0.85)
        if err is not None:
            if S["errorbars"]:
                ax.errorbar(x+off, vals, yerr=err, fmt='none', color='black', capsize=4)
            for bb, v in zip(b, vals):
                ax.text(bb.get_x()+bb.get_width()/2, bb.get_height()+0.003, f"{v:.4f}",
                        ha='center', va='bottom', fontsize=FS["annot"]-1, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in L], fontsize=FS["tick"])
    ax.set_ylabel(A["fid"], fontsize=FS["label"])
    ax.set_title(T["fig1_c"], fontweight='bold', fontsize=FS["title"])
    ax.legend(fontsize=FS["legend"]); ax.grid(True, alpha=ga, axis='y')

    # (d) logical spectrum error
    ax = axes[1, 1]; bw2 = 0.25
    ax.bar(x-bw2, [R["per"][nl]["dzl"].mean() for nl in L], bw2, label="$\\Delta Z_L$", color=CO["zl"], alpha=0.85)
    ax.bar(x,     [R["per"][nl]["dxl"].mean() for nl in L], bw2, label="$\\Delta X_L$", color=CO["xl"], alpha=0.85)
    ax.bar(x+bw2, [R["per"][nl]["dyl"].mean() for nl in L], bw2, label="$\\Delta Y_L$", color=CO["yl"], alpha=0.85)
    if S["fig1_petz_spec_lines"] and R["pz_dzl"] is not None:
        pl = CONFIG["labels"]["petz"]
        ax.axhline(R["pz_dzl"].mean(), color=CO["petz"], ls='-.', lw=1.3, label=f"{pl} $\\Delta Z_L$")
        ax.axhline(R["pz_dxl"].mean(), color=CO["petz"], ls='--', lw=1.1, label=f"{pl} $\\Delta X_L$")
    ax.set_xticks(x); ax.set_xticklabels([f"L={nl}" for nl in L], fontsize=FS["tick"])
    ax.set_ylabel(A["specerr"], fontsize=FS["label"])
    ax.set_title(T["fig1_d"], fontweight='bold', fontsize=FS["title"])
    ax.legend(fontsize=FS["legend"]); ax.grid(True, alpha=ga, axis='y')

    m = R["meta"]
    plt.suptitle(T["fig1_suptitle"].format(p=fmt_g(m["p"]), sigma=fmt_g(m["sigma"]),
                                           rho=fmt_g(m["rho"]), mu=fmt_g(m["mu"]),
                                           gamma=fmt_g(m["gamma"])),
                 fontsize=FS["suptitle"], fontweight='bold', y=1.01)
    plt.tight_layout(); save(fig, "fig1")


# ---------------------------------------------------------------------
# FIGURE 2 — difference heat-map grid
# ---------------------------------------------------------------------
def _is_diff(key):
    return key.startswith("diff_")

def _ref_of(key):
    return key.split("_", 1)[1]          # 'diff_std' -> 'std'

def fig2(G, meta=None, best_nl=None):
    C, T, A, GC, FS = CONFIG, CONFIG["titles"], CONFIG["axis"], CONFIG["grid"], CONFIG["fs"]
    LB = CONFIG["labels"]
    nl = best_nl if best_nl is not None else (G["best_nl"] or "?")
    name = dict(qae=LB["qae"].format(L=nl), petz=LB["petz"],
                std=LB["std"], noisy=LB["noisy"])

    rows = list(range(len(G["gamma"]))) if GC["gammas"] is None else \
           [int(np.argmin(abs(G["gamma"]-g))) for g in GC["gammas"]]
    cols = list(GC["columns"])
    nR, nC = len(rows), len(cols)

    # ---- matrices ----
    def mat(key, ig):
        return (G["qae"][ig] - G[_ref_of(key)][ig]) if _is_diff(key) else G[key][ig]

    # ---- color scales ----
    diff_cols = [c for c in cols if _is_diff(c)]
    abs_cols  = [c for c in cols if not _is_diff(c)]
    vmin = vmax = None
    if GC["shared_scale"] and abs_cols:
        vmin = float(min(np.nanmin(G[c][rows]) for c in abs_cols))
        vmax = float(max(np.nanmax(G[c][rows]) for c in abs_cols))
    dmax_col = {}
    if diff_cols:
        per_col = {c: max(float(np.nanmax(np.abs(G["qae"][rows]-G[_ref_of(c)][rows]))), 1e-6)
                   for c in diff_cols}
        if GC["diff_scale"] == "shared":
            d = max(per_col.values()); dmax_col = {c: d for c in diff_cols}
        elif GC["diff_scale"] == "column":
            dmax_col = per_col
        # 'panel' -> computed inside the loop

    col_title = {k: v.format(**name) for k, v in T["fig2_col"].items()}

    fig, axes = plt.subplots(nR, nC, figsize=(GC["panel_w"]*nC, GC["panel_h"]*nR),
                             squeeze=False)
    for r, ig in enumerate(rows):
        for c, key in enumerate(cols):
            ax = axes[r][c]
            M = mat(key, ig)

            if _is_diff(key):
                dmax = dmax_col.get(key) or max(float(np.nanmax(np.abs(M))), 1e-6)
                dmax *= GC.get("diff_headroom", 1.0)
                norm = TwoSlopeNorm(vmin=-dmax, vcenter=0.0, vmax=dmax)
                im = ax.imshow(M, origin='lower', aspect='auto',
                               cmap=GC["cmap_diff"], norm=norm)
                lo, hi, cb, fmt = -dmax, dmax, A["cbar_d"], GC["annot_fmt_diff"]
            else:
                lo = vmin if vmin is not None else float(np.nanmin(M))
                hi = vmax if vmax is not None else float(np.nanmax(M))
                im = ax.imshow(M, origin='lower', aspect='auto',
                               cmap=GC["cmap"], vmin=lo, vmax=hi)
                cb, fmt = A["cbar_f"], GC["annot_fmt"]

            ax.set_xticks(np.arange(len(G["p"])));     ax.set_xticklabels([fmt_g(v) for v in G["p"]])
            ax.set_yticks(np.arange(len(G["sigma"]))); ax.set_yticklabels([fmt_g(v) for v in G["sigma"]])
            # ax.set_xlabel(A["p"], fontsize=15); ax.set_ylabel(A["sigma"], fontsize=15)
            if r == nR - 1: ax.set_xlabel(A["p"], fontsize=15)
            if c == 0: ax.set_ylabel(A["sigma"], fontsize=15)
            ax.tick_params(axis='both', labelsize=12)

            # ttl = T["fig2_panel"].format(gamma=G["gamma"][ig], col=col_title.get(key, key))
            # ax.set_title(ttl, fontsize=FS["title"])
            # if r == 0:
            #     ttl = T["fig2_panel"].format(gamma=G["gamma"][ig], col=col_title.get(key, key))
            #     ax.set_title(ttl, fontsize=15)
            if r == 0:
                ttl = T["fig2_panel"].format(gamma=G["gamma"][ig], col=col_title.get(key, key))
            else:
                ttl = f"γ={G['gamma'][ig]:g}"
            ax.set_title(ttl, fontsize=15)

            # annotations + loss hatching
            for a_ in range(M.shape[0]):
                for b_ in range(M.shape[1]):
                    val = M[a_, b_]
                    if not np.isfinite(val): continue
                    if _is_diff(key) and GC["hatch_negative"] and val < 0:
                        ax.add_patch(plt.Rectangle((b_-0.5, a_-0.5), 1, 1, fill=False,
                                                   hatch='///', edgecolor='0.25', lw=0.0))
                    if GC["annotate"]:
                        rel = (val-lo)/(hi-lo+1e-12)
                        tc = 'black' if _is_diff(key) else ('white' if rel < 0.5 else 'black')
                        ax.text(b_, a_, fmt.format(val), ha='center', va='center',
                                fontsize=GC["annot_fs"], color=tc,
                                fontweight='bold' if _is_diff(key) else 'normal')

            if GC["mark_training_point"] and meta is not None:
                if np.isclose(G["gamma"][ig], meta["gamma"], atol=1e-9) and \
                   np.any(np.isclose(G["p"], meta["p"])) and np.any(np.isclose(G["sigma"], meta["sigma"])):
                    # ax.plot(int(np.argmin(abs(G["p"]-meta["p"]))),
                    #         int(np.argmin(abs(G["sigma"]-meta["sigma"]))),
                    #         marker='*', ms=14, mfc='none', mec='lime', mew=2.2)
                    tp_i = int(np.argmin(abs(G["p"]-meta["p"])))
                    tp_j = int(np.argmin(abs(G["sigma"]-meta["sigma"])))
                    ax.add_patch(plt.Rectangle((tp_i-0.5, tp_j-0.5), 1, 1, fill=False,
                                               edgecolor='lime', lw=2.5))

            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=cb)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cbar.set_label(r"$\Delta \langle F \rangle$", fontsize=15)
            cbar.ax.tick_params(labelsize=12)

        st = T["fig2_suptitle"].format(
        nl=nl,
        p=fmt_g(meta["p"]) if meta else "?",
        sigma=fmt_g(meta["sigma"]) if meta else "?",
        gamma=fmt_g(meta["gamma"]) if meta else "?")

    extra = []
    if diff_cols and GC.get("note_colorkey", False):
        extra.append("blue = QAE better")
    if diff_cols and GC.get("hatch_negative", False) and GC.get("note_hatch", True):
        extra.append("hatched = QAE worse")
    if GC.get("mark_training_point", False) and meta is not None:
        extra.append("★ = training point")
    if extra:
        st += "   (" + ", ".join(extra) + ")"

    #plt.suptitle(st, fontsize=FS["suptitle"], fontweight='bold', y=1.005)
    plt.tight_layout()
    save(fig, "fig2")          


# ---------------------------------------------------------------------
# FIGURE 3 — 1D cuts
# ---------------------------------------------------------------------
def fig3(K, meta=None, best_nl=None):
    C, T, A, CO, LS, S, FS = (CONFIG, CONFIG["titles"], CONFIG["axis"], CONFIG["colors"],
                              CONFIG["linestyles"], CONFIG["show"], CONFIG["fs"])
    LB = CONFIG["labels"]; axes_order = [a for a in ("p", "sigma", "gamma") if a in K]
    fig, axs = plt.subplots(1, len(axes_order), figsize=(6.4*len(axes_order), 5.5), squeeze=False)
    for ax, key in zip(axs[0], axes_order):
        d = K[key]; v = d["v"]
        ax.plot(v, d["qae"], LS["qae"], color=CO["qae"], lw=2,
                label=LB["qae"].format(L=best_nl if best_nl is not None else "?"))
        if S["fig3_bands"]:                      # variance band (off by default)
            ax.fill_between(v, d["qae"]-d["qae_std"], d["qae"]+d["qae_std"],
                            color=CO["qae"], alpha=0.15)
        if S["fig3_petz"]:  ax.plot(v, d["petz"],  LS["petz"], color=CO["petz"], lw=2, label=LB["petz"])
        if S["fig3_std"]:   ax.plot(v, d["std"],   LS["std"],  color=CO["std"],  lw=1.6, label=LB["std"])
        if S["fig3_noisy"]: ax.plot(v, d["noisy"], LS["noisy"], color=CO["noisy"], lw=1.4, label=LB["noisy"])
        if meta is not None and key in meta:
            tp = meta[key]
            ax.axvline(tp, color='black', ls=':', alpha=0.6)
            lo, hi = ax.get_ylim()
            ax.text(tp, lo+0.03*(hi-lo), f"  trained @ {fmt_g(tp)}", fontsize=FS["annot"]-1)
        ax.set_xlabel(A[key], fontsize=FS["label"]); ax.set_ylabel(A["fid"], fontsize=FS["label"])
        ax.set_title(T["fig3_panel"].format(axis=A[key]), fontweight='bold', fontsize=FS["title"]-1)
        ax.grid(True, alpha=C["grid_alpha"]); ax.legend(fontsize=FS["legend"]-1)
    plt.suptitle(T["fig3_suptitle"].format(nl=best_nl if best_nl is not None else "?"),
                 fontsize=FS["suptitle"], fontweight='bold', y=1.03)
    plt.tight_layout(); save(fig, "fig3")


# ---------------------------------------------------------------------
def main():
    parse_args()
    print(f"  replotting -> *{CONFIG['out_suffix']}.{CONFIG['fmt']}  (out_dir={CONFIG['out_dir']})")
    R = load_results()
    meta   = R["meta"]    if R else None
    bestnl = R["best_nl"] if R else None
    #if 1 in CONFIG["figs"] and R: fig1(R)
    if 2 in CONFIG["figs"]:
        G = load_grid()
        if G: fig2(G, meta=meta, best_nl=bestnl if bestnl is not None else G["best_nl"])
    #if 3 in CONFIG["figs"]:
    #    K = load_cuts()
    #    if K: fig3(K, meta=meta, best_nl=bestnl)
    print("  done ✓")


if __name__ == "__main__":
    main()