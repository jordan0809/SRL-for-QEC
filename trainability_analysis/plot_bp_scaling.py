# plot_bp_scaling.py — raw Var[∂C]: fidelity vs spectral, semilog + log-log views
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- config ----
#CSV_IN        = "bp_scaling_repetition.csv"
CSV_IN        = "data/bp_scaling_repetition_n11.csv"
PNG_SEMILOG   = "bp_scaling_repetition_semilog_n11.pdf"
PNG_LOGLOG    = "bp_scaling_repetition_loglog_n11.pdf"
MODES  = ["fidelity", "spectral_full"]          
cmap = {
    "fidelity":  "#e74c3c",
    "spectral_full": '#3498db',
}
mark   = {"fidelity": "s-",      "spectral_full": "o-"}
LABELS = {"fidelity": "fidelity", "spectral_full": "spectral"}

# ---- load per-n data section (stop at blank line before summary) ----
rows = []
with open(CSV_IN) as f:
    header = f.readline().strip().split(",")
    for line in f:
        s = line.strip()
        if s == "":
            break
        rows.append(s.split(","))

col  = {name: i for i, name in enumerate(header)}
n    = np.array([float(r[col["n_code"]]) for r in rows])
data = {m: np.array([float(r[col[m + "_var"]]) for r in rows]) for m in MODES}

# ---- fits ----
def exp_fit(y):
    """Var ~ exp(a*n): linear fit of log(y) vs n."""
    y = np.asarray(y, float); mk = (y > 0) & np.isfinite(y)
    if mk.sum() < 2:
        return np.nan, None
    a, b = np.polyfit(n[mk], np.log(y[mk]), 1)
    return a, (a, b)

def pow_fit(y):
    """Var ~ n^b: linear fit of log(y) vs log(n)."""
    y = np.asarray(y, float); mk = (y > 0) & np.isfinite(y)
    if mk.sum() < 2:
        return np.nan, None
    b, c = np.polyfit(np.log(n[mk]), np.log(y[mk]), 1)
    return b, (b, c)

# ---- figure 1: semilog-y (straight line => exponential decay) ----
fig, ax = plt.subplots(figsize=(8, 6))
for m in MODES:
    y = data[m]
    a, fit = exp_fit(y)
    ax.semilogy(n, y, mark[m], color=cmap[m], lw=2, ms=8,
                label=f"{LABELS[m]}  slope={a:+.2f}")
    if fit is not None:
        aa, bb = fit
        nn = np.linspace(n.min(), n.max(), 100)
        ax.semilogy(nn, np.exp(aa * nn + bb), color=cmap[m], lw=1, ls=":", alpha=0.6)

ax.set_xlabel("n",fontsize = 15)
ax.set_ylabel(r"Var[$\partial C$]",fontsize = 15)
ax.set_xticks([3, 5, 7, 9, 11])
ax.tick_params(axis="both", labelsize=12)
#ax.set_title(r"Var[$\partial C$] decay — semilog (exponential test)")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(PNG_SEMILOG, dpi=150, bbox_inches="tight")
print(f"  Saved : {PNG_SEMILOG}")
plt.close(fig)

# ---- figure 2: log-log (straight line => polynomial decay) ----
fig, ax = plt.subplots(figsize=(8, 6))
for m in MODES:
    y = data[m]
    b, fit = pow_fit(y)
    ax.loglog(n, y, mark[m], color=cmap[m], lw=2, ms=8,
              label=f"{LABELS[m]}  exponent={b:+.2f}")
    if fit is not None:
        bb, cc = fit
        nn = np.linspace(n.min(), n.max(), 100)
        ax.loglog(nn, np.exp(cc) * nn ** bb, color=cmap[m], lw=1, ls=":", alpha=0.6)

ax.set_xlabel("n",fontsize = 15)
ax.set_ylabel(r"Var[$\partial C$]",fontsize = 15)
ax.set_xticks([3, 5, 7, 9, 11])
ax.set_xticklabels(["3", "5", "7", "9", "11"], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
#ax.set_xticklabels([f"{int(v)}" for v in n])   # readable ticks on log x-axis
ax.xaxis.set_minor_locator(mticker.NullLocator())   # drop the 4·10⁰ / 6·10⁰ labels
#ax.set_title(r"Var[$\partial C$] decay — log-log (polynomial test)")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
fig.savefig(PNG_LOGLOG, dpi=150, bbox_inches="tight")
print(f"  Saved : {PNG_LOGLOG}")
plt.close(fig)