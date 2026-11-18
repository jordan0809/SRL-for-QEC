# plot_bp_scaling.py — raw Var[∂C]: fidelity vs spectral, semilog + log-log views
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---- config ----
CSV_IN  = "data/bp_scaling_repetition_n11.csv"
PDF_OUT = "bp_scaling_repetition_combined_n11.pdf"
MODES   = ["fidelity", "spectral_full"]
LABELS  = {"fidelity": "fidelity", "spectral_full": "spectral"}

# axis colors (strongly contrasting)
C_LIN = "#1f4e9c"   # bottom axis: blue
C_LOG = "#d35400"   # top axis: orange

# per-curve colors: similar shade within each axis group
COLOR = {
    "lin": {"fidelity": "#1f4e9c", "spectral_full": "#5dade2"},
    "log": {"fidelity": "#d35400", "spectral_full": "#f5b041"},
}
LS     = {"fidelity": "-",  "spectral_full": "--"}   # line style -> loss type
MARKER = {"fidelity": "s",  "spectral_full": "o"}    # marker     -> loss type

TICKS = [3, 5, 7, 9, 11]
XLIM  = (2.7, 11.5)   # same range on both axes -> endpoints align

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

# ---- fits (only used for legend values) ----
def _mask(y):
    y = np.asarray(y, float)
    return y, (y > 0) & np.isfinite(y)

def exp_slope(y):
    """Var ~ exp(a*n): slope of log(y) vs n."""
    y, mk = _mask(y)
    return np.polyfit(n[mk], np.log(y[mk]), 1)[0] if mk.sum() >= 2 else np.nan

def pow_exponent(y):
    """Var ~ n^b: slope of log(y) vs log(n)."""
    y, mk = _mask(y)
    return np.polyfit(np.log(n[mk]), np.log(y[mk]), 1)[0] if mk.sum() >= 2 else np.nan

# ---- figure ----
fig, ax_lin = plt.subplots(figsize=(9, 7))
ax_log = ax_lin.twiny()           # same y-axis, own x-axis on top
ax_lin.set_yscale("log")
ax_log.set_xscale("log")

h_lin, l_lin, h_log, l_log = [], [], [], []
for m in MODES:
    y = data[m]

    # linear-n curve (bottom axis)
    ln, = ax_lin.plot(n, y, ls=LS[m], marker=MARKER[m], color=COLOR["lin"][m],
                      lw=3, ms=12)
    h_lin.append(ln)
    l_lin.append(f"{LABELS[m]}  slope={exp_slope(y):+.2f}")

    # log-n curve (top axis), hollow markers for extra separation
    ln, = ax_log.plot(n, y, ls=LS[m], marker=MARKER[m], color=COLOR["log"][m],
                      lw=3, ms=12, markerfacecolor="white", markeredgewidth=2.5)
    h_log.append(ln)
    l_log.append(f"{LABELS[m]}  slope={pow_exponent(y):+.2f}")

# ---- x-axes ----
for axis in (ax_lin, ax_log):
    axis.set_xlim(XLIM)
    axis.set_xticks(TICKS)
    axis.set_xticklabels([str(t) for t in TICKS])
ax_log.xaxis.set_minor_locator(mticker.NullLocator())

ax_lin.set_xlabel("n  (linear scale)", fontsize=20, color=C_LIN)
ax_log.set_xlabel("n  (log scale)", fontsize=20, color=C_LOG)
ax_lin.tick_params(axis="x", colors=C_LIN, labelsize=20)
ax_log.tick_params(axis="x", colors=C_LOG, labelsize=20)

ax_lin.spines["top"].set_visible(False)
ax_log.spines["bottom"].set_visible(False)
ax_lin.spines["bottom"].set_color(C_LIN); ax_lin.spines["bottom"].set_linewidth(2)
ax_log.spines["top"].set_color(C_LOG);    ax_log.spines["top"].set_linewidth(2)

# ---- y-axis, grid ----
ax_lin.set_ylabel(r"Var[$\partial C$]", fontsize=20)
ax_lin.tick_params(axis="y", labelsize=20)
ax_lin.grid(True, alpha=0.3, which="both")

# ---- two grouped legends (both on top axes so they are drawn above all lines) ----
leg_lin = ax_log.legend(h_lin, l_lin, title="linear n — exp. test",
                        fontsize=16, title_fontsize=16, loc="lower left",
                        edgecolor=C_LIN)
leg_lin.get_title().set_color(C_LIN)
ax_log.add_artist(leg_lin)        # keep first legend when adding the second

leg_log = ax_log.legend(h_log, l_log, title="log n — power-law test",
                        fontsize=16, title_fontsize=16, loc="upper right",
                        edgecolor=C_LOG)
leg_log.get_title().set_color(C_LOG)

plt.tight_layout()
fig.savefig(PDF_OUT, dpi=150, bbox_inches="tight")
print(f"  Saved : {PDF_OUT}")
plt.close(fig)