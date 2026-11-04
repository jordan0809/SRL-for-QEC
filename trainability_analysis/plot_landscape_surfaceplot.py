# plot_landscape_surfaceplot.py
# Reload saved surfaces and regenerate the 4-panel overlay with fixed title + legend.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

sunset = LinearSegmentedColormap.from_list(
    "fidelity",
    [
        "#fff2c5",
        "#fca55d",
        "#d84a6a",
        "#8e1c81",
    ]
)

ocean = cm.viridis

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
SURF_FILE = "data/rep_bitflip_L2_surfaces.npz"
OUT_FILE  = "rep_bitflip_L2_overlay.pdf"
OUT_FILE  = "fidelity_vs_spectral_landscape.pdf"
NORM_MODE = "shared"          # "shared" | "per_surface" | "raw"
TITLE     = "Repetition code  p=0.05, L=2"

# ----------------------------------------------------------------
# Load surfaces
# ----------------------------------------------------------------
z = np.load(SURF_FILE)
ns = sorted({int(k.split("_")[0][1:]) for k in z.files if k.startswith("n") and k.endswith("_A")})
print(f"  Loaded {SURF_FILE} — n present: {ns}")

results = {}
for n in ns:
    A   = z[f"n{n}_A"]; B = z[f"n{n}_B"]
    FID = z[f"n{n}_FID"]; SPEC = z[f"n{n}_SPEC"]
    ci  = A.shape[0] // 2                     # center index (A=B=0)
    results[n] = dict(A=A, B=B, FID=FID, SPEC=SPEC,
                      fid_c=float(FID[ci, ci]), spec_c=float(SPEC[ci, ci]))

# ----------------------------------------------------------------
# Normalizers
# ----------------------------------------------------------------
def normalize_shared(FID, SPEC, cf, cs):
    FIDr, SPECr = FID - cf, SPEC - cs
    scale = max(np.abs(FIDr).max(), np.abs(SPECr).max(), 1e-12)
    return FIDr / scale, SPECr / scale

def normalize_per_surface(FID, SPEC):
    def mm(Z):
        lo, hi = Z.min(), Z.max()
        return (Z - lo) / (hi - lo) if hi > lo else np.zeros_like(Z)
    return mm(FID), mm(SPEC)

# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
fig = plt.figure(figsize=(11, 8.5))
for idx, n in enumerate(sorted(results.keys())):
    R = results[n]; A, B = R['A'], R['B']

    if NORM_MODE == "shared":
        FIDn, SPECn = normalize_shared(R['FID'], R['SPEC'], R['fid_c'], R['spec_c'])
    elif NORM_MODE == "per_surface":
        FIDn, SPECn = normalize_per_surface(R['FID'], R['SPEC'])
    else:
        FIDn, SPECn = R['FID'], R['SPEC']

    ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
    ax.set_facecolor("#ffffff")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#d0d0d0")
        axis._axinfo["grid"]["color"] = "#ffffff"
        axis._axinfo["grid"]["linewidth"] = 0.8
        axis._axinfo["grid"]["alpha"] = 0.7
    ax.plot_surface(
        A, B, FIDn,
        cmap=sunset,
        alpha=0.60,
        linewidth=0,
        antialiased=True,
        shade=True
    )

    ax.plot_surface(
        A, B, SPECn,
        cmap=ocean,
        alpha=0.60,
        linewidth=0,
        antialiased=True,
        shade=True
    )
    ci = A.shape[0] // 2
    #ax.scatter([0], [0], [FIDn[ci, ci]],  color='darkred', s=40, marker='o')
    #ax.scatter([0], [0], [SPECn[ci, ci]], color="#2ecc71",    s=40, marker='^')
    ax.scatter([0], [0], [FIDn[ci, ci]],
            color="#FF4F8B", s=45)

    ax.scatter([0], [0], [SPECn[ci, ci]],
            color="#0098C8", s=45)
    #ax.set_title(f"n={n}  (fid_center={R['fid_c']:.5f}, spec_center={R['spec_c']:.5f})",
                 #fontweight='bold', fontsize=10)
    ax.text2D(
        0.03, 0.96, f"n={n}",
        transform=ax.transAxes,
        fontsize=15,
        fontweight='bold',
        ha='left',
        va='top',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
    )
    ax.set_xlabel(r'$\theta_1$',fontsize = 15); ax.set_ylabel(r'$\theta_2$',fontsize =15); ax.set_zlabel(r'$\mathcal{L}(\theta)$',fontsize = 15, labelpad=12)
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.tick_params(axis='z', labelsize=12, pad=4)
    ax.zaxis.set_ticks_position('lower')
    ax.zaxis.set_label_position('lower')
    ax.view_init(elev=30, azim=-60)

# Title at top
#fig.suptitle(TITLE, fontsize=15, fontweight='bold', y=0.98)


fig.legend(
    handles=[
        Patch(color="#8e1c81", alpha=0.65, label="Fidelity loss"),
        Patch(color="#238a8d", alpha=0.65, label="Spectral loss"),
    ],
    loc="lower center",
    ncol=2,
    fontsize=13,
    frameon=False,
    bbox_to_anchor=(0.5, 0.015),
)

# Compact layout while leaving room for 3D tick labels and the legend.
fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.09,
                    hspace=0.02, wspace=0.02)

fig.savefig(OUT_FILE, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"  Saved -> {OUT_FILE}")