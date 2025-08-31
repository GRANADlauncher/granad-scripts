# Minimalist matplotlib illustration of a 1D Mott metal -> insulator transition
import numpy as np
import matplotlib.pyplot as plt

# Set up canvas
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_xlim(-5.6, 5.6)
ax.set_ylim(-1.6, 1.6)
ax.axis('off')

# Lattice sites
n_sites = 12
sites = np.linspace(-5, 5, n_sites)
y0 = 0.0

# Draw sites as hollow circles and a thin backbone
ax.plot(sites, np.full_like(sites, y0), linewidth=1.0)
for x in sites:
    ax.plot([x], [y0], marker='o', markersize=8, fillstyle='none', linewidth=1.5)

# Separator between regimes
ax.axvline(0, linestyle='--', linewidth=1.0)

# --- Left: Metal (delocalized electrons) ---
# Wavy "electron cloud" along the chain to suggest delocalization
t = np.linspace(-5.0, 0.0, 400)
wave = 0.55 * np.sin(3 * np.pi * (t - t.min()) / (t.max() - t.min()))
ax.plot(t, wave, linewidth=2.0)
ax.plot(t, -wave, linewidth=2.0, alpha=0.8)

# Light "hopping" arcs between neighboring sites (only on left half)
left_sites = sites[sites < 0]
for i in range(len(left_sites) - 1):
    xa, xb = left_sites[i], left_sites[i + 1]
    mid = 0.5 * (xa + xb)
    xs = np.linspace(xa, xb, 60)
    # simple arch
    ys = 0.35 * np.sin(np.pi * (xs - xa) / (xb - xa))
    ax.plot(xs, ys, linewidth=0.8, alpha=0.8)

# --- Right: Mott Insulator (localized spins, one per site) ---
right_sites = sites[sites > 0]
# Alternate up/down spin arrows on each site (one electron per site)
for j, x in enumerate(right_sites):
    direction = 1 if j % 2 == 0 else -1
    ax.arrow(x, 0, 0, 0.75 * direction, head_width=0.12, head_length=0.12, length_includes_head=True, linewidth=1.8)

# Small "no-double-occupancy" x marks near each right site to hint at localization
for x in right_sites:
    ax.plot([x - 0.1, x + 0.1], [0.18, -0.18], linewidth=1.0)
    ax.plot([x - 0.1, x + 0.1], [-0.18, 0.18], linewidth=1.0)

# Trend arrow for increasing U/W
ax.annotate(
    "",
    xy=(4.7, 1.25),
    xytext=(-4.7, 1.25),
    arrowprops=dict(arrowstyle="->", lw=1.6)
)
ax.text(0, 1.33, "increasing U/W", ha="center", va="bottom", fontsize=11)

# Labels
ax.text(-3.2, 1.05, "metal (delocalized)", ha="center", va="center", fontsize=11)
ax.text(3.2, 1.05, "Mott insulator (localized)", ha="center", va="center", fontsize=11)

# Save
out_path = "/mnt/data/mott_metal_to_insulator_chain.png"
fig.savefig(out_path, dpi=220, bbox_inches="tight")
