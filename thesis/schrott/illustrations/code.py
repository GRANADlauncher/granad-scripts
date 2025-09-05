# Create two example figures: (1) a clean pipeline diagram with a scissors icon, (2) a publication-style line plot.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
import numpy as np

# ---------- 1) Pipeline diagram (SVG + PDF) ----------
plt.rcParams.update({
    "figure.constrained_layout.use": True,
    "font.size": 11,
    "font.family": "DejaVu Sans",
})

fig, ax = plt.subplots(figsize=(8, 2.2))
ax.set_axis_off()

# Helper to draw a labeled rounded box
def rounded_box(ax, xy, width, height, label):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2
    )
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, label,
            ha="center", va="center")
    return box

# Positions
y = 0.35
w, h = 1.9, 0.55

b1 = rounded_box(ax, (0.3, y), w, h, "Structure Builder")
b2 = rounded_box(ax, (2.6, y), w, h, "Geometry Ops\n(cutting, tiling)")
b3 = rounded_box(ax, (4.9, y), w, h, "Static Solver")
b4 = rounded_box(ax, (7.2, y), w, h, "Dynamic Simulation")

# Arrows
def connect(a, b, text=None):
    ax.add_patch(FancyArrowPatch(
        (a.get_x()+a.get_width(), a.get_y()+a.get_height()/2),
        (b.get_x(), b.get_y()+b.get_height()/2),
        arrowstyle=ArrowStyle.Simple(head_length=6, head_width=6, tail_width=1.2),
        mutation_scale=10,
        linewidth=1.1
    ))
    if text:
        xmid = (a.get_x()+a.get_width() + b.get_x())/2
        ymid = a.get_y()+a.get_height()/2 + 0.05
        ax.text(xmid, ymid, text, ha="center", va="bottom", fontsize=10)

connect(b1, b2)
connect(b2, b3)
connect(b3, b4)

# Scissors icon near "Geometry Ops"
ax.text(b2.get_x()+b2.get_width()/2, b2.get_y()+b2.get_height()+0.18, "✂",
        ha="center", va="center", fontsize=16)
ax.text(b2.get_x()+b2.get_width()/2, b2.get_y()+b2.get_height()+0.02, "material cutting",
        ha="center", va="center", fontsize=9)

# Title-like caption
ax.text(0.02, 0.95, "Nanostructure Simulation Pipeline", transform=ax.transAxes, va="top", fontsize=12)

# Save vector outputs
pipeline_pdf = "nanostructure_pipeline.pdf"
fig.savefig(pipeline_pdf, bbox_inches="tight")
plt.close(fig)


# # ---------- 2) Publication-style line plot (SVG + PDF) ----------
# # Sample data
# x = np.linspace(0, 10, 200)
# y = np.exp(-0.2*x)*np.cos(2*np.pi*0.4*x)

# fig2, ax2 = plt.subplots(figsize=(4.2, 3.1))
# ax2.plot(x, y, linewidth=2, label="I(t) / I₀")
# ax2.set_xlabel("Time t [ps]")
# ax2.set_ylabel("Normalized current")
# ax2.grid(True, alpha=0.35)
# ax2.legend(frameon=False, loc="upper right")
# ax2.annotate("decay envelope",
#              xy=(6, np.exp(-0.2*6)),
#              xytext=(7.5, 0.7),
#              arrowprops=dict(arrowstyle="->", linewidth=1.0),
#              fontsize=10)

# plot_svg = "publication_plot_template.svg"
# plot_pdf = "publication_plot_template.pdf"
# fig2.savefig(plot_svg, bbox_inches="tight")
# fig2.savefig(plot_pdf, bbox_inches="tight")
# plt.close(fig2)
