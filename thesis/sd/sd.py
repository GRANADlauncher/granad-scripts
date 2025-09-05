import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
plt.style.use("../thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9  

import matplotlib
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import get_cmap

from granad import *
from tbfit import *

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import TwoSlopeNorm

def pz(r):
    x,y,z = r
    a = 1
    r = jnp.sqrt(x*x + y*y + z*z)
    # cosθ = z/r (define as 0 at r=0 to avoid division by zero warnings)    
    cos_theta = jnp.where(r != 0.0, z / r, 0.0)

    prefac = 1.0 / (4.0 * jnp.sqrt(2.0 * jnp.pi))
    psi = prefac * (1.0 / (a ** 1.5)) * (r / a) * jnp.exp(-r / (2.0 * a)) * cos_theta
    return psi

# --- (A) draw band structure on a given Axes ---
def draw_band_structure_fit(ax):
    data = jnp.load('params.npz')
    hoppings, overlap, basis, positions, ks, dft_data = (
        data["hoppings"], data["overlap"], data["basis"],
        data["positions"], data["ks"], data["dft_data"]
    )

    # Tight-binding band structure
    H, S = hamiltonian_overlap(hoppings, overlap, basis, positions, ks.T)
    band = jnp.linalg.eigh(jnp.linalg.inv(S.T) @ H.T)[0][:, 0]

    # x-axis and k-point labels
    x_vals = jnp.arange(ks.shape[0])
    kpoint_indices = [0, 30, 60, 91]
    kpoint_labels = [r'$\Gamma$', 'K', 'M', 'K']

    # Plot
    ax.plot(x_vals, band, label='Tight-Binding Fit', linewidth=2)
    ax.plot(x_vals, dft_data, '.', label='DFT Data', markersize=6)

    ax.set_xticks(kpoint_indices)
    ax.set_xticklabels(kpoint_labels)

    ax.set_xlabel("k-path")
    ax.set_ylabel("Energy (eV)")
    ax.legend(frameon=False, fontsize='small', loc='best')
    ax.grid(True, linestyle=':', alpha=0.5)

    for kpt in kpoint_indices:
        ax.axvline(x=kpt, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

# --- (B) draw GS density grid on a given Axes ---
def draw_gs_grid(ax, flake, x, y, z, levels=200):
    # compute once
    X, Y, Z = jnp.meshgrid(x, y, z)
    positions = jnp.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    distance_vector = positions[:, None, :] - flake.positions
    pz_vals = jax.vmap(jax.vmap(pz, in_axes=0), in_axes=1)(distance_vector)

    vals = flake.initial_density_matrix.real @ pz_vals
    vals = jnp.sum(pz_vals * vals, axis=0)
    vals = vals.reshape(X[:, :, 0].shape)
    psi2 = np.asarray(jnp.abs(vals) ** 2)  # CPU numpy for matplotlib

    # colormap + norm
    cmap = get_cmap("plasma")
    norm = TwoSlopeNorm(vmin=psi2.min(), vcenter=np.median(psi2), vmax=psi2.max())

    # filled contours
    cf = ax.contourf(
        X[:, :, 0], Y[:, :, 0], psi2,
        levels=levels, cmap=cmap, norm=norm, antialiased=True
    )

    # light isocontours
    ax.contour(
        X[:, :, 0], Y[:, :, 0], psi2,
        levels=10, colors="k", linewidths=0.3, alpha=0.25
    )

    # atoms
    ax.scatter(
        *zip(*flake.positions[:, :2]),
        s=4, c="black", zorder=10,
        linewidth=0.5, edgecolors="white"
    )

    ax.set(xlabel="x (Å)", ylabel="y (Å)", aspect="equal")
    ax.set_facecolor("white")

    ax.grid(False)

    return cf  # return mappable for colorbar

# --- Put them side-by-side with (a)/(b) panel labels ---
def figure_band_plus_gs(flake, x, y, z, levels=200,
                        savepath_pdf="fig_band+gs.pdf", savepath_png=None):

    fig, axes = plt.subplots(1, 2, figsize=(LATEX_TEXTWIDTH_IN, LATEX_TEXTWIDTH_IN * 0.45), constrained_layout=True)


    # (a) band structure
    ax_a = axes[0]
    draw_band_structure_fit(ax_a)

    # (b) GS density
    ax_b = axes[1]
    mappable = draw_gs_grid(ax_b, flake, x, y, z, levels=levels)

    # single colorbar for panel (b)
    cbar = fig.colorbar(mappable, ax=ax_b, shrink=0.85, pad=0.02)
    cbar.set_label(r"$|\psi(x)|^2$")

    fig.canvas.draw_idle()  # make sure positions are up-to-date

    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()

    common_y = max(pos_a.y1, pos_b.y1) + 0.01  # a little above the taller axes
    offset = 0.01  # leftward offset

    fig.text(pos_a.x0 - offset, common_y, "(a)",
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)

    fig.text(pos_b.x0 - offset, common_y, "(b)",
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)

    # save
    fig.savefig(savepath_pdf, dpi=300, bbox_inches="tight")
    if savepath_png:
        fig.savefig(savepath_png, dpi=300, bbox_inches="tight")

    return fig, axes

# --- Optional: if you still want the original stand-alone API ---
def plot_band_structure_fit():
    plt.style.use("../thesis.mplstyle")
    fig, ax = plt.subplots(figsize=(7, 5))
    draw_band_structure_fit(ax)
    fig.suptitle("Tight-Binding vs DFT Band Structure")
    fig.savefig("graphene_tb_fit.pdf", dpi=300, bbox_inches='tight')
    return fig, ax

def evaluate_gs_grid(flake, x, y, z, levels=200, figsize=(4.0, 3.2)):
    plt.style.use("../thesis.mplstyle")
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    m = draw_gs_grid(ax, flake, x, y, z, levels=levels)
    cbar = fig.colorbar(m, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(r"$|\psi(x)|^2$")
    fig.savefig("sd.svg", bbox_inches="tight")
    fig.savefig("sd.pdf", bbox_inches="tight")
    return fig, ax


# use (hopping only, neglect overlap) values from tbfit
flake = get_graphene([0, -2.78  -0.36  -0.12  -0.068]).cut_flake(Triangle(18, armchair = True), plot = 0)
xmin, xmax = flake.positions[:, 0].min(), flake.positions[:, 0].max()
x = jnp.linspace(xmin - 1, xmax + 1, 40)
ymin, ymax = flake.positions[:, 1].min(), flake.positions[:, 1].max()
y = jnp.linspace(ymin - 1, ymax + 1, 40)
z = jnp.array([1])

# assuming you already have flake, x, y, z:
fig, axes = figure_band_plus_gs(flake, x, y, z, levels=200,
                                savepath_pdf="sd.pdf",
                                savepath_png="sd.png")
