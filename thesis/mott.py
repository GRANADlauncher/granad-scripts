import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import jax.numpy as jnp

from granad import *
from granad._plotting import *

plt.style.use("thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9  

# --- Your spin-pol plot, but always draw on a provided axis if given ---
def plot_spin_polarization(orbs, show_tags=None, circle_scale: float = 1e3, inset_ax=None, ax=None):
    up_idxs = jnp.array(orbs.filter_orbs("up", int))
    down_idxs = jnp.array(orbs.filter_orbs("down", int))
    rho = orbs.initial_density_matrix
    diff = rho.diagonal()[up_idxs] - rho.diagonal()[down_idxs]

    # Choose axis: explicit ax > inset_ax (back-compat) > new fig
    draw_ax = ax if ax is not None else (inset_ax if inset_ax is not None else plt.subplots(figsize=(5, 4))[1])

    positions = jnp.unique(orbs.positions, axis=0)
    for pos, d in zip(positions, diff.real):
        if d > 0:
            draw_ax.scatter(pos[0], pos[1], s=circle_scale*0.01, c='tab:blue', marker='v', edgecolors='k')
        else:
            draw_ax.scatter(pos[0], pos[1], s=circle_scale*0.01, c='tab:red', marker='^', edgecolors='k')

    draw_ax.set_aspect('equal')
    draw_ax.set_xticks([])
    draw_ax.set_yticks([])
    draw_ax.grid(False)
    return draw_ax

# --- Energy spectrum that plots into a given axis and adds a tiny orbital inset ---
def show_energies_ax(flake, ax, inset_size=("30%", "30%")):
    energies = flake.energies
    e_max = energies.max()
    e_min = energies.min()
    padding = (e_max - e_min) * 0.01
    e_max += padding
    e_min -= padding

    mask = (energies >= e_min) & (energies <= e_max)
    state_indices = jnp.arange(len(energies))[mask]
    filtered_energies = energies[mask]

    # Occupation weight
    colors = jnp.diag(flake.electrons * flake.initial_density_matrix_e)[mask]

    label = "Initial State Occupation"

    sc = ax.scatter(state_indices, filtered_energies, c=colors, cmap="viridis", edgecolors='k', alpha=0.9)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(label, fontsize=9)

    ax.set_xlabel("Eigenstate Index", fontsize=10)
    ax.set_ylabel(r"$E/t$", fontsize=10)
    ax.set_ylim(e_min, e_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title("Energy Spectrum", fontsize=11)
    return ax

def get_hubbard(U):
    t = 1. # nearest-neighbor hopping
    return (
        Material("Hubbard")
        .lattice_constant(1.0)
        .lattice_basis([[1, 0, 0],[0, 1, 0]])
        .add_orbital_species("up", s = -1)
        .add_orbital_species("down", s = 1)
        .add_orbital(position=(0, 0), species = "up",   tag = "up")
        .add_orbital(position=(0, 0), species = "down", tag = "down")
        .add_interaction("hamiltonian", participants=("up","up"),   parameters=[0.0, t])
        .add_interaction("hamiltonian", participants=("down","down"), parameters=[0.0, t])
        .add_interaction("coulomb",     participants=("up","down"), parameters=[U])
    )

# ------------------------
# Build flakes for U = 0, 5 and plot 2x2 grid
# ------------------------
rect = Rectangle(20, 8)
flake0 = get_hubbard(0).cut_flake(rect)
flake0.set_open_shell()
flake0.set_electrons(len(flake0)//2)

flake5 = get_hubbard(5).cut_flake(rect)
flake5.set_open_shell()
flake5.set_electrons(len(flake0)//2)
flake5.set_mean_field()

fig, axs = plt.subplots(2, 2, figsize=(LATEX_TEXTWIDTH_IN, LATEX_TEXTWIDTH_IN * 0.45))
((ax_a, ax_b), (ax_c, ax_d)) = axs

# (a) Spin pol @ U=0
plot_spin_polarization(flake0, ax=ax_a)
ax_a.set_title("Spin polarization (U = 0)")
ax_a.text(0.02, 0.96, "(a)", transform=ax_a.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

# (b) Energies @ U=0
show_energies_ax(flake0, ax_b)
ax_b.text(0.02, 0.96, "(b)", transform=ax_b.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")
ax_b.set_title("Energies (U = 0)")

# (c) Spin pol @ U=5
plot_spin_polarization(flake5, ax=ax_c)
ax_c.set_title("Spin polarization (U = 5)")
ax_c.text(0.02, 0.96, "(c)", transform=ax_c.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

# (d) Energies @ U=5
show_energies_ax(flake5, ax_d)
ax_d.text(0.02, 0.96, "(d)", transform=ax_d.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")
ax_d.set_title("Energies (U = 5)")

plt.tight_layout()
plt.savefig("mott.pdf")
