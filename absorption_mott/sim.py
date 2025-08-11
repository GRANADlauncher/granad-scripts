import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from collections import defaultdict
import numpy as np
import jax.numpy as jnp

import jax.numpy as jnp
from granad import *
from granad._plotting import *

def plot_orbital_layout_2d(orbs, show_tags=None, circle_scale: float = 1e3, inset_ax=None):
    """
    Displays a 2D orbital layout with tagging, either as a standalone plot or an inset.

    Parameters:
        orbs: list of orbital objects with `.tag` and `.position`.
        show_tags: list or set of tags to include (defaults to all present).
        circle_scale: float scaling factor for marker size.
        inset_ax: optional matplotlib axis object (for embedding as inset).
    """
    up_idxs = jnp.array(orbs.filter_orbs("up", int))
    down_idxs = jnp.array(orbs.filter_orbs("down", int))
    rho = orbs.initial_density_matrix
    diff = rho.diagonal()[up_idxs] - rho.diagonal()[down_idxs]

    # Choose axis: standalone or inset
    ax = inset_ax if inset_ax else plt.subplots(figsize=(5, 4))[1]
    positions = jnp.unique(orbs.positions, axis = 0)
    for pos, d in zip(positions, diff.real):
        if d > 0:
            ax.scatter(pos[0], pos[1], s=circle_scale*0.01, c='tab:blue', marker='v', edgecolors='k')
        else:
            ax.scatter(pos[0], pos[1], s=circle_scale*0.01, c='tab:red', marker='^', edgecolors='k')
    
    # Format
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if not inset_ax:
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        plt.tight_layout()
        plt.show()

def show_energies_with_inset_2d(flake, name):
    """
    Shows the energy spectrum with occupation and an inset showing 2D orbital layout.
    """

    # --- Setup Energy Spectrum Plot (Main Axes) ---
    energies = flake.energies
    e_max = energies.max()
    e_min = energies.min()

    padding = (e_max - e_min) * 0.01
    e_max += padding
    e_min -= padding

    mask = (energies >= e_min) & (energies <= e_max)
    state_indices = jnp.arange(len(energies))[mask]
    filtered_energies = energies[mask]

    display1 = jnp.diag(flake.electrons * flake.initial_density_matrix_e)
    colors = display1[mask]
    label = "Initial State Occupation"

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        state_indices,
        filtered_energies,
        c=colors,
        cmap="viridis",
        edgecolors='k',
        alpha=0.9
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label, fontsize=10)

    ax.set_xlabel("Eigenstate Index", fontsize=11)
    ax.set_ylabel(r"$\frac{E}{t}$", fontsize=11)
    ax.set_ylim(e_min, e_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title("Energy Spectrum with Occupation Weight", fontsize=13)

    inset_ax = ax.inset_axes([0.0, 0.6, 0.7, 0.7])
    plot_orbital_layout_2d(flake, circle_scale=1e3, inset_ax=inset_ax)

    plt.tight_layout()
    plt.savefig(name)
    plt.close()
        
def get_hubbard(U):
    t = 1. # nearest-neighbor hopping
    # U => onsite coulomb repulsion for opposite spins

    hubbard = (
        Material("Hubbard")
        .lattice_constant(1.0)
        .lattice_basis([
            [1, 0, 0],
            [0, 1, 0],
        ])
        .add_orbital_species("up", s = -1)
        .add_orbital_species("down", s = 1)
        .add_orbital(position=(0, 0), species = "up",  tag = "up")
        .add_orbital(position=(0, 0), species = "down",  tag = "down")   
        .add_interaction(
            "hamiltonian",
            participants=("up", "up"),
            parameters=[0.0, t],
        )
        .add_interaction(
            "hamiltonian",
            participants=("down", "down"),
            parameters=[0.0, t],
        )
        .add_interaction(
            "coulomb",
            participants=("up", "down"),
            parameters=[U]
            )
    )

    return hubbard

def gs(flake):
    return flake.energies[:flake.electrons].sum()

data = []
gs_energies = []
Us = [0, 5.7, 8]

for U in Us:
    flake = get_hubbard(U).cut_flake( Rectangle(10, 4) )
    flake.set_open_shell()
    flake.set_electrons(len(flake)//2)
    print(flake.electrons)

    # flake.show_energies()
    flake.set_mean_field()
    # flake.show_energies()

    show_energies_with_inset_2d(flake, f"hubbard_energies_{U}.pdf")
    gs_energies.append(gs(flake))

    diff = flake.energies - flake.energies[:, None]
    m = jnp.abs(diff).max()
    omegas = jnp.linspace(0, m + 1, 20)
    
    correlator = flake.get_ip_green_function(flake.dipole_operator_e[0], flake.dipole_operator_e[0], omegas)

    occs = jnp.diagonal(flake.initial_density_matrix)
    spin_density = occs[:len(flake)//2] - occs[len(flake)//2:]    
    # show_2d(flake[:len(flake)//2], display = spin_density * flake.electrons )

    # check AF order
    print(((flake.electrons*spin_density).round(1) > 0).sum() == ((flake.electrons*spin_density).round(1) < 0).sum())

    data.append(correlator)

import matplotlib.pyplot as plt

# Style
plt.style.use('seaborn-v0_8-paper')

# Figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each curve with automatic distinct colors
for i, c in enumerate(data):
    absorption = -c.imag
    absorption /= absorption.max()  # normalize
    ax.plot(omegas, absorption, linewidth=2.0,
            label=rf"$\frac{{U}}{{t}} = {Us[i]}$")

# Labels & title
ax.set_xlabel(r'$\frac{\omega}{t}$ (eV)', fontsize=18)
ax.set_ylabel(r'Normalized Absorption', fontsize=18)
ax.set_title(r'Absorption Spectra for Different $\frac{U}{t}$ Values', fontsize=20, pad=15)

# Grid & ticks
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2)
ax.tick_params(axis='both', which='minor', length=3, width=1)

# Legend
ax.legend(fontsize=14, loc='best', frameon=True)

# Layout
plt.tight_layout()
plt.savefig("hubbard_phase_transition.pdf")
