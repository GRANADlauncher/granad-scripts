import jax.numpy as jnp
from granad import *
from granad._plotting import *

import jax.numpy as jnp
import matplotlib.pyplot as plt

plt.style.use("../thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9

# --- Your spin-pol plot, but always draw on a provided axis if given ---
def plot_spin_polarization(orbs, show_tags=None, circle_scale: float = 1e3, ax=None):
    occs = jnp.diag(orbs.initial_density_matrix) * orbs.electrons
    diff = occs[:len(orbs)//2] - occs[len(orbs)//2:]

    positions = orbs.positions[:(len(orbs)//2)]
    sc = ax.scatter(positions[:, 0], positions[:, 1], c = diff, cmap = "magma", s = 1e1)

    # Add colorbar on the right
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label("Spin Polarization")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return ax

# define spinful graphene
def get_graphene_spinful(U):
    return (
        Material("graphene_spinful")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz+", atom='C')
        .add_orbital_species("pz-", atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1+", species="pz+")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2+", species="pz+")
        .add_orbital(position=(0, 0), tag="sublattice_1-", species="pz-")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2-", species="pz-")
        .add_interaction(
            "hamiltonian",
            participants=("pz+", "pz+"),
            parameters=[0.0, 2.55],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz-", "pz-"),
            parameters=[0.0, 2.55],
        )
        .add_interaction(
            "coulomb",
            participants=("pz+", "pz-"),
            parameters=[U],
        )
    )





fig, axs = plt.subplots(1, 2, figsize=(LATEX_TEXTWIDTH_IN, LATEX_TEXTWIDTH_IN * 0.45))


shapes = ["triangle", "rhomboid"]
labels = ["(a)", "(b)"]
for i, shape in enumerate([Triangle(24, armchair = False), Rhomboid(30, 30, armchair = False)]):
    flake = get_graphene_spinful(U = 4).cut_flake(shape)
    print(len(flake))
    flake.set_open_shell()
    flake.set_electrons(len(flake)//2)

    rho_0 = jnp.ones(len(flake)) 
    rho_0 = jnp.diag(rho_0).astype(complex) 
    flake.set_mean_field(mix = 0.4, iterations = 200, rho_0 = rho_0)

    plot_spin_polarization(flake, ax=axs[i])
    axs[i].text(-0.1, 1.2, labels[i], transform=axs[i].transAxes, va="top", ha="left", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("hubbard_graphene.pdf")
