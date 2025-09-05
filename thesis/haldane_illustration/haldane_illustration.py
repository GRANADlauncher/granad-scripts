# relevant scales:
# length in nm, energies in eV, hbar = 1
import jax
import jax.numpy as jnp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, SymLogNorm

from granad import *

plt.style.use("../thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9  

### UTILITIES ###
def localization(flake):
    """Compute eigenstates edge localization"""
    # edges => neighboring unit cells are incomplete => all points that are not inside a "big hexagon" made up of nearest neighbors
    positions, states, energies = flake.positions, flake.eigenvectors, flake.energies 

    distances = jnp.round(jnp.linalg.norm(positions - positions[:, None], axis = -1), 4)
    nnn = jnp.unique(distances)[2]
    mask = (distances == nnn).sum(axis=0) < 6


    # localization => how much eingenstate 
    l = (jnp.abs(states[mask, :])**2).sum(axis = 0) # vectors are normed 

    return l

def get_haldane_graphene(t1, t2, delta):
    """Constructs a graphene model with onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2

    threshold is at $\\lambda > \\frac{\\delta}{3 \\sqrt{3}}$
    """
    
    return (
        Material("haldane_graphene")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz1", atom='C')
        .add_orbital_species("pz2", atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz1")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz2")
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz2"),
            parameters= [t1],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz1"),            
            # a bit overcomplicated
            parameters=[                
                [0, 0, 0, delta/2], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, t2], 
                [2.46, 0, 0, jnp.conj(t2)],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)]
            ],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz2", "pz2"),
            parameters=[                
                [0, 0, 0, -delta/2], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, jnp.conj(t2)], 
                [2.46, 0, 0, t2],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2]
            ],
        )
        .add_interaction(                
            "coulomb",
            participants=("pz1", "pz2"),
            parameters=[8.64],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz1", "pz1"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz2", "pz2"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
    )
def plot_combined():
    # Custom rcParams for consistency across both subplots
    fig, axes = plt.subplots(1, 2, figsize=(LATEX_TEXTWIDTH_IN, LATEX_TEXTWIDTH_IN * 0.45))

    # ---------- (a) Geometry plot ----------
    ax = axes[0]
    shape = Rhomboid(20, 20, armchair=False)
    orbs = get_haldane_graphene(1.0, 1j*0.3, 1.0).cut_flake(shape)
    l = localization(orbs)
    display = jnp.abs(orbs.eigenvectors[:, l.argsort()[-4]])**2

    colors = display / display.max()
    scatter = ax.scatter([orb.position[0] for orb in orbs],
                         [orb.position[1] for orb in orbs],
                         c=colors, edgecolor="none",
                         cmap="magma",
                         s=1e3 * display * 5)
    ax.scatter([orb.position[0] for orb in orbs],
               [orb.position[1] for orb in orbs],
               color="black", s=5, marker="o")
    fig.colorbar(scatter, ax=ax, label=r"$|\Psi|^2$")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.axis("equal")
    ax.set_title("(a)", loc="left", fontsize=26, fontweight="bold")

    # ---------- (b) Projected polarization ----------
    ax = axes[1]
    flake = get_haldane_graphene(1.0, 1j*0.2, 1.0).cut_flake(shape)
    v = flake.velocity_operator_e[:2]
    im = ax.matshow(jnp.abs(v[0])**2, cmap="plasma")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$n$")
    ax.set_title("(b)", loc="left", fontsize=26, fontweight="bold")

    plt.tight_layout()
    plt.savefig("haldane_illustration.pdf")
    # plt.show()

plot_combined()
