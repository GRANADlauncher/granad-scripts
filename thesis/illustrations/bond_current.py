import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import get_cmap

from granad import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
plt.style.use("../thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9

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

def combined_plot():
    fig, axes = plt.subplots(1, 2, figsize=(LATEX_TEXTWIDTH_IN, LATEX_TEXTWIDTH_IN * 0.45), constrained_layout=True)

    # --------- (a) Haldane sketch ---------
    ax = axes[0]

    # reuse your plot_haldane_sketch code but direct output to `ax`
    rt3 = np.sqrt(3.0)
    a1 = np.array([rt3, 0.0])
    a2 = np.array([rt3/2.0, 3.0/2.0])
    delta1 = np.array([0.0, 1.0])
    rng = range(-2, 3)
    A, B = {}, {}
    for n in rng:
        for m in rng:
            rA = n * a1 + m * a2
            rB = rA + delta1
            A[(n, m)] = rA
            B[(n, m)] = rB
    A_coords = np.array(list(A.values()))
    B_coords = np.array(list(B.values()))

    for (n, m), rA in A.items():
        for key in [(n, m), (n, m-1), (n+1, m-1)]:
            if key in B:
                rB = B[key]
                ax.plot([rA[0], rB[0]], [rA[1], rB[1]], linewidth=1.0, alpha=0.35)
    ax.scatter(A_coords[:, 0], A_coords[:, 1], s=55, label="A sublattice")
    ax.scatter(B_coords[:, 0], B_coords[:, 1], s=55, marker="s", label="B sublattice")

    r0 = A[(0, 0)]
    r1 = A[(1, 0)]
    ax.annotate("", xy=r1, xytext=r0,
                arrowprops=dict(arrowstyle="->", lw=2.0))
    mid = 0.5 * (r0 + r1)
    ax.text(mid[0] + 0.12, mid[1] + 0.12, "NNN hop", fontsize=11)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("(a) Haldane model sketch")

    # --------- (b) Geometry plot ---------
    ax = axes[1]
    shape = Rhomboid(40, 40, armchair=False)
    orbs = get_haldane_graphene(1.0, 1j*0.3, 1.0).cut_flake(shape)
    l = localization(orbs)
    display = jnp.abs(orbs.eigenvectors[:, l.argsort()[-4]])**2

    colors = display 
    scatter = ax.scatter([orb.position[0] for orb in orbs],
                         [orb.position[1] for orb in orbs],
                         c=colors,
                         edgecolor='none',
                         cmap='magma',
                         s=display*5)
    ax.scatter([orb.position[0] for orb in orbs],
               [orb.position[1] for orb in orbs],
               color='black', s=5, marker='o')

    fig.colorbar(scatter, ax=ax, label=r'$|\Psi|^2$')
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.axis("equal")
    ax.set_title("(b) Eigenstate localization")

    plt.savefig("combined.pdf")
    plt.show()

combined_plot()
