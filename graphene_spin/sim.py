# hubbard u like in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.101.235427

import jax.numpy as jnp
from granad import *
from granad._plotting import *

# define spinful graphene
graphene_spinful = (
    Material("graphene_spinful")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0]
    ])
    .add_orbital_species("pz+", atom='C')
    .add_orbital_species("pz-", atom='C')
    .add_orbital(position=(0, 0), tag="sublattice_1", species="pz+")
    .add_orbital(position=(0, 0), tag="sublattice_1", species="pz-")
    .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz+")
    .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz-")
    .add_interaction(
        "hamiltonian",
        participants=("pz+", "pz+"),
        parameters=[0.0, -2.66],
    )
    .add_interaction(
        "hamiltonian",
        participants=("pz-", "pz-"),
        parameters=[0.0, -2.66],
    )
    .add_interaction(
        "coulomb",
        participants=("pz+", "pz+"),
        parameters=[0, 8.64, 5.333],
        expression=lambda r : 1/r + 0j
    )
    .add_interaction(
        "coulomb",
        participants=("pz-", "pz-"),
        parameters=[0, 8.64, 5.333],
        expression=lambda r : 1/r + 0j
    )
    .add_interaction(
        "coulomb",
        participants=("pz+", "pz-"),
        parameters=[16.522, 8.64, 5.333],
        expression=lambda r : 1/r + 0j
    )
)

U = 1
graphene_spinful_hubbard = (
    Material("graphene_spinful_hubbard")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0]
    ])
    .add_orbital_species("pz+", atom='C')
    .add_orbital_species("pz-", atom='C')
    .add_orbital(position=(0, 0), tag="sublattice_1", species="pz+")
    .add_orbital(position=(0, 0), tag="sublattice_1", species="pz-")
    .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz+")
    .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz-")
    .add_interaction(
        "hamiltonian",
        participants=("pz+", "pz+"),
        parameters=[0.0, -2.7],
    )
    .add_interaction(
        "hamiltonian",
        participants=("pz-", "pz-"),
        parameters=[0.0, -2.7],
    )
    .add_interaction(
        "coulomb",
        participants=("pz+", "pz-"),
        parameters=[U],
    )
)

def sublattice_diff(flake):
    return abs(len([o for o in flake if o.tag == "sublattice_2"]) - len([o for o in flake if o.tag == "sublattice_1"]))


dop = 2
for size in [20]:

    shape = Triangle(size, armchair = False)
    # shape = Rectangle(10, 10, armchair = False)
    flake = graphene_spinful_hubbard.cut_flake(shape) 
    flake.set_open_shell()
    flake.set_electrons(len(flake)//2 + dop)
    flake.show_energies()
    flake.set_mean_field( coulomb_strength = 1, mix = 0.001, iterations = 1000, accuracy = 1e-6)
    flake.show_energies()

    occs = jnp.diagonal(flake.initial_density_matrix)
    spin_density = occs[:len(flake)//2] - occs[len(flake)//2:]
    show_2d(flake[:len(flake)//2], display = spin_density * flake.electrons )


    total_spin = spin_density.sum()

    print(sublattice_diff(flake)/2, total_spin)
