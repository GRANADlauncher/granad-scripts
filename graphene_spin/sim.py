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

shape = Triangle(30, armchair = False)
# shape = Rectangle(10, 10, armchair = False)
flake = graphene_spinful.cut_flake(shape) 
flake.set_open_shell()
flake.set_electrons(len(flake)//2)
flake.show_energies()
flake.set_mean_field( coulomb_strength = 0.5 * 1/6, mix = 0.001, iterations = 4000, accuracy = 1e-6)
flake.show_energies()

occs = jnp.diagonal(flake.initial_density_matrix)
spin_density = occs[:len(flake)//2] - occs[len(flake)//2:]    
show_2d(flake[:len(flake)//2], display = spin_density * flake.electrons )

