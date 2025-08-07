import jax.numpy as jnp
from granad import *
from granad._plotting import *

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

data = []
Us = [5, 6, 7, 8]

for U in Us:
    flake = get_hubbard(U).cut_flake( Rectangle(10, 4) )
    flake.set_open_shell()
    flake.set_electrons(len(flake)//2)
    print(flake.electrons)

    # flake.show_energies()
    flake.set_mean_field()
    # flake.show_energies()

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

for i, c in enumerate(data):
    absorption = -c.imag
    absorption /= absorption.max()
    plt.plot(omegas, absorption, label = f"{Us[i]}")

plt.legend()
plt.show()
