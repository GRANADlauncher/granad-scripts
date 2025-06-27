import jax.numpy as jnp
from granad import *
from granad._plotting import *

def peierls_coupling(B, t, r1, r2):
    plus = r1 + r2
    minus = r1 - r2
    return complex(t * jnp.exp(1j * B * minus[0] * plus[1]))

def get_graphene_b_field(B, t, shape):
    """obtain tb model for graphene in magnetic field from peierls substitution

    Args:
        B : B-field strength
        t : nn hopping
    
    """
    graphene_peierls = (
        Material("graphene_peierls")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz", atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz")
        .add_interaction(
            "hamiltonian",
            participants=("pz", "pz"),
            parameters = [0],
        )
        .add_interaction(
            "coulomb",
            participants=("pz", "pz"),
            expression = ohno_potential(1.42)
        )
    )
    
    flake = graphene_peierls.cut_flake(shape)
    distances = jnp.round(jnp.linalg.norm(flake.positions - flake.positions[:, None], axis = -1), 4)
    nn = 1.5

    # this is slightly awkward: our coupling only depends on the distance vector, but peierls substitution in landau
    # gauge A = (0, Bx, 0) leads to peierls phase of \int_r1^r2 A dl = (y2 - y1) (x2 + x1) / 2
    # we patch this manually here, but this is very, very slow
    for i, orb1 in enumerate(flake):
        for j in range(i+1):
            if 0 < distances[i, j] <= nn:
                orb2 = flake[j]
                flake.set_hamiltonian_element(orb1, orb2, peierls_coupling(B, t, orb1.position, orb2.position))
    return flake

def get_absorption_rpa(flake):    
    # compute the polarizability in the RPA
    polarizability = flake.get_polarizability_rpa(
        omegas,
        relaxation_rate = 1/10,
        polarization = 0,
        hungry = 1 # higher numbers are faster and consume more RAM
    )
    absorption = polarizability.imag * 4 * jnp.pi * omegas

    return absorption

def sim():
    # boltzmann constant
    kb = 8.617 * 1e-5

    # beta at room temperature
    beta = 1 / (kb * 300)

    # magnetic field
    Bs = jnp.linspace(0, 10, 20)

    # frequency
    omegas = jnp.linspace(0, 1, 100)

    # results
    corr_map = []

    # structure
    shape = Hexagon(50)

    for B in Bs:
        flake = get_graphene_b_field(B, -2.7, shape)
        print(len(flake))
        flake.set_beta(beta)
        flake.show_2d(name = "flake.pdf")
        flake.show_energies(name = "ip_energies.pdf")
        corr = flake.get_ip_green_function(flake.dipole_operator_e[0], flake.dipole_operator_e[0], omegas)
        corr_map.append(corr)

    corr_map = jnp.array(corr_map).T  # Shape: (omegas, Bs)
    absorption_map = corr_map.imag * 4 * jnp.pi * omegas[:, None]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(absorption_map, aspect='auto', origin='lower',
               extent=[Bs[0], Bs[-1], omegas[0], omegas[-1]])
    plt.xlabel("Magnetic Field B")
    plt.ylabel(r"Photon Energy $\omega$")
    plt.title("Absorption Spectrum")
    plt.colorbar(label="Absorption")
    plt.tight_layout()
    plt.savefig("absorption_map.pdf")

sim()
