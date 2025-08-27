import jax.numpy as jnp
from granad import *
from granad._plotting import *

def get_peierls_coupling(B, t):

    def inner(r1, r2):        
        d = jnp.linalg.norm(r1 - r2)
        return jax.lax.cond(jnp.abs(d - 1.42) < 1e-2,
                            lambda x : t * jnp.exp(1j * B * (r1-r2)[0] * (r1+r2)[1]),
                            lambda x : 0j,
                            d
                            )

    return inner

def get_graphene_b_field(B, t, shape):
    """obtain tb model for graphene in magnetic field from peierls substitution

    Args:
        B : B-field strength
        t : nn hopping
    
    """
    flake = get_graphene().cut_flake(shape)
    flake.set_hamiltonian_groups(flake[0], flake[0], get_peierls_coupling(B, t))
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
    kb = 1 # 8.617 * 1e-5

    # beta at room temperature
    beta = 1 / (kb * 300)

    # magnetic field / scaled by e * (1e-4 eV)**2
    Bs = jnp.linspace(0, 10, 20)  / 33
    Bs = [1/3]

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
        flake.show_energies(name = "ip_energies.pdf", e_max = 0.2, e_min = -0.2)
        import pdb; pdb.set_trace()

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
