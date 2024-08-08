from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *

def get_haldane_graphene(t1, t2, delta = 0.2):
    """Constructs a graphene model with
    onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2
    """
    return (Material("graphene")
            .lattice_constant(2.46)
            .lattice_basis([
                [1, 0, 0],
                [-0.5, jnp.sqrt(3)/2, 0]
            ])
            .add_orbital_species("pz1", l=1, atom='C')
            .add_orbital_species("pz2", l=1, atom='C')
            .add_orbital(position=(0, 0), tag="sublattice_1", species="pz1")
            .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz2")
            .add_interaction(
                "hamiltonian",
                participants=("pz1", "pz2"),
                parameters=[t1],
            )
            .add_interaction(
                "hamiltonian",
                participants=("pz1", "pz1"),
                parameters=[delta, t2],
            )
            .add_interaction(
                "hamiltonian",
                participants=("pz2", "pz2"),
                parameters=[0.0, t2],
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

def field_at(source_positions, target_position):
    """returns a postprocessing function to save the field at the specified positions"""
    def inner(rhos, args):        
        return jnp.einsum('Tii,iC->TC',
                          args.electrons*(args.stationary_density_matrix - rhos),
                          propagator)
    
    # N x 3
    distance_vector = target_position[None, :] - source_positions

    # N
    norms = jnp.linalg.norm(distance_vector, axis=-1)

    # N
    one_over_distance_cubed = jnp.where(norms > 0, 1 / norms**3, 0)
    
    # N x 3
    propagator = distance_vector * one_over_distance_cubed[:, None]
    
    return inner

# TODO : test
def static_sim(shape,
        t1,
        t2,
        delta,
        end_time,
        dipole_moment,
        source_location,
        omega,
        sigma,
        t0,
        name):
    """saves a pdf plot of the energy landscape and edge states in a cut of haldane graphene.
    returns the constructed flake
    """

    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    
    # energies
    flake.show_energies(name = f"{name}.pdf")

    # edge state
    idx = jnp.argwhere(jnp.abs(flake.energies) < 1e-1)[0].item()
    flake.show_2d( display = flake.eigenvectors[:, idx], scale = True, name = f"{name}_es.pdf" )

    return flake

def td_sim(shape,
           t1,
           t2,
           delta,
           end_time,
           dipole_moment,
           source_location,
           omega,
           sigma,
           t0,
           name):
    """saves the induced field during the time evolution of haldane graphene kicked by a spectrally 
    narrow dipole at the dipole position
    """
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    hamiltonian = flake.get_hamiltonian()
    hamiltonian["dipole"] = DipolePulse(dipole_moment = dipole_moment,
                                        source_location = source_location,
                                        omega = omega,
                                        sigma = sigma,
                                        t0 = t0)    
    result = flake.master_equation(
        end_time = end_time,
        hamiltonian = hamiltonian,
        relaxation_rate = 1/10,
        postprocesses = { "field" : field_at(flake.positions, jnp.array(source_location).astype(float)) }
    )
    result.save(name)
    return result

def chiral_ldos(name, omega_max, omega_min):
    """returns tuple containing (FFT(load(result)) * e_r, FFT(load(result)) * e_l)"""
    e_l, e_r = jnp.array([1., 1j, 0]), jnp.array([1., -1j, 0])
    result = TDResult.load(name)
    omegas, _ = result.ft_illumination(omega_max = omega_max, omega_min = omega_min)
    e_field = result.ft_output(omega_max = omega_max, omega_min = omega_min)[0]
    return omegas, e_field @ e_l, e_field @ e_r


def plot_chiral_ldos(args_list, omega_max, omega_min):
    for arg in args_list:
        omegas, left, right = chiral_ldos(arg[-1], omega_max, omega_min)
        plt.plot(omegas, left.imag)
        plt.plot(omegas, right.imag, '--')
    plt.savefig("res.pdf")
    plt.close()

if __name__ == '__main__':
    # (shape, t1, t2, delta, end_time, dipole_moment, source_location, omega, sigma, t0, postprocess, name)
    args_list = [
        (Hexagon(20, armchair = True), 1.0, 1j*t2, 0.2, 40, [0, 0, 1.0], [0, 0, 1.0], 2.3, 0.5 / 2.355, 0.5, f"{t2}" )
        for t2 in jnp.linspace(0, 0.1, 10)
        ]
    for arg in args_list:
        td_sim(arg)
    plot_chiral_ldos(args_list, 5, 0)
    
    # flake = static_sim(*args_list[1])
    # result = td_sim(*args_list[1])
    # plot_chiral_ldos( [args_list[1]], 5, 0 )
