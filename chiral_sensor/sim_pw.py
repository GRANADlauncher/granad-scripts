from itertools import combinations

import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *

# chiral potential
def ChiralPulse(
    omega: float,
    peak: float,
    fwhm: float,
    phi : float,
):
    """Function for computing temporally located time-harmonics electric fields. The pulse is implemented as a temporal Gaussian.

    Args:
        omega: frequency of the electric field
        peak: time where the pulse reaches its peak
        fwhm: full width at half maximum
        phi : relative rotation angle

    Returns:
       Function that computes the electric field
    """

    sigma = fwhm / (2.0 * jnp.sqrt(jnp.log(2)))
    return lambda t: 1e-4 * (
        jnp.array([jnp.cos(omega * t), jnp.cos(omega * t - phi), 0.])
        * jnp.exp(-((t - peak) ** 2) / sigma**2)
    )

def get_haldane_graphene(t1, t2, delta):
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

def td_sim(shape,
           t1,
           t2,
           delta,
           end_time,
           omega,
           peak,
           fwhm,
           phi,
           name):
    """saves the induced field during the time evolution of haldane graphene kicked by a spectrally 
    narrow dipole at the dipole position
    """
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    hamiltonian = flake.get_hamiltonian()
    result = flake.master_equation(
        dt = 1e-5,
        end_time = end_time,
        relaxation_rate = 1/10,
        expectation_values = [ flake.dipole_operator ],
        illumination = ChiralPulse(omega, peak, fwhm, phi),
        grid = 100
    )
    result.save(name)
    return result


### postprocessing ###
def plot_edge_states(shape,
                     t1,
                     t2,
                     delta,
                     end_time,
                     omega,
                     peak,
                     fwhm,
                     phi,
                     name):
    """saves a pdf plot of the edge states in a cut of haldane graphene.
    returns the constructed flake
    """

    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
        
    # edge states    
    for i, idx in enumerate(jnp.argwhere(jnp.abs(flake.energies) < 1e-1)):
        flake.show_2d( display = flake.eigenvectors[:, idx.item()], scale = True, name = f"{name}_{i}_es.pdf" )

    return flake

def plot_energies(shape,
                  t1,
                  t2,
                  delta,
                  end_time,
                  omega,
                  peak,
                  fwhm,
                  phi,
                  name):
    """saves a pdf plot of the energy landscape and edge states in a cut of haldane graphene.
    returns the constructed flake
    """

    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    
    # energies
    flake.show_energies(name = f"{name}.pdf")    

    return flake

def get_abs(arg, omega_max, omega_min):
    """returns absorption and omega axis
    """
    
    result = TDResult.load(arg[-1])
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
    alpha = p_omega[:, :, None] / pulse_omega[:, None, :]
    absorption_td = jnp.abs( -omegas_td[:, None, None] * jnp.imag(alpha) )
    return omegas_td, absorption_td

def plot_absorption_components(args_list, omega_max, omega_min, name = "absorption_components_pw_pulse.pdf"):
    fig, axs = plt.subplots(2, 2)
    for arg in args_list:
        omegas_td, absorption_td = get_abs(arg, omega_max, omega_min)

        axs[0,0].set_title('xx')
        axs[0,0].plot(omegas_td, absorption_td[:, 0, 0], label = f'{arg[-1]}')
        axs[0,0].legend()
        
        axs[0,1].set_title('yy')
        axs[0,1].plot(omegas_td, absorption_td[:, 1, 1], label = f'{arg[-1]}')
        axs[0,1].legend()

        axs[1,0].set_title('xy')
        axs[1,0].plot(omegas_td, absorption_td[:, 1, 0], label = f'{arg[-1]}')
        axs[1,0].legend()
        
        axs[1,1].set_title('yx')
        axs[1,1].plot(omegas_td, absorption_td[:, 0, 1], label = f'{arg[-1]}')
        axs[1,1].legend()

    plt.savefig(name)
    plt.close()

def plot_absorption(args_list, omega_max, omega_min, name = "total_absorption_pw_pulse.pdf"):
    
    for arg in args_list:
        omegas_td, absorption_td = get_abs(arg, omega_max, omega_min)
        plt.plot(omegas_td, absorption_td[:, 1, 1], label = f'{arg[-1]}')
        # plt.semilogy(omegas_td, jnp.trace(absorption_td[:, :2, :2], axis1=1, axis2=2), label = f'{arg[-1]}')
        
    plt.legend()
    plt.savefig(name)
    plt.close()
    
def plot_td(args_list, name = "dipole_oscillations_pw_pulse.pdf"):
    for arg in args_list:
        result = TDResult.load(arg[-1])        
        plt.plot(result.time_axis, result.output[0])
    plt.legend()
    plt.savefig(name)
    plt.close()
        
def plot_cd(args_list, omega_max, omega_min, name = "cd_plane_wave_pulse.pdf"):
    
    for args_tuple in combinations(args_list, 2):
        # b field unequal or no opposite phi
        if args_tuple[0][2] != args_tuple[1][2] or args_tuple[0][-2] != -args_tuple[1][-2]:
            continue        
        print(args_tuple[0][-1], args_tuple[1][-1])
        
        omegas_td, abs_r = get_abs(args_tuple[0], omega_max, omega_min)
        omegas_td, abs_l = get_abs(args_tuple[1], omega_max, omega_min)
        
        plt.plot(omegas_td, jnp.trace((abs_r - abs_l)[:, :2, :2], axis1=1, axis2=2), label = rf'$\Delta$ {args_tuple[0][-1]}')                
    plt.legend()
    plt.savefig(name)
    plt.close()

if __name__ == '__main__':
    # haldane model has topological phase for Im[t2] > \frac{M}{3 \sqrt{3}} => for 0.3 Im[t_2]_crit ~ 0.06
    # (shape, t1, t2, delta (mass term), end_time, omega, peak, fwhm, phi, name)
    args_list = [
        (Hexagon(20, armchair = True), 1.0, -1j*t2, 0.3, 40, 0.3, 5, 2, phi, f"graphene_{t2}_{phi}" )
        for t2 in [0., 0.5, 1.] for phi in jnp.linspace(-jnp.pi, jnp.pi, 30)
        ]
    args_list = [
        (Hexagon(30, armchair = True), -2.66, -1j*t2, 0.3, 40, 0.3, 5, 2, phi, f"graphene_two_phases{t2}_{phi}" )
        for t2 in [0.04, 0.1] for phi in [-jnp.pi/2, 0, jnp.pi/2]
        ]

    # import pdb; pdb.set_trace()
    flake = plot_energies(*args_list[-1])    
    flake = plot_edge_states(*args_list[-1])    

    run, plot = 0, 0
    if run:
        for arg in args_list:
            td_sim(*arg)

    if plot:
        plot_absorption(args_list, 40, 0)
        plot_td(args_list)
    
    # plot_cd(args_list, 40, 0)
    # flake = static_sim(Hexagon(31, armchair = True), 1.0, 1j*0.1, 0.3, 40, 2.6, 5, 2, "ff", f"g2_no_coul_pw")    
    # result = td_sim(*args_list[1])
    # plot_chiral_ldos( [args_list[1]], 5, 0 )

    # shape, t1, t2, delta = Triangle(20), -2.66, -1j, 0.0
    # jnp.abs(MaterialCatalog.get("graphene").cut_flake(shape).coulomb - get_haldane_graphene(t1, t2, delta).cut_flake(shape).coulomb).max()
