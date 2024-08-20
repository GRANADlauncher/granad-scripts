import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad import *

def get_haldane_graphene(t1, t2, delta1, delta2):
    """Constructs a graphene model with
    onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2
    """
    return (
        Material("haldane_graphene")
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
            parameters= [t1],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz1"),            
            # a bit overcomplicated
            parameters=[                
                [0, 0, 0, delta1], # onsite                
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
                [0, 0, 0, delta2], # onsite                
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

def get_bilayer_graphene(shape,
                         delta1,
                         delta2,
                         g11,
                         g12,
                         g21,
                         g22,
                         inter_nn,
                         inter_nnn
                         ):
    """AB stack of two graphene layers
    
    Parameters:
        shape : granad shape array
        delta1, delta2 : dimer SLS breaking mass terms, SWC: delta1 = delta2 = 0.018
        g11, g21 : intralayer nn hoppings, SWC: 3.00
        g12, g22 : intralayer nnn hoppings (taken as Haldane couplings), SCW: 0.0
        inter_nn : interlayer nn coupling, SWC: 0.38
        inter_nnn : interlayer nnn coupling, SWC: 0.1

    Returns:
        flake
    """
    
    def interlayer_coupling(v):
        index = jnp.abs(jnp.linalg.norm(v) - distances).argmin()
        return jax.lax.switch(index, [lambda x : inter_nn + 0j, lambda x : inter_nnn + 0j, lambda x : 0j], index)
        
    # TODO: rotate for armchair
    shift = jnp.array( [0, 1.42, 3.35] )
    n1 = jnp.abs(shift[2]) 
    n2 = jnp.linalg.norm(shift)
    cutoff = n2 + 0.01
    distances = jnp.array( [n1, n2, cutoff] )    
    
    m1, m2 = get_haldane_graphene(g11, g12, delta1, 0), get_haldane_graphene(g21, g22, 0, delta2)    
    flake1, flake2 = m1.cut_flake(shape), m2.cut_flake(shape)
    
    flake2.shift_by_vector(shift)    
    flake = flake1 + flake2
    
    flake.set_hamiltonian_groups(flake1, flake2, interlayer_coupling)

    return flake


def get_dip(name, omega_max, omega_min, omega_0):
    """returns induced dipole moment, normalized to its value at omega_0
    and omega axis.
    """
    
    result = TDResult.load(name)
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas, _ = result.ft_illumination( omega_max, omega_min )
    closest_index = jnp.argmin(jnp.abs(omegas - omega_0))
    p_0 = 1.0#jnp.linalg.norm(p_omega[closest_index])
    p_normalized = jnp.linalg.norm(p_omega, axis = -1) / p_0
    return omegas, p_normalized

def plot_omega_dipole(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.semilogy(omegas / omega_0, p)
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()

def plot_omega_abs(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.semilogy(omegas / omega_0, p)
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()


def plot_t_dipole(name, params):
    shape, delta1, delta2, g11, g12, g21, g22, inter_nn, inter_nnn, end_time, amplitudes, omega, peak, fwhm = params
    time = jnp.linspace(0, end_time, 1000)
    pulse = Pulse(amplitudes, omega, peak, fwhm)
    e_field = jax.vmap(pulse) (time)
    plt.plot(time, e_field.real)
    # plt.plot(time, e_field.imag, '--')
    result = TDResult.load(name)        
    plt.plot(result.time_axis, result.output[0], '--')
    plt.savefig(f"t_dipole_moment_{name}.pdf")
    plt.close()
    
def get_abs(name, omega_max, omega_min):
    """returns absorption and omega axis
    """
    
    result = TDResult.load(name)
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
    alpha = p_omega[:, :, None] / pulse_omega[:, None, :]
    absorption_td = jnp.abs( -omegas_td[:, None, None] * jnp.imag(alpha) )
    return omegas_td, absorption_td[:, 0, 0]

def plot_absorption(name, omega_max, omega_min, rpa = False):    
    omegas, absorption = get_abs(name, omega_max, omega_min)
    plt.plot(omegas, absorption / absorption.max())
    if rpa:
        with jnp.load(f'rpa_{name}.npz') as data:
            omegas_rpa = data["omegas"]
            absorption_rpa = jnp.abs( data["pol"].imag * 4 * jnp.pi * omegas_rpa )
            plt.plot(omegas_rpa, absorption_rpa / absorption_rpa.max(), '--')
    plt.savefig(f'absorption_{name}')
    plt.close()

def sim(flake, params, name):
    shape, delta1, delta2, g11, g12, g21, g22, inter_nn, inter_nnn, end_time, amplitudes, omega, peak, fwhm = params
    
    result = flake.master_equation(
        dt = 1e-4,
        end_time = end_time,
        relaxation_rate = 1/10,
        expectation_values = [ flake.dipole_operator ],
        illumination = Pulse(amplitudes, omega, peak, fwhm),
        # stepsize_controller = diffrax.ConstantStepSize(),
        # solver = diffrax.Dopri8(),
        # max_mem_gb = 50,
        grid = 100
    )
    result.save(name)

def setup(params):
    """plots energies of single and bilayer"""
    shape, delta1, delta2, g11, g12, g21, g22, inter_nn, inter_nnn, end_time, amplitudes, omega, peak, fwhm = params    
    m1, m2 = get_haldane_graphene(g11, g12, delta1, 0), get_haldane_graphene(g21, g22, 0, delta2)    
    flake1, flake2 = m1.cut_flake(shape), m2.cut_flake(shape)
    
    flake1.show_energies(name="1.pdf")
    print(len(flake1))
    flake2.show_energies(name="2.pdf")
    print(len(flake2))
    flake = get_bilayer_graphene(*params[:9])
    flake.show_energies(name="1+2.pdf")
    flake.show_2d(name = "geometry.pdf")

    return flake1, flake2, flake
           
    
if __name__ == '__main__':    
    # shape, delta1, delta2, g11, g12, g21, g22, inter_nn, inter_nnn, end_time, amplitudes, omega, peak, fwhm = params
    omega_0 = 4
    single, bi = "linear_response_single", "linear_response_bilayer"
    params = [Triangle(30), 0.0, 0.0, 3.0, 0., 3.0, 0., 0.38, 0.1, 100, [1e-5, 0, 0], omega_0, 4, 0.5]

    # plot geometry, combined and separate energies, return entire and parts of the flake
    flake1, flake2, flake =  setup(params)

    # time domain simulation of single layer
    sim(flake1, params, single)

    # time domain simulation of bilayer
    sim(flake, params, bi)

    # plot results
    for name in [single, bi]:
        plot_omega_dipole(name, 6*omega_0, 0, omega_0)
        plot_t_dipole(name, params)
        plot_absorption(name, 10, 0)
