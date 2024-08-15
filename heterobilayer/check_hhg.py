import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax

from granad import *

def get_dip(name, omega_max, omega_min, omega_0):
    """returns induced dipole moment, normalized to its value at omega_0
    and omega axis.
    """
    
    result = TDResult.load(name)
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas, _ = result.ft_illumination( omega_max, omega_min )
    closest_index = jnp.argmin(jnp.abs(omegas - omega_0))
    p_0 = jnp.linalg.norm(p_omega[closest_index])
    p_normalized = jnp.linalg.norm(p_omega, axis = -1) / p_0
    return omegas, p_normalized

def plot_omega_dipole(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.plot(omegas / omega_0, p)
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()

def plot_t_dipole(params):
    name, end_time, amplitudes, omega, peak, fwhm = params
    time = jnp.linspace(0, end_time, 1000)
    pulse = Pulse(amplitudes, omega, peak, fwhm)
    e_field = jax.vmap(pulse) (time)
    plt.plot(time, e_field.real)
    # plt.plot(time, e_field.imag, '--')
    result = TDResult.load(name)        
    plt.plot(result.time_axis, result.output[0], '--')
    plt.savefig(f"t_dipole_moment_{name}.pdf")
    plt.close()
    
def sim(flake, params):
    name, end_time, amplitudes, omega, peak, fwhm = params
    
    result = flake.master_equation(
        dt = 1e-5,
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

def sim_rpa(flake, name, omega_max, omega_min):        
    omegas = jnp.linspace( omega_max, omega_min, 40 )
    polarizability = flake.get_polarizability_rpa(
        omegas,
        relaxation_rate = 1/10,
        polarization = 0, 
        hungry = 1)
    jnp.savez(f"rpa_{name}.npz", **{"pol" : polarizability, "omegas" : omegas}  )

    
if __name__ == '__main__':
    # params: Q = 2, N = 330 armchair, light : frequency = 0.68 eV, fwhm = 166fs, pol perp to triangle side, duration: 700, peak at 200
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(45, armchair = True))
    flake.set_electrons(flake.electrons + 2)
    flake.show_energies(name = "energies")

    # name, end_time, amplitudes, omega, peak, fwhm = params
    omega_0 = 0.68
    params = ["cox", 700, [0.03, 0, 0], omega_0, 0.659 * 200, 0.659 * 166]    

    sim(flake, params)
    
    # sim_rpa(flake, params[0], 4, 0)
    # plot_absorption(params[0], 4, 0, rpa = True)    
    # plot_absorption(params[0], 4, 0)
    
    plot_omega_dipole(params[0], 4*omega_0, 0, omega_0)
    plot_t_dipole(params)
