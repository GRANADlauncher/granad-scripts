import jax.numpy as jnp
import matplotlib.pyplot as plt

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

def plot_res(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.plot(omegas / omega_0, p)
    plt.savefig(f"dipole_moment_{name}.pdf")

def sim(flake, params):
    name, end_time, amplitudes, omega, peak, fwhm = params
    
    result = flake.master_equation(
        dt = 1e-5,
        end_time = end_time,
        relaxation_rate = 1/10,
        expectation_values = [ flake.dipole_operator ],
        illumination = Pulse(amplitudes, omega, peak, fwhm),
        grid = 100
    )
    result.save(name)

if __name__ == '__main__':    
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(40, armchair = True))
    flake.set_electrons(flake.electrons + 20)

    # name, end_time, amplitudes, omega, peak, fwhm = params
    omega_0 = 1.36
    params_list = [
        ["test", 40, [1, 0, 0], omega_0, 5, 2]
        ]
    
    sim(flake, params)
    
    plot_res(params[0], 4*omega_0, 0, omega_0)
