import matplotlib.pyplot as plt

import jax.numpy as jnp

from granad import *
from granad._numerics import bare_susceptibility_function

def plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm):
    time = jnp.linspace(0, end_time, 1000)
    pulse = Pulse(amplitudes, omega, peak, fwhm)
    e_field = jax.vmap(pulse) (time)
    plt.plot(time, e_field.real)
    # plt.plot(time, e_field.imag, '--')
    result = TDResult.load(name)        
    plt.plot(result.time_axis, result.output[0], '--')
    plt.savefig(f"t_dipole_moment_{name}.pdf")
    plt.close()

def plot_omega_dipole(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.semilogy(omegas / omega_0, p)
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()

flake = MaterialCatalog.get("graphene").cut_flake(Triangle(45, armchair = True))
flake.set_electrons(flake.electrons + 2)
flake.show_energies(name = "energies")

name, end_time, amplitudes, omega, peak, fwhm = "cox_50_1e-4_new", 700, [0.03, 0, 0], 0.68, 0.659 * 200, 0.659 * 166
    
result = flake.master_equation(
    dt = 1e-4,
    end_time = end_time,
    relaxation_rate = 1/10,
    expectation_values = [ flake.dipole_operator ],
    illumination = Pulse(amplitudes, omega, peak, fwhm),
    max_mem_gb = 50,
    grid = 100
)
result.save(name)        
plot_omega_dipole(name, 6*omega, 0, omega)
plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm)
