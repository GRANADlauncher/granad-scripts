import matplotlib.pyplot as plt

import jax.numpy as jnp

from granad import *
from granad._numerics import bare_susceptibility_function
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-paper')

def plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm):
    time = jnp.linspace(0, end_time, 1000)
    pulse = Pulse(amplitudes, omega, peak, fwhm)
    e_field = jax.vmap(pulse)(time)
    
    result = TDResult.load(name)        

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, e_field.real, linewidth=2.0, label='Driving Field (Re)')
    # ax.plot(time, e_field.imag, '--', linewidth=2.0, label='Driving Field (Im)')
    ax.plot(result.time_axis, result.output[0], '--', linewidth=2.0, label='Dipole Response')

    ax.set_xlabel('Time (a.u.)', fontsize=18)
    ax.set_ylabel('Amplitude (a.u.)', fontsize=18)
    ax.set_title('Time-Domain Dipole Moment', fontsize=20, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(f"t_dipole_moment_{name}.pdf")
    plt.close()

def plot_omega_dipole(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(omegas / omega_0, p, linewidth=2.0, color='C0')

    ax.set_xlabel(r'$\omega / \omega_0$', fontsize=18)
    ax.set_ylabel('Dipole Strength (a.u.)', fontsize=18)
    ax.set_title('Frequency-Domain Dipole Moment', fontsize=20, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()

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
