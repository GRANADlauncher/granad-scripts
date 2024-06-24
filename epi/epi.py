import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
from granad import *

def get_epi(flake, omega):
    """computes the epi from a full time domain simulation under plane wave illumination"""
    res = flake.master_equation(
        relaxation_rate = 1/10,
        illumination = Wave(frequency = omega / 2*jnp.pi, amplitudes = [1e-5, 0, 0]),
        end_time = 80)
    return flake.get_epi(res.final_density_matrix, f, 0.05)

def plot_rho(flake, omega):
    """plots the quantity rho / |\Delta E - w|"""
    res = flake.master_equation(
        relaxation_rate = 1/10,
        illumination = Wave(frequency = omega / 2*jnp.pi, amplitudes = [1e-5, 0, 0]),
        end_time = 80)
    rho = res.final_density_matrix

def plot_absorption(
    orbs,
    res,
    plot_only : jax.Array = None,
    plot_labels : list[str] = None,
    show_illumination = False,
    omega_max = None,
    omega_min = None,
):
    """Depicts an expectation value as a function of time.


    - `res`: result object
    - `plot_only`: only these indices will be plotted
    - `plot_legend`: names associated to the indexed quantities
    """
    def _show( obs, name ):
        ax.plot(x_axis, obs, label = name)
    
    fig, ax = plt.subplots(1, 1)    
    ax.set_xlabel(r"time [$\hbar$/eV]")
    plot_obs = res.output
    illu = res.td_illumination
    x_axis = res.time_axis
    cart_list = ["x", "y", "z"]
    
    
    if omega_max is not None and omega_min is not None:
        plot_obs = res.ft_output( omega_max, omega_min )
        x_axis, illu = res.ft_illumination( omega_max, omega_min )
        ax.set_xlabel(r"$\omega$ [$\hbar$ eV]")
    
    for obs in plot_obs:
        obs = obs if plot_only is None else obs[:, plot_only]
        for i, obs_flat in enumerate(obs.T):
            label = '' if plot_labels is None else plot_labels[i]
            _show( obs_flat, label )
        if show_illumination == True:
            for component, illu_flat in enumerate(illu.T):            
                _show(illu_flat, f'illumination_{cart_list[component]}')
            
    plt.legend()


if __name__ == '__main__':

    # flake
    n, m = 20, 20
    flake = MaterialCatalog.get("graphene").cut_flake( Rhomboid(n,m))

    # illumination
    pulse = Pulse(
        amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
    )

    # absorption run
    result = flake.master_equation(
        relaxation_rate = 1/10,
        illumination = pulse,
        dt = 1e-5,
        expectation_values = [flake.dipole_operator],
        end_time = 100)
    flake.show_res(result)

    # postprocessing
    omega_min, omega_max = 0, 5
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas, pulse_omega = result.ft_illumination( omega_max, omega_min )
    absorption = jnp.abs( -omegas * jnp.imag( p_omega[:,0] / pulse_omega[:,0] ) )

    # find peaks
    peaks = omegas[scipy.signal.find_peaks(absorption)[0]]

    # find epi for each peak 
    for p in peaks:
        epi = get_epi(flake, p)

        # black if sp, red if plasmonic
        plt.axvline(p, ls = '--')

    # pick a random peak and plot the matrix
    plot_rho(v)
