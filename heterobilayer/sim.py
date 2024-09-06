import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad import *
from lib import *

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
