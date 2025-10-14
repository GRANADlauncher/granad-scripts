import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from granad import *

hbn = MaterialCatalog.get("hBN")

flake = hbn.cut_flake( Triangle(18)  )
print(len(flake))

omegas_rpa = jnp.linspace( 0, 3, 100 )

polarizability = flake.get_polarizability_rpa(
    omegas_rpa,
    relaxation_rate = 1/10,
    polarization = 0,
    hungry = 1 # higher numbers are faster and consume more RAM
)

absorption_rpa = polarizability.imag * 4 * jnp.pi * omegas_rpa


pulse = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)

result = flake.master_equation(
    expectation_values = [ flake.dipole_operator ],
    end_time=40,
    relaxation_rate=1/10,
    illumination=pulse,
)
omega_max = omegas_rpa.max()
omega_min = omegas_rpa.min()
p_omega = result.ft_output( omega_max, omega_min )[0]
omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
absorption_td = jnp.abs( -omegas_td * jnp.imag( p_omega[:,0] / pulse_omega[:,0] )  )

# Save in a single compressed file
jnp.savez(
    "rpa_vs_td.npz",
    omegas_rpa=omegas_rpa,
    absorption_rpa=absorption_rpa,
    omegas_td=omegas_td,
    absorption_td=absorption_td,
)
