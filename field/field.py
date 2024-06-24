import jax.numpy as jnp
import scipy
import matplotlib.pyplot as plt
from granad import *
from granad import _numerics

if __name__ == '__main__':

    # flake
    n, m = 10, 10
    flake = MaterialCatalog.get("graphene").cut_flake(Rhomboid(n,m))    
    flake.show_2d(name = "flake.pdf")
    
    # illumination
    pulse = Pulse(
        amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
    )
    
    # get density matrix at peak of illumination
    grid = jnp.array([0.1, 5., 10.])
    result = flake.master_equation(
        relaxation_rate = 1/10,
        illumination = pulse,
        dt = 1e-5,
        density_matrix = 'full',
        grid = grid,
        end_time = 10
    )

    
    x = jnp.linspace(flake.positions[:, 0].min()-1, flake.positions[:,0].max()+1, 20)
    y = jnp.linspace(flake.positions[:, 1].min()-1, flake.positions[:,1].max()+1, 20)
    for i in range(3):
        flake.show_induced_field(x, y, jnp.array([0]),
                                 density_matrix = flake.stationary_density_matrix - result.output[0][i],
                                 name = f"t = {grid[i]}.pdf"                       
                                 )
