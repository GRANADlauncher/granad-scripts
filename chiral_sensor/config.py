import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *

from lib import *

IP_RESPONSE = True
RPA_RESPONSE = True
MEAN_FIELD = True
GEOMETRIES = True
PLOT_IP_RESPONSE = True
PLOT_RPA_RESPONSE = True
PLOT_MEAN_FIELD = True
PLOT_GEOMETRIES = True

if __name__ == '__main__':
    if IP_RESPONSE:
        args_list = [
            (Triangle(30, armchair = True), -2.66, -1j*t2, delta, f"haldane_graphene_{t2}" )
            for (t2, delta) in [(0.0, 0.0), (0.1, 0.3), (0.5, 0.3)]
        ]
        ip_response(f, args_list)
        
    if PLOT_IP_RESPONSE:
        plot_chirality_difference("cond_" + f)
        plot_chirality("cond_" + f)
        plot_power("cond_" + f)
        plot_response_functions(f)

    if RPA_RESPONSE:
        flake = get_haldane_graphene(-2.66, -0.5j, 0.3).cut_flake(Triangle(30))
        rpa_response(flake, "triangle", [0, 0.01, 0.1, 0.5, 0.7, 1.0])

    if PLOT_RPA_RESPONSE:
        plot_rpa_response("rpa_triangle.npz")

    if MEAN_FIELD:
        t1, t2, delta, shape = -2.66, -1j, 0.3, Triangle(30)
        flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
        Us = [0, 0.1, 0.2, 1., 1.5, 2., 2.5, 3.]
        res = [scf_loop(flake, U, 0.0, 1e-10, 100) for U in Us]
        jnp.savez("scf.npz", res = res, Us = Us, pos = flake.positions)

    if PLOT_MEAN_FIELD:
        plot_stability("scf.npz")

    if PLOT_GEOMETRIES:
        # plot edge states vs localization-annotated energy landscape of a few structures
        setups = [
            (shape, -2.66, -1j, 0.3, f"haldane_graphene" )
            for shape in [Triangle(18, armchair = False), Rectangle(10, 10), Hexagon(20, armchair = True)]
        ]
        plot_edge_states_energy_landscape(setups)
        plot_localization_varying_hopping(setups)


