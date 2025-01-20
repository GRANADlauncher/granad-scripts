import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *

from lib import *

# pt at t_2 = delta / (3 * np.sqrt(3))

IP_RESPONSE = True
PLOT_IP_RESPONSE = True
LRT_FILE = 'lrt.npz'
IP_ARGS = [ (get_haldane_graphene(-2.66, -1j*t2, delta).cut_flake(Triangle(30, armchair = True)), f"haldane_graphene_{t2}") for (t2, delta) in [(0.0, 0.0), (0.1, 1), (0.5, 1)] ]


RPA_RESPONSE = False
PLOT_RPA_RESPONSE = False
RPA_FILE = 'rpa_triangle.npz'
RPA_FLAKE = get_haldane_graphene(-2.66, -0.5j, 0.3).cut_flake(Triangle(30))
RPA_VALS = [0, 0.01, 0.1, 0.5, 0.7, 1.0]

MEAN_FIELD = False
PLOT_MEAN_FIELD = False
MF_FLAKE = get_haldane_graphene(-2.66, -1j, 0.3).cut_flake(Triangle(30))
MF_VALS = [0, 0.1, 0.2, 1., 1.5, 2., 2.5, 3.]
MF_MIX, MF_ITER, MF_PREC = 0.0, 100, 1e-10
MF_FILE = 'mf.npz'

PLOT_GEOMETRIES = False
GEOMETRIES = [
    (shape, -2.66, -1j, 0.3, f"haldane_graphene" )
    for shape in [Triangle(18, armchair = False), Rectangle(10, 10), Hexagon(20, armchair = True)]
]

if __name__ == '__main__':
    if IP_RESPONSE:
        ip_response(IP_ARGS, LRT_FILE)
        
    if PLOT_IP_RESPONSE:
        plot_chirality_difference("cond_" + LRT_FILE)
        plot_power("cond_" + LRT_FILE)
        plot_response_functions(LRT_FILE)

        flake = IP_ARGS[2][0]
        loc = localization(flake.positions, flake.eigenvectors, flake.energies)
        plot_chirality("cond_" + LRT_FILE, flake, display = jnp.abs(flake.eigenvectors[:, loc.argmax()])**2)
        plot_chirality_topo("cond_" + LRT_FILE, flake, display = jnp.abs(flake.eigenvectors[:, loc.argmax()])**2)

    if RPA_RESPONSE:
        rpa_response(RPA_FLAKE, RPA_FILE, RPA_VALS)

    if PLOT_RPA_RESPONSE:
        plot_rpa_response(RPA_FILE)

    if MEAN_FIELD:
        res = [scf_loop(MF_FLAKE, u, MF_MIX, MF_PREC, MF_ITER) for u in MF_VALS]
        jnp.savez(MF_FILE, res = res, Us = MF_VALS, pos = MF_FLAKE.positions)

    if PLOT_MEAN_FIELD:
        plot_stability(MF_FILE)

    if PLOT_GEOMETRIES:
        plot_edge_states_energy_landscape(GEOMETRIES)
        plot_localization_varying_hopping(GEOMETRIES)        
