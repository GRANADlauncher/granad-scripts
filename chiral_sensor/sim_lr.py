from itertools import combinations

import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *

# haldane model has topological phase for Im[t2] > \frac{M}{3 \sqrt{3}} => for 0.3 Im[t_2]_crit ~ 0.06
# sim input : (shape, t1, t2, delta (mass term), name)
def get_haldane_graphene(t1, t2, delta):
    """Constructs a graphene model with
    onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2
    """
    return (
        Material("haldane_graphene")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz1", l=1, atom='C')
        .add_orbital_species("pz2", l=1, atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz1")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz2")
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz2"),
            parameters= [t1],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz1"),            
            # a bit overcomplicated
            parameters=[                
                [0, 0, 0, delta], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, t2], 
                [2.46, 0, 0, jnp.conj(t2)],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)]
            ],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz2", "pz2"),
            parameters=[                
                [0, 0, 0, 0], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, jnp.conj(t2)], 
                [2.46, 0, 0, t2],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2]
            ],
        )
        .add_interaction(                
            "coulomb",
            participants=("pz1", "pz2"),
            parameters=[8.64],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz1", "pz1"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz2", "pz2"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
    )

### sim ###

# MAYBE: conductivity from chi and RPA
def rpa_conductivity(args_list):    
    return

def conductivity(results_file):    
    args_list = [
        (Hexagon(40, armchair = True), -2.66, -1j*t2, 0.3, f"haldane_graphene_{t2}" )
        for t2 in [0.01, 0.5]
        ]

    print("plotting edge states")
    for args in args_list:
        plot_edge_states(args)
        plot_energies(args)
    print("lrt")

    cond, pol = {}, {}
    omegas = jnp.linspace(10, 40, 300)    
    for args in args_list:        
        flake = get_haldane_graphene(*args[1:4]).cut_flake(args[0])        
        v, p = flake.velocity_operator, flake.dipole_operator
        cond[args[-1]] = jnp.array([[flake.get_ip_green_function(v[i], v[j], omegas, relaxation_rate = 0.01) for i in range(2)] for j in range(2)])
        pol[args[-1]] = jnp.array([[flake.get_ip_green_function(p[i], p[j], omegas, relaxation_rate = 0.01) for i in range(2)] for j in range(2)])

        # compute only topological sector
        trivial = jnp.abs(flake.energies) > 1e-1
        mask = jnp.logical_and(trivial[:, None], trivial)
        cond["topological." + args[-1]] = jnp.array([[flake.get_ip_green_function(v[i], v[j], omegas, relaxation_rate = 0.01, mask = mask) for i in range(2)] for j in range(2)])
        pol["topological." + args[-1]] = jnp.array([[flake.get_ip_green_function(p[i], p[j], omegas, relaxation_rate = 0.01, mask = mask) for i in range(2)] for j in range(2)])
        
    cond["omegas"], pol["omegas"] = omegas, omegas
    jnp.savez("cond_" + results_file, **cond)
    jnp.savez("pol_" + results_file, **pol)

### postprocessing ###
def plot_edge_states(args):
    shape, t1, t2, delta, name = args
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)    
    idx = jnp.argwhere(jnp.abs(flake.energies) < 1e-1)[0].item()    
    flake.show_2d(display = flake.eigenvectors[:, idx], scale = True, name = name + ".pdf")
    
def plot_energies(args):
    shape, t1, t2, delta, name = args
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    flake.show_energies(name = name + "_energies.pdf")

def plot_static():
    plot_args_list = [
        (Triangle(10, armchair = True), -2.66, -0.1j, 0.3, f"triangle" ),
        (Hexagon(40, armchair = True), -2.66, -0.1j, 0.3, f"hexagon" ),
        (Rectangle(40, 20, armchair = True), -2.66, -0.1j, 0.3, f"ribbon" ),        
        ]    
    for args in plot_args_list:
        plot_edge_states(args)
        plot_energies(args)

# TODO: net current in excited state going around the flake clock-wise
def current_map(args_list):
    return

# TODO: lookup greens function
def scattered_field(results_file, illu, r):
    return

# TODO:
def average_chirality_density(sigma, illu):
    return

def to_helicity(mat):
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    trafo_inv = jnp.linalg.inv(trafo)
    return jnp.einsum('ij,jmk,ml->ilk', trafo_inv, mat, trafo)

def plot_response_functions(results_file):
    with jnp.load("cond_" + results_file) as data:
        cond = dict(data)
        cond_omegas = cond.pop("omegas")        
    with jnp.load("pol_" + results_file) as data:
        pol = dict(data)
        pol_omegas = pol.pop("omegas")
        
    keys = cond.keys() if keys is None else keys
    for k in keys:    
        plt.plot(cond_omegas, cond[k], label = 'cond_' + key )
        plt.plot(pol_omegas, pol_omegas ** 2 * pol[k], '--', label = 'pol_' + key)
    plt.savefig("cond_pol_comparison.pdf")
    
def plot_chirality_difference(results_file, keys = None):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")        
        keys = data.keys() if keys is None else keys
        print(data.keys())
        
        for i, key in enumerate(keys):
            mat = to_helicity(data[key])
            mat_real, mat_imag = jnp.abs(mat.real), jnp.abs(mat.imag)
            ls = '--' if 'topological' in key else '-'                
            # plt.semilogy(omegas, (mat[0, 0] - mat[1, 1]).real, ls = ls, label = key.split("_")[-1])
            plt.semilogy(omegas, mat_imag[0, 0], ls = ls, label = key.split("_")[-1])
            plt.semilogy(omegas, mat_imag[1, 1], ls = ls, label = 'foo_' + key.split("_")[-1])
            # mat = data[key]
            # plt.plot(mat[0, 0], label = key.split("_")[-1])
            # plt.plot(mat[1, 1], label = key.split("_")[-1])

        # plt.ylim(1)
        plt.legend(loc = "upper left")
        plt.savefig("chirality_difference.pdf")
        plt.close()

if __name__ == '__main__':
    f = "lrt.npz"    
    conductivity(f)
    plot_response_functions(f)
    plot_chirality_difference(f, keys = keys)
