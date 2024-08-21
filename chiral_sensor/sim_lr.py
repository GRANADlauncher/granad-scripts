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

# TODO: conductivity from chi and RPA
def rpa_conductivity(args_list):    
    return

def sim(results_file):
    """runs simulation for IP j-j and p-p total and topological response.     
    saves j-j, pp in "cond_" + results_file, "pol_" + results_file
    """
    
    args_list = [
        (Triangle(30, armchair = True), -2.66, -1j*t2, delta, f"haldane_graphene_{t2}" )
        for (t2, delta) in [(0.0, 0.0), (-0.5, 0.3), (-0.1, 0.3), (0.1, 0.3), (0.5, 0.3)]
        ]

    print("plotting edge states")
    for args in args_list:
        plot_edge_states(args)
        plot_energies(args)
    print("lrt")

    cond, pol = {}, {}
    omegas = jnp.linspace(0, 10, 100)    
    for args in args_list:        
        flake = get_haldane_graphene(*args[1:4]).cut_flake(args[0])
        
        v, p = flake.velocity_operator_e, flake.dipole_operator_e
              
        # compute only topological sector
        trivial = jnp.abs(flake.energies) > 1e-1
        mask = jnp.logical_and(trivial[:, None], trivial)

        cond[args[-1]] = jnp.array([[flake.get_ip_green_function(v[i], v[j], omegas, relaxation_rate = 0.01) for i in range(2)] for j in range(2)])
        pol[args[-1]] = jnp.array([[flake.get_ip_green_function(p[i], p[j], omegas, relaxation_rate = 0.01) for i in range(2)] for j in range(2)])
        
        cond["topological." + args[-1]] = jnp.array([[flake.get_ip_green_function(v[i], v[j], omegas, relaxation_rate = 0.01, mask = mask) for i in range(2)] for j in range(2)])
        pol["topological." + args[-1]] = jnp.array([[flake.get_ip_green_function(p[i], p[j], omegas, relaxation_rate = 0.01, mask = mask) for i in range(2)] for j in range(2)])
        
    cond["omegas"], pol["omegas"] = omegas, omegas
    jnp.savez("cond_" + results_file, **cond)
    jnp.savez("pol_" + results_file, **pol)

### postprocessing ###
def plot_edge_states(args):
    """plot topological E~0 excitations, if present"""    
    shape, t1, t2, delta, name = args
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    try:
        idx = jnp.argwhere(jnp.abs(flake.energies) < 1e-1)[0].item()
        flake.show_2d(display = flake.eigenvectors[:, idx], scale = True, name = name + ".pdf")
    except IndexError:
        print(args, " has no edge states")
    
def plot_energies(args):
    """plots energy landscape of haldane graphene flake specified by args"""    
    shape, t1, t2, delta, name = args
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    flake.show_energies(name = name + "_energies.pdf")

def plot_static(args):
    """wrapper around plot_energies and plot_edge_states"""
    for args in args_list:
        plot_edge_states(args)
        plot_energies(args)

# TODO: net current in excited state going around the flake clock-wise
def current_map(args_list):
    return

# TODO: lookup greens function
def scattered_field(results_file, illu, r):
    return

def to_helicity(mat):
    """converts mat to helicity basis"""
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    trafo_inv = jnp.linalg.inv(trafo)
    return jnp.einsum('ij,jmk,ml->ilk', trafo, mat, trafo_inv)

def plot_response_functions(results_file):
    """plots j-j response directly and as obtained from p-p response"""    
    with jnp.load("cond_" + results_file) as data:
        cond = dict(data)
        cond_omegas = cond.pop("omegas")        
    with jnp.load("pol_" + results_file) as data:
        pol = dict(data)
        pol_omegas = pol.pop("omegas")
        
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    keys = cond.keys()
    for k in keys:    
        for i in range(2):
            for j in range(2):
                axs[i, j].plot(cond_omegas, cond[k][i, j].real, label='cond_' + k)
                axs[i, j].plot(pol_omegas, pol_omegas**2 * pol[k][i, j].real, '--', label='pol_' + k)
                axs[i, j].set_title(f'i,j = {i,j}')
                
    axs[0, 0].legend(loc="upper left")
    plt.savefig("cond_pol_comparison.pdf")
    plt.close()

def load_data(results_file, keys):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")        
    return omegas, data, data.keys() if keys is None else keys
    
def plot_excess_chirality(results_file, keys = None):
    """plots excess chirality of the total response"""
    
    omegas, data, keys = load_data(results_file, keys)
    
    for i, key in enumerate(keys):
        mat = to_helicity(data[key])
        mat_real, mat_imag = mat.real, mat.imag

        if 'topological' in key:
            continue
        
        excess = jnp.abs(mat_imag[0,0]) / jnp.abs(mat_imag[1, 1])
        excess /= excess.max()
        plt.plot(omegas, excess, label = key.split("_")[-1])
        
    plt.legend(loc = "upper left")
    plt.savefig("excess_chirality.pdf")
    plt.close()

def plot_topological_total(results_file, keys = None):
    """plots topological and total response on same x-axis"""
    
    omegas, data, keys = load_data(results_file, keys)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    for i, key in enumerate(keys):
        mat = to_helicity(data[key])
        mat_real, mat_imag = mat.real, mat.imag

        ls = '--' if 'topological' in key else '-'
        if 'topological' in key:            
            ax2.plot(omegas, mat_imag[0, 0], ls = ls, label = key.split("_")[-1])
            ax2.legend(loc = "upper left")

        excess = jnp.abs(mat_imag[0,0]) / jnp.abs(mat_imag[1, 1])
        excess /= excess.max()
        ax1.plot(omegas, excess, ls = ls, label = key.split("_")[-1])
        ax1.legend(loc = "upper left")

    plt.savefig("chirality_topological_total.pdf")
    plt.close()

            
if __name__ == '__main__':
    f = "lrt.npz"    
    # sim(f)
    plot_response_functions(f)
    plot_excess_chirality("cond_" + f)
    plot_topological_total("cond_" + f)
