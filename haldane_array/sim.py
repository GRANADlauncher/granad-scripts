"""common utilities"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import jax.numpy as jnp
from flax import struct

from granad import *

### UTILITIES ###
def load_data(results_file, keys):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")
    return omegas, data, data.keys() if keys is None else keys    

def get_threshold(delta):
    """threshold for topological nontriviality for t_2"""
    return delta / (3 * jnp.sqrt(3) )

def get_haldane_graphene(t1, t2, delta):
    """Constructs a graphene model with
    onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2

    threshold is at $t_2 > \frac{\delta}{3 \sqrt{3}}$
    """
    return (
        Material("haldane_graphene")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz1", atom='C')
        .add_orbital_species("pz2", atom='C')
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


### RESPONSE ###
def ip_response(args_list, results_file):
    """computes IP j-j and p-p response. saves j-j, pp in "cond_" + results_file, "pol_" + results_file.
    """

    def get_correlator(operators, mask = None):
        return jnp.array([[flake.get_ip_green_function(o1, o2, omegas, relaxation_rate = 0.05, mask = mask) for o1 in operators] for o2 in operators])

    pol = {}
    omegas = jnp.linspace(0, 6, 200)    
    for (flake, name) in args_list:        
        v, p = flake.velocity_operator_e, flake.dipole_operator_e
        pol[name] = get_correlator(p[:2])

        trivial = jnp.abs(flake.energies) > 0.1
        print("edge states for", name, len(flake) - trivial.sum())

        mask = jnp.logical_and(trivial[:, None], trivial)
        
        pol["topological." + name] = get_correlator(p[:2], mask)
        
    pol["omegas"] = omegas
    jnp.savez("pol_" + results_file, **pol)

### RPA ###
def rpa_sus(evs, omegas, occupations, energies, coulomb, electrons, relaxation_rate = 0.05):
    """computes pp-response in RPA, following https://pubs.acs.org/doi/10.1021/nn204780e"""
    
    def inner(omega):
        mat = delta_occ / (omega + delta_e + 1j*relaxation_rate)
        sus = jnp.einsum('ab, ak, al, bk, bl -> kl', mat, evs, evs.conj(), evs.conj(), evs)
        return sus @ jnp.linalg.inv(one - coulomb @ sus)

    one = jnp.identity(evs.shape[0])
    evs = evs.T
    delta_occ = (occupations[:, None] - occupations) * electrons
    delta_e = energies[:, None] - energies
    
    return jax.lax.map(jax.jit(inner), omegas)

def rpa_response(flake, results_file, cs):
    """computes j-j response from p-p in RPA"""
    
    omegas =  jnp.linspace(0, 6, 200)
    res = []
    
    for c in cs:        
        sus = rpa_sus(flake.eigenvectors, omegas, flake.initial_density_matrix_e.diagonal(), flake.energies, c*flake.coulomb, flake.electrons)        
        p = flake.positions.T        
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)        
        res.append(omegas[None, None, :]**2 * ref)
        
    jnp.savez(results_file, cond = res, omegas = omegas, cs = cs)

def show_2d(orbs, show_tags=None, show_index=False, display = None, scale = False, cmap = None, circle_scale : float = 1e3, title = None, mode = None, indicate_atoms = False, grid = False):

    # decider whether to take abs val and normalize 
    def scale_vals( vals ):
        return jnp.abs(vals) / jnp.abs(vals).max() if scale else vals

            
    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 33,
        "axes.labelsize": 22,
        "xtick.labelsize": 8*2,
        "ytick.labelsize": 8*2,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }

    scale = 1

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        # Create plot
        fig, ax = plt.subplots()
        cmap = plt.cm.bwr if cmap is None else cmap
        colors = scale_vals(display)            
        scatter = ax.scatter([orb.position[0] * scale for orb in orbs], [orb.position[1]  * scale for orb in orbs], c=colors, edgecolor='none', cmap=cmap, s = circle_scale*jnp.abs(display) )
        ax.scatter([orb.position[0] * scale  for orb in orbs], [orb.position[1]  * scale  for orb in orbs], color='black', s=5, marker='o')            
        cbar = fig.colorbar(scatter, ax=ax)

        # Finalize plot settings
        if title is not None:
            plt.title(title)
            
        plt.xlabel('X (Å)')
        plt.ylabel('Y (Å)')    
        ax.grid(True)
        ax.axis('equal')
        plt.savefig('geometry.pdf')

# TODO: pick g, a, t0, r0, area
# TODO: bring in alpha
# TODO: fix speed of light

LIGHT = 1

@struct.dataclass
class Lattice:
    r0: float
    t0: float
    g: float
    a: float
    area: float
        
def S_s(area, omegas, theta):
    return 2*jnp.pi*omegas/(LIGHT * area * jnp.cos(theta))

def S_p(area, omegas, theta):
    return 2*jnp.pi*omegas*jnp.cos(theta)/(LIGHT * area)
        
def reflection_layer(lattice, alpha, S):
    # periodically patterned haldane model
    G_im = S - 2 * (omegas/LIGHT)**3 / 3
    G_re = lattice.g / lattice.a**3    
    r = 1j * S / (alpha - G)
    return r, -r

def amplitude(lattice, alpha, S):
    r = reflection_layer(lattice, alpha, S)
    eta_s = r*(1 + lattice.r0) / (1 - lattice.r0 * r)
    eta_p= r*(1 - lattice.r0) / (1 - lattice.r0 * r)
    return eta_s, eta_p

def total_reflection(lattice, alpha, S):
    eta = amplitude(lattice, alpha, incidence, S)    
    R_s = lattice.r0 + (1 + lattice.r0)*eta
    R_p = lattice.r0 + (1 - lattice.r0)*eta
    return R_s, R_p

def total_transmission(lattice, alpha, S):
    eta = amplitude(lattice, alpha, incidence, S)    
    T_s = lattice.t0 + lattice.t0 *eta
    T_p = lattice.t0 - lattice.t0*eta
    return T_s, T_p

def total_absorption(lattice, alpha, incidence, S):
    x = total_reflection(lattice, alpha, incidence, S) + total_transmission(lattice, alpha, incidence, S)
    return 1 - x[0], 1 - x[1]
        
if __name__ == '__main__':    
    lattice()
