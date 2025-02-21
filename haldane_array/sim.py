# relevant scales:
# length in nm, energies in eV, hbar = 1
import jax
import jax.numpy as jnp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

from granad import *


### UTILITIES ###
def load_data(results_file, keys):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")
    return omegas, data, data.keys() if keys is None else keys    

def get_threshold(delta):
    """threshold for topological nontriviality for lambda"""
    return delta / (3 * jnp.sqrt(3) )

LIGHT = 299.8
def wavelength(omega):
    return LIGHT / (omega / 2*jnp.pi)

def omega(wavelength):
    return 2*jnp.pi * LIGHT / wavelength
    
def to_helicity(mat):
    """converts mat to helicity basis"""
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1], [1j, -1j] ])    
    return jnp.einsum('ij,jmk,ml->ilk', trafo.conj().T, mat, trafo)

### MATERIAL ###
def sample_brillouin_zone(num_kpoints=100):
    """
    Samples the hexagonal Brillouin zone along the high-symmetry path Γ-K-M-K'-Γ.
    
    Parameters:
    num_kpoints : int
        Number of k-points to sample along the path.
    
    Returns:
    kx, ky : ndarray
        Arrays of kx and ky values along the high-symmetry path.
    """
    # High-symmetry points in the hexagonal Brillouin zone
    Gamma = np.array([0, 0])
    K = np.array([4*np.pi/3, 0])
    M = np.array([2*np.pi/3, 2*np.pi/3*np.sqrt(3)])
    Kp = np.array([-4*np.pi/3, 0])
    
    # Define path segments
    segments = [Gamma, K, M, Kp, Gamma]
    
    # Generate k-point path
    kx, ky = [], []
    for i in range(len(segments)-1):
        k_start, k_end = segments[i], segments[i+1]
        k_linspace = np.linspace(k_start, k_end, num_kpoints // (len(segments)-1))
        kx.extend(k_linspace[:, 0])
        ky.extend(k_linspace[:, 1])
    
    return np.array(kx), np.array(ky)

# def haldane_hamiltonian(k, t1=1.0, t2=0.0, phi=jnp.pi/2, M=0.0):
# def haldane_hamiltonian(k, t1=1.0, t2=0.2/(jnp.sqrt(3)*3), phi=jnp.pi/2, M=0.2):
def haldane_hamiltonian(k, t1=1.0, t2=-0.1, phi=jnp.pi/2, M=0.2):
    """
    Computes the Haldane model Hamiltonian in momentum space.
    
    Parameters:
    k : array-like
        momentum in the Brillouin zone.
    t1 : float
        Nearest-neighbor hopping amplitude.
    t2 : float
        Next-nearest-neighbor hopping amplitude.
    phi : float
        Phase associated with the complex hopping (breaks time-reversal symmetry).
    M : float
        Sublattice potential term (breaks inversion symmetry).
    
    Returns:
    H : ndarray
        Hamiltonian matrix of shape (2, 2, len(kx))
    """    
    sigma_0 = jnp.eye(2)
    sigma_x = jnp.array( [ [0, 1], [1, 0] ] )
    sigma_y = 1j * jnp.array( [ [0, 1], [-1, 0] ] )
    sigma_z = jnp.array( [ [1, 0], [0, -1] ] )

    # nearest-neighbor vectors
    a_vecs = jnp.array( [
        [1, 0],
        [-1/2, jnp.sqrt(3)/2],
        [-1/2, -jnp.sqrt(3)/2]        
        ]
    )
    
    # Next-nearest-neighbor vectors
    b_vecs = jnp.array([
        [0, jnp.sqrt(3)],
        [-3/2, -jnp.sqrt(3)/2],
        [3/2, -jnp.sqrt(3)/2]
        ]
    )

    # wave numbers
    ka = a_vecs @ k 
    kb = b_vecs @ k

    # Hamiltonian components
    A0 = 2 * t2 * jnp.cos(phi) * jnp.cos(kb).sum()
    A1 = t1 * jnp.cos(ka).sum()
    A2 = t1 * jnp.sin(ka).sum()
    A3 = -2 * t2 * jnp.sin(phi) * jnp.sin(kb).sum() + M

    # hamiltonian matrix
    H = A0 * sigma_0 + A1 * sigma_x + A2 * sigma_y + A3 * sigma_z
    
    return H

def get_haldane_graphene(t1, t2, delta):
    """Constructs a graphene model with onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2

    threshold is at $\\lambda > \\frac{\\delta}{3 \\sqrt{3}}$
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


### IP RESPONSE ###
def get_correlator(flake, omegas, os1, os2, relaxation_rate, mask = None):
    return jnp.array([
        [
            flake.get_ip_green_function(o1, o2, omegas, relaxation_rate = relaxation_rate, mask = mask) for o1 in os1
        ]
        for o2 in os2]
                     )

def ip_response(flake, omegas, relaxation_rate = 0.05, os1 = None, os2 = None, results_file = None, topology = False):
    """computes Wx3x3 IP polarizability according to usual lehman representation"""
    corr = {}
    os1 = os1 if os1 is not None else flake.dipole_operator_e[:2]
    os2 = os2 if os2 is not None else flake.dipole_operator_e[:2]
    
    corr["total"] =  get_correlator(flake, omegas, os1, os2, relaxation_rate = relaxation_rate)

    if topology == True:
        trivial = jnp.abs(flake.energies) > 0.1
        mask = jnp.logical_and(trivial[:, None], trivial)        
        corr["topological"] = get_correlator(flake, omegas, os1, os2, relaxation_rate = relaxation_rate, mask = mask)

    corr["omegas"] = omegas
    
    if results_file is not None:
        jnp.savez(results_file, **corr)
        
    return corr

### RPA ###
def rpa_susceptibility(flake, c, relaxation_rate):
    """computes RPA susceptibility, following https://pubs.acs.org/doi/10.1021/nn204780e"""
    
    def inner(omega):
        mat = delta_occ / (omega + delta_e + 1j*relaxation_rate)
        sus = jnp.einsum('ab, ak, al, bk, bl -> kl', mat, evs, evs.conj(), evs.conj(), evs)
        return sus @ jnp.linalg.inv(one - coulomb @ sus)
    
    one = jnp.identity(len(flake))
    coulomb = c * flake.coulomb
    evs = flake.eigenvectors.T
    occupations = flake.initial_density_matrix_e.diagonal()
    delta_occ = (occupations[:, None] - occupations) * flake.electrons
    delta_e = flake.energies[:, None] - flake.energies
    
    return jax.lax.map(jax.jit(inner), omegas)

def rpa_polarizability(flake, omegas, cs, relaxation_rate = 0.05, results_file = None):
    """computes RPA polarizability, following https://pubs.acs.org/doi/10.1021/nn204780e"""
    pol = []    
    for c in cs:
        # sus is sandwiched like x * sus * x
        sus = rpa_susceptibility(flake, c, relaxation_rate = 0.05)
        
        p = flake.positions.T
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)

        # TODO: check if this is right, maybe missing omegas?
        pol.append(ref)

    if results_file is not None:
        jnp.savez(results_file, pol = pol, omegas = omegas, cs = cs)
        
    return res

### PLOTTING ###    
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

def get_projection(dip):
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, -1j], [1, 1j] ])
    return jnp.einsum('ij, jkl -> ikl', trafo, dip)

def plot_projected_polarization():
    """Plots projection of polarization operator matrix elements onto circular basis."""
    shape = Triangle(20, armchair=False)
    
    delta = 1.0
    t_nn = 1.0
    ts = [0.0, 0.1, 0.2]
    
    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 33,
        "axes.labelsize": 33,
        "xtick.labelsize": 8*3,
        "ytick.labelsize": 8*3,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }
    
    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
        
        # Create a figure with a 1x3 grid of subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))

        for i, t in enumerate(ts):
            flake = get_haldane_graphene(t_nn, -1j*t, delta).cut_flake(shape)
            dip = flake.velocity_operator_e[:2]
            projection = get_projection(dip)
            
            norm = None #LogNorm()
            im = axes[i].matshow(jnp.abs(projection[0])**2, norm=norm, cmap = "twilight_r")

            # Move x-ticks below the plot
            axes[i].xaxis.set_ticks_position("bottom")

            # Attach a colorbar on top of the matshow plot
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("top", size="5%", pad=0.3)  # "top" places it above

            # Create colorbar with horizontal orientation
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")

            # Set label above the colorbar
            cax.xaxis.set_ticks_position("top")  # Move ticks to the top
            axes[i].set_title(rf"$\lambda / t$ = {t:.2f}", pad = 110)
            
            if i != 0:
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            else:
                axes[i].set_xlabel(r"$m$")
                axes[i].set_ylabel(r"$n$")

            if i == 1:
                cbar.set_label(r"$\vert J_+ \vert^2$ (a.u.)", fontsize=25, labelpad=-70)
                        
    plt.tight_layout()
    plt.savefig("projected_polarizations.pdf")


def plot_phase_shift():
    """plots phase between E_x and E_y"""
    shape = Triangle(20, armchair = False)
    
    delta = 1.0
    t_nn = -2.66
    
    ts = [0, 0.15, 0.4]
    
    # omegas
    omegas = jnp.linspace(-0.0, 2, 100)

    
    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 33,
        "axes.labelsize": 33,
        "xtick.labelsize": 8*3,
        "ytick.labelsize": 8*3,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        f_dip = lambda xx : xx.sum(axis=1)

        for t in ts:
            flake = get_haldane_graphene(t_nn, -1j*t, delta).cut_flake(shape)  
            alpha_cart = ip_response(flake, omegas, relaxation_rate = 0.01)["total"]        

            dip = f_dip(alpha_cart)

            ls = '-' if t > get_threshold(delta) else '--'

            plt.plot(omegas, jnp.angle(dip[0] / dip[1]), label = rf'$p_+$ {t:.2f}')


        plt.legend()
        plt.savefig("phase.pdf")
        plt.close()
        
def find_peaks(arr):
    # Create boolean masks for peak conditions
    left = arr[1:-1] > arr[:-2]   # Compare each element to its left neighbor
    right = arr[1:-1] > arr[2:]   # Compare each element to its right neighbor
    
    peaks = jnp.where(left & right)[0] + 1  # Get indices and shift by 1 to match original array
    return peaks

def get_closest_transition(flake, omega):
    diff = jnp.abs(flake.energies - flake.energies[:, None])
    
    # Find the index of the closest element to omega
    idx = jnp.argmin(jnp.abs(diff - omega))
    
    # Convert flattened index to row and column indices
    row, col = jnp.unravel_index(idx, diff.shape)
    
    return row, col
    
def plot_dipole_moments():
    """plots p_+, p_-"""
    shape = Triangle(30, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = [0, 0.15, 0.4]
    ts = [0.4]
    
    # omegas
    omegas = jnp.linspace(0., 0.8, 300)    

    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 8*2,
        "ytick.labelsize": 8*2,
        "legend.fontsize": 9*1.2,
        "pdf.fonttype": 42
    }
    
    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, -1j], [1, 1j] ])
        f_dip = lambda xx : jnp.abs(  jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)) )

        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)  
            alpha_cart = ip_response(flake, omegas, relaxation_rate = 1e-3)["total"]
            dip = f_dip(alpha_cart)

            proj = get_projection(flake.velocity_operator_e[:2])
            
            # diff = dip[0] - dip[1]
            
            plt.plot(omegas, dip[0], label = rf'$|p_+|$')
            plt.plot(omegas, dip[1], label = rf'$|p_-|$', ls = '--')
            plt.yscale('log')
            plt.grid(True)

            plt.xlabel(r'$\omega / t$')
            plt.ylabel(r'$|p|$ (a.u.)')

        pp = find_peaks(dip[0])[0]
        pp_max, omega_p = dip[0][pp].item(), omegas[pp].item()
        pm = find_peaks(dip[1])[0]
        pm_max, omega_m = dip[1][pm].item(), omegas[pm].item()
        
        peaks = [
            ([(omega_p, pp_max),
              (omega_p*1.2, pp_max*1)],
             r"$\propto$ max$(\vert J_{+} \vert^2)$" ),
            ([(omega_m, pm_max),
              (omega_m * 1.2, pm_max*1.3)],
             r"$\propto$ max$(\vert J_{-} \vert^2)$" ),
        ]

            
        for p in peaks:
            pos, annotation = p
            plt.annotate(annotation, xy=pos[0], xytext=pos[1], arrowprops=dict(arrowstyle="->,head_width=.15"), fontsize = 15)

        plt.legend()
        plt.savefig("p.pdf")
        plt.close()


def plot_dipole_moments_sweep():
    """plots p_+ - p_- in colormap"""
    shape = Triangle(10, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = jnp.linspace(0, 0.4, 10)
    
    # omegas
    omegas = jnp.linspace(0., 0.8, 300)    

    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 22,
        "axes.labelsize": 22,
        "xtick.labelsize": 8*2,
        "ytick.labelsize": 8*2,
        "legend.fontsize": 9*1.2,
        "pdf.fonttype": 42
    }

    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, -1j], [1, 1j] ])
    f_dip = lambda xx : jnp.abs(  jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)) )    
    res = []    
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)  
        alpha_cart = ip_response(flake, omegas, relaxation_rate = 1e-3)["total"]
        dip = f_dip(alpha_cart)
        res.append(dip[0] - dip[1])
    res = jnp.array(res)
    
    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        im = plt.matshow(res,
                         aspect='auto', 
                         cmap='coolwarm', 
                         origin='lower', 
                        extent=[ts.min(), ts.max(), omegas.min(), omegas.max()]
                         )
        
        # Axis labels
        plt.xlabel(r'$\lambda / t$', weight='bold')
        plt.ylabel(r'$\omega / t$', weight='bold')

        # Create colorbar with horizontal orientation
        cbar = plt.colorbar(im, label = r'$p_+ - p_-$')

        plt.legend()
        plt.savefig("p_sweep.pdf")
        plt.close()
        

def plot_dipole_moments_topological_vs_trivial():
    """plots p_+, p_- for topological and trivial contributions"""
    shape = Triangle(30, armchair = False)
    
    delta = 1.0
    t_nn = -2.66
    
    ts = [0, 0.15, 0.4]
    
    # omegas
    omegas = jnp.linspace(0., 2, 200)    

    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 33,
        "axes.labelsize": 33,
        "xtick.labelsize": 8*3,
        "ytick.labelsize": 8*3,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, -1j], [1, 1j] ])
        f_dip = lambda xx : jnp.abs(  jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)) )

        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)  
            alpha_cart = ip_response(flake, omegas, relaxation_rate = 1e-3)["total"]
            dip = f_dip(alpha_cart)

            proj = get_projection(flake.velocity_operator_e[:2])
            diff = dip[0] - dip[1]
            
            plt.plot(omegas, diff, label = rf'$\lambda$ = {t:.2f}')

            plt.xlabel(r'$\omega$ (eV)')
            plt.ylabel(r'$\Delta p$ (a.u.)')

            # plt.plot(omegas, dip[0], label = rf'$p_+$ {t:.2f}')
            # plt.plot(omegas, dip[1], label = rf'$p_-$ {t:.2f}', ls = '--')

        plt.legend()
        plt.savefig("p.pdf")
        plt.close()

        
def plot_dipole_moments_p_j():
    """plots p_+, p_- computed from xpp and xjj"""
    shape = Triangle(20, armchair = True)
    
    delta = 1.0
    t_nn = -2.66
    
    ts = [0, 0.15, 0.4]
    
    # omegas
    omegas = jnp.linspace(-0., 8, 100)    

    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    f_dip = lambda xx : jnp.abs(  jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)) )

    # xjj = w**2 xpp
    f_dip_j = lambda jj : f_dip( (jj - jj[..., 0][:, :, None]) / omegas**2 )
    
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)  
        alpha_cart = ip_response(flake, omegas, relaxation_rate = 0.01)["total"]        
        dip = f_dip(alpha_cart)

        chi_cart = ip_response(flake, omegas,
                                 relaxation_rate = 0.01,
                                 os1 = flake.velocity_operator_e[:2],
                                 os2 = flake.velocity_operator_e[:2])["total"]
        
        dip2 = f_dip_j(chi_cart)

        plt.plot(omegas, dip[0], label = rf'$p_+$ {t:.2f}')
        plt.plot(omegas, dip[1], label = rf'$p_-$ {t:.2f}')
        
        plt.plot(omegas, dip2[0], label = rf'$jp_+$ {t:.2f}', ls = '--')
        plt.plot(omegas, dip2[1], label = rf'$jp_-$ {t:.2f}', ls = '--')

    plt.legend()
    plt.savefig("p.pdf")
    plt.close()
    

def get_ip_abs(flake, omegas, comp, relaxation_rate = 1e-2):

    def inner(omega):
        return jnp.trace( (delta_occ / (omega + delta_e + 1j*relaxation_rate)) @ trans)

    print("Computing Greens function. Remember we default to site basis")

    dip = flake.velocity_operator_e[:2]    
    projection = get_projection(dip)

    trans = jnp.abs(projection[comp])**2
        
    occupations = flake.initial_density_matrix_e.diagonal() * flake.electrons 
    energies = flake.energies
    delta_occ = (occupations[:, None] - occupations)
    delta_e = energies[:, None] - energies

    return jax.lax.map(jax.jit(inner), omegas)

def plot_flake_ip_cd():
    # vary shape?
    shape = Triangle(20, armchair = False)
    
    # vary?    
    delta = 1.0
    t_nn = -2.66
    
    ts = [0, 1e-5, 0.4]
    
    # omegas
    omegas = jnp.linspace(0, 8, 100)    

    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 33,
        "axes.labelsize": 33,
        "xtick.labelsize": 8*3,
        "ytick.labelsize": 8*3,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
    
        f_cd = lambda pph: (pph[0, 0].imag - pph[1, 1].imag) / ((pph[0, 0].imag + pph[1, 1].imag))
        f_cd = lambda pph: (jnp.abs(pph[0, 0].imag) - jnp.abs(pph[1, 1].imag)) / ((jnp.abs(pph[0, 0].imag) + jnp.abs(pph[1, 1].imag)))
        


        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)

            absp = get_ip_abs(flake, omegas, 0, relaxation_rate = 1e-2)
            absm = get_ip_abs(flake, omegas, 1, relaxation_rate = 1e-2)
            nom  = (absp.imag - absm.imag)
            denom = (absp.imag + absm.imag)
            cd = nom / denom

            print(nom.max(), denom.min())            
            
            ls = '-' if t > get_threshold(delta) else '--'
            plt.plot(omegas, cd, label = f'{t:.2f}', ls = ls)

            # import pdb; pdb.set_trace()


        plt.legend()
        plt.savefig("ip_cd.pdf")
        plt.close()
    
def plot_flake_cd():
    # vary shape?
    shape = Triangle(20, armchair = False)
    
    # vary?    
    delta = 1.0
    t_nn = -2.66
    
    ts = [0, 1e-5, 0.4]
    
    # omegas
    omegas = jnp.linspace(0.01, 8, 100)    

    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 33,
        "axes.labelsize": 33,
        "xtick.labelsize": 8*3,
        "ytick.labelsize": 8*3,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
    
        f_cd = lambda pph: (pph[0, 0].imag - pph[1, 1].imag) / ((pph[0, 0].imag + pph[1, 1].imag))
        f_cd = lambda pph: (jnp.abs(pph[0, 0].imag) - jnp.abs(pph[1, 1].imag)) / ((jnp.abs(pph[0, 0].imag) + jnp.abs(pph[1, 1].imag)))

        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)        
            alpha_cart = ip_response(flake, omegas, relaxation_rate = 0.01)["total"]
            alpha_circ = to_helicity(alpha_cart)
            # alpha_cart_j = ip_response(flake, omegas, os1 = flake.velocity_operator_e[:2], os2 = flake.velocity_operator_e[:2], relaxation_rate = 0.01)["total"]
            # alpha_cart_j = (alpha_cart_j - alpha_cart_j[..., 0][:, :, None]) / omegas**2
            # alpha_circ_j = to_helicity(alpha_cart_j)

            alpha_circ = to_helicity(alpha_cart)
            # f_cd = lambda xx : jnp.abs(xx.sum(axis=1)).T[:, 0] - jnp.abs(xx.sum(axis=1)).T[:, 1]
            # f_cd = lambda xx : jnp.trace(xx).imag * omegas
            cd = f_cd(alpha_circ)
            # cd = f_cd(jj)
            # import pdb; pdb.set_trace()
            print((jnp.diagonal(alpha_circ.imag) > 0).sum())

            ls = '-' if t > get_threshold(delta) else '--'
            plt.plot(omegas, cd, label = f'{t:.2f}', ls = ls)


        plt.legend()
        plt.savefig("cd.pdf")
        plt.close()
        
if __name__ == '__main__':
    # plot_projected_polarization() # DONE
    # plot_dipole_moments() # DONE
    plot_dipole_moments_sweep() 

    # plot_flake_cd()
    # plot_flake_ip_cd() # selectivity measure

    # APPENDIX
    # plot_dipole_moments_p_j() # ensure gauge invariant jj results match pp results    
