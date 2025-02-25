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
def localization(flake):
    """Compute eigenstates edge localization"""
    # edges => neighboring unit cells are incomplete => all points that are not inside a "big hexagon" made up of nearest neighbors
    positions, states, energies = flake.positions, flake.eigenvectors, flake.energies 

    distances = jnp.round(jnp.linalg.norm(positions - positions[:, None], axis = -1), 4)
    nnn = jnp.unique(distances)[2]
    mask = (distances == nnn).sum(axis=0) < 6


    # localization => how much eingenstate 
    l = (jnp.abs(states[mask, :])**2).sum(axis = 0) # vectors are normed 

    return l

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
    """computes Wx3x3 IP polarizability according to usual lehmann representation"""
    corr = {}
    os1 = os1 if os1 is not None else flake.dipole_operator_e[:2]
    os2 = os2 if os2 is not None else flake.dipole_operator_e[:2]
    
    corr["total"] =  get_correlator(flake, omegas, os1, os2, relaxation_rate = relaxation_rate)

    if topology == True:
        l = localization(flake)
        trivial = jnp.argsort(l)[:-10] # keep only largest 10        
        print("topological states", len(flake) - trivial.sum())
        mask = jnp.logical_and(trivial[:, None], trivial)        
        corr["topological"] = get_correlator(flake, omegas, os1, os2, relaxation_rate = relaxation_rate, mask = mask)

    corr["omegas"] = omegas
    
    if results_file is not None:
        jnp.savez(results_file, **corr)
        
    return corr

### RPA ###
def rpa_susceptibility(flake, c, omegas, relaxation_rate):
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
        sus = rpa_susceptibility(flake, c, omegas, relaxation_rate = 0.05)
        
        p = flake.positions.T
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)

        # TODO: check if this is right, maybe missing omegas?
        pol.append(ref)

    if results_file is not None:
        jnp.savez(results_file, pol = pol, omegas = omegas, cs = cs)
        
    return pol

### PLOTTING ###    
def plot_2d_geometry(show_tags=None, show_index=False, scale = False, cmap = None, circle_scale : float = 1e3, title = None, mode = None, indicate_atoms = False, grid = False):

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
    
    shape = Rhomboid(40, 40, armchair = False)
    orbs = get_haldane_graphene(1.0, 1j*0.3, 1.0).cut_flake(shape)
    l = localization(orbs)
    display = jnp.abs(orbs.eigenvectors[:, l.argsort()[-4]])**2
    
    
    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        # Create plot
        fig, ax = plt.subplots()
        cmap = plt.cm.bwr if cmap is None else cmap
        colors = scale_vals(display)
        scatter = ax.scatter([orb.position[0] * scale for orb in orbs], [orb.position[1]  * scale for orb in orbs], c=colors, edgecolor='none', cmap=cmap, s = circle_scale*jnp.abs(display) * 5)
        ax.scatter([orb.position[0] * scale  for orb in orbs], [orb.position[1]  * scale  for orb in orbs], color='black', s=5, marker='o')            
        cbar = fig.colorbar(scatter, ax=ax, label = r'$|\Psi|^2$')

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
    shape = Rhomboid(20, 20, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    ts = [0.0, 0.1, 0.2]
    
    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 20,
        "axes.labelsize": 33,
        "xtick.labelsize": 8*3,
        "ytick.labelsize": 8*3,
        "legend.fontsize": 9*2,
        "pdf.fonttype": 42
    }

    labels = ["(a)", "(b)", "(c)"]  # Define labels for each subplot
    
    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
        
        # Create a figure with a 1x3 grid of subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))

        for i, t in enumerate(ts):
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
            dip = flake.velocity_operator_e[:2]
            projection = get_projection(dip)
            
            norm = None #LogNorm()
            im = axes[i].matshow(jnp.abs(projection[0])**2, norm=norm, cmap = "gist_heat_r")

            # Move x-ticks below the plot
            axes[i].xaxis.set_ticks_position("bottom")

            # Attach a colorbar on top of the matshow plot
            divider = make_axes_locatable(axes[i])
            cax = divider.append_axes("top", size="5%", pad=0.3)  # "top" places it above

            # Create colorbar with horizontal orientation
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")

            # Set label above the colorbar
            cax.xaxis.set_ticks_position("top")  # Move ticks to the top
            # axes[i].set_title(rf"$\lambda / t$ = {t:.2f}", pad = 110)
            
            axes[i].annotate(
                labels[i], xy=(-0.3, 1.5), xycoords="axes fraction",
                fontsize=22, fontweight="bold", ha="left", va="top"                
            )
            
            if i != 0:
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            else:
                axes[i].set_xlabel(r"$m$")
                axes[i].set_ylabel(r"$n$")

            if i == 1:
                cbar.set_label(r"$\vert J_+ \vert^2$ (a.u.)", fontsize=20, labelpad=-85)
                        
    plt.tight_layout()
    plt.savefig("projected_polarizations.pdf")
        
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
    shape = Rhomboid(40, 40, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = [0, 0.15, 0.4]
    ts = [0.4]
    
    # omegas
    omegas = jnp.linspace(0., 0.5, 300)    

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
        pm = find_peaks(dip[1])[2]
        pm_max, omega_m = dip[1][pm].item(), omegas[pm].item()
        
        peaks = [
            ([(omega_p, pp_max),
              (omega_p*1.2, pp_max*1)],
             r"$\vert J_{+} \vert > \vert J_{-} \vert$" ),            
            ([(omega_m, pm_max),
              (omega_m * 1.2, pm_max*1.3)],
             r"$\vert J_{-} \vert > \vert J_{+} \vert$" ),            
        ]

            
        for p in peaks:
            pos, annotation = p
            plt.annotate(annotation, xy=pos[0], xytext=pos[1], arrowprops=dict(arrowstyle="->,head_width=.15"), fontsize = 15)

        plt.legend()
        plt.tight_layout()
        plt.savefig("p.pdf")
        plt.close()

def plot_dipole_moments_sweep():
    """plots p_+ - p_- in colormap"""
    shape = Rhomboid(40, 40, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = jnp.linspace(0, 0.4, 40)
    
    # omegas
    omegas = jnp.linspace(0., 0.5, 300)    

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
        fig, ax = plt.subplots(figsize=(6, 6))  # Ensure the figure is square

        # Create the main plot
        im = ax.imshow(res.T, 
                       aspect='auto', 
                       cmap="coolwarm", 
                       origin='lower', 
                       extent=[ts.min(), ts.max(), omegas.min(), omegas.max()])


        ax.axvline(get_threshold(delta), color='k', linestyle='--', linewidth=2)

        # Axis labels
        ax.set_xlabel(r'$\lambda / t$', weight='bold')
        ax.set_ylabel(r'$\omega / t$', weight='bold')

        # Adjust colorbar size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and spacing

        # Create smaller colorbar
        cbar = plt.colorbar(im, cax=cax, label=r'$|p_+| - |p_-|$ (a.u.)')
        cbar.formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((0, 0))  # Forces scientific notation when needed
        cbar.update_ticks()

        # Save and close
        plt.savefig("p_sweep.pdf", bbox_inches='tight')
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

        plt.legend()
        plt.savefig("ip_cd.pdf")
        plt.close()
        
def plot_selectivity_sweep():
    """plots selectivity in colormap"""
    shape = Rhomboid(40, 40, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = jnp.linspace(0, 0.4, 40)
    
    # omegas
    omegas = jnp.linspace(0., 0.5, 300)    

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

    # f_cd = lambda pph: (pph[0, 0].imag - pph[1, 1].imag) / ((pph[0, 0].imag + pph[1, 1].imag))
    f_cd = lambda pph: (jnp.abs(pph[0, 0].imag) - jnp.abs(pph[1, 1].imag)) / ((jnp.abs(pph[0, 0].imag) + jnp.abs(pph[1, 1].imag)))

    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, -1j], [1, 1j] ])
    f_dip = lambda xx : jnp.abs(  jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)) )    
    res = []    
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)  
        alpha_cart = ip_response(flake, omegas, relaxation_rate = 1e-3)["total"]
        alpha_circ = to_helicity(alpha_cart)
        cd = f_cd(alpha_circ)                    
        res.append(cd)
    res = jnp.array(res)

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
        fig, ax = plt.subplots(figsize=(6, 6))  # Ensure the figure is square

        # Create the main plot
        im = ax.imshow(res.T, 
                       aspect='equal', 
                       cmap="coolwarm",
                       origin='lower', 
                       extent=[ts.min(), ts.max(), omegas.min(), omegas.max()])


        ax.axvline(get_threshold(delta), color='k', linestyle='--', linewidth=2)

        # Axis labels
        ax.set_xlabel(r'$\lambda / t$', weight='bold')
        ax.set_ylabel(r'$\omega / t$', weight='bold')

        # Adjust colorbar size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and spacing

        # Create smaller colorbar
        cbar = plt.colorbar(im, cax=cax, label=r'$s$')

        # Save and close
        plt.savefig("selectivity_sweep.pdf", bbox_inches='tight')
        plt.close()        
    
def plot_energy_localization():
    from matplotlib.ticker import MaxNLocator
    shape = Rhomboid(20, 20, armchair = False)
    ts = [0.05, 0.3]
    
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

    with mpl.rc_context(rc=custom_params):

        fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey = True)  # Shared y-axis for alignment
        
        labels = ["(a)", "(b)", "(c)"]  # Define labels for each subplot

        # Second loop for plotting with consistent color scale
        for i, ax in enumerate(axs):
            flake = get_haldane_graphene(1., 1j * ts[i], 1.).cut_flake(shape)
            
            e_max = flake.energies.max()
            e_min = flake.energies.min()
            widening = (e_max - e_min) * 0.01  # 1% margin
            e_max += widening
            e_min -= widening
            
            energies_filtered_idxs = jnp.argwhere(
                jnp.logical_and(flake.energies <= e_max, flake.energies >= e_min)
            )
            state_numbers = energies_filtered_idxs[:, 0]
            energies_filtered = flake.energies[energies_filtered_idxs]
            
            scatter = ax.scatter(
                state_numbers,
                energies_filtered,
                c=localization(flake),
                vmin=0, vmax=1,  # Ensure shared color scale
            )

            ax.set_xlabel("State Index")

            ax.annotate(
                labels[i], xy=(-0.25, 1.), xycoords="axes fraction",
                fontsize=22, fontweight="bold", ha="left", va="top"                
            )
            
            if i == 0:
                ax.set_ylabel(r"$E / t$")

            ax.axhline((e_max + e_min) / 2, color="grey", linestyle="--", alpha=0.5)
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(e_min, e_max)


            # Create a divider for the existing axis
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)  # "right" places the colorbar on the right

            # Create a colorbar for each subplot
            colorbar = plt.colorbar(scatter, cax=cax)
            colorbar.ax.set_ylabel(r"$\mathcal{L}$", rotation=270, labelpad=15)  # Rotate and space the
        
        plt.tight_layout()
        # fig.subplots_adjust(top=0.85)  # Moves subplots down slightly
        plt.savefig("energy_localization.pdf")
        plt.close()

def plot_size_sweep():
    """plots p_+ - p_- in colormap depending on size and frequency"""

    delta = 1.0
    t_nn = 1.0
    
    # omegas
    omegas = jnp.linspace(0., 0.5, 300)    

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
    res, plot_sizes = [], []
    sizes = jnp.arange(20, 100, 2)
    for s in sizes:
        shape = Rhomboid(s, s, armchair = False)
        flake = get_haldane_graphene(t_nn, 1j*0.5, delta).cut_flake(shape)  
        alpha_cart = ip_response(flake, omegas, relaxation_rate = 1e-3)["total"]
        dip = f_dip(alpha_cart)
        res.append( (dip[0] - dip[1]) / len(flake) )
        plot_sizes.append(len(flake))
    res = jnp.array(res)
    plot_sizes = jnp.array(plot_sizes)

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
        fig, ax = plt.subplots(figsize=(6, 6))  # Ensure the figure is square

        # Create the main plot
        im = ax.imshow(res.T, 
                       aspect='auto', 
                       cmap="coolwarm", 
                       origin='lower',
                       extent=[plot_sizes.min(), plot_sizes.max(), omegas.min(), omegas.max()])


        # Axis labels
        ax.set_xlabel(r'$N$', weight='bold')
        ax.set_ylabel(r'$\omega / t$', weight='bold')

        # Adjust colorbar size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and spacing

        # Create smaller colorbar
        cbar = plt.colorbar(im, cax=cax, label=r'$(|p_+| - |p_-|) / N$ (a.u.)')
        cbar.formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        cbar.formatter.set_powerlimits((0, 0))  # Forces scientific notation when needed
        cbar.update_ticks()

        # Save and close
        plt.savefig("size_sweep.pdf", bbox_inches='tight')
        plt.close()


## APPENDIX
def plot_rpa_sweep():
    """plots p_+ - p_- in colormap"""
    shape = Rhomboid(40, 40, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    # omegas
    omegas = jnp.linspace(0., 2, 300)    

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
    cs = jnp.linspace(0, 1, 10)    
    for c in cs:
        flake = get_haldane_graphene(t_nn, 1j*0.5, delta).cut_flake(shape)  
        alpha_cart = rpa_polarizability(flake, omegas, [c], relaxation_rate = 1e-3)[0][:2, :2]
        dip = f_dip(alpha_cart)
        res.append(dip[0] - dip[1])
    res = jnp.abs(jnp.array(res))

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):
        fig, ax = plt.subplots(figsize=(6, 6))  # Ensure the figure is square

        # Create the main plot
        im = ax.imshow(res.T, 
                       aspect='equal', 
                       cmap="coolwarm",
                       origin='lower',
                       norm=mpl.colors.LogNorm(),
                       extent=[cs.min(), cs.max(), omegas.min(), omegas.max()])


        # Axis labels
        ax.set_xlabel(r'Coulomb strength $c$', weight='bold')
        ax.set_ylabel(r'$\omega / t$', weight='bold')

        # Adjust colorbar size
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and spacing

        # Create smaller colorbar
        cbar = plt.colorbar(im, cax=cax, label=r'$|p_+ - p_-|$ (a.u.)')
        # cbar.formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        # cbar.formatter.set_powerlimits((0, 0))  # Forces scientific notation when needed
        cbar.formatter = mpl.ticker.LogFormatter()  # Use logarithmic tick formatting
        cbar.update_ticks()

        # Save and close
        plt.savefig("rpa_sweep.pdf", bbox_inches='tight')
        plt.close()        

        
def plot_dipole_moments_p_j():
    """plots p_+, p_- computed from xpp and xjj"""
    shape = Rhomboid(20, 20, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = [0, 0.15, 0.4]
    
    # omegas
    omegas = jnp.linspace(0., 1, 300)    

    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, -1j], [1, 1j] ])
    f_dip = lambda xx : jnp.abs(  jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)) )

    # xjj = w**2 xpp
    f_dip_j = lambda jj : f_dip( (jj - jj[..., 0][:, :, None])  )

    
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

        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)  
            alpha_cart = ip_response(flake, omegas, relaxation_rate = 1e-3)["total"]        
            dip = f_dip(alpha_cart)

            chi_cart = ip_response(flake, omegas,
                                     relaxation_rate = 1e-3,
                                     os1 = flake.velocity_operator_e[:2],
                                     os2 = flake.velocity_operator_e[:2])["total"]

            dip2 = f_dip_j(chi_cart) / omegas**2

            diff1 = dip[0] - dip[1]
            plt.plot(omegas, diff1, label = rf'$\lambda / t =$ {t:.2f}')
            
            diff2 = dip2[0] - dip2[1]
            plt.plot(omegas, diff2, label = rf'$\lambda / t =$ {t:.2f}', ls = '--')

        plt.xlabel(r'$\omega / t$')
        plt.ylabel(r'$p_+ - p_-$ (a.u.)')
        
        plt.legend()
        plt.savefig("p_trk.pdf")
        plt.close()           


def plot_dipole_moments_broken_symmetry():
    """plots p_+, p_-"""
    shape = Triangle(20, armchair = False)
    
    delta = 1.0
    t_nn = 1.0
    
    ts = [0, 0.15, 0.4]
    ts = [0.5]
    
    # omegas
    omegas = jnp.linspace(0., 0.8, 300)    

    # Define custom settings for this plot only
    custom_params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 20,
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
        
            
        plt.legend()
        plt.tight_layout()
        plt.savefig("p_broken.pdf")
        plt.close()

        
if __name__ == '__main__':
    # plot_2d_geometry() # DONE
    # plot_projected_polarization() # DONE
    # plot_dipole_moments() # DONE
    # plot_dipole_moments_sweep() # DONE
    plot_energy_localization() # DONE
    # plot_selectivity_sweep() # DONE
    # plot_size_sweep()  # DONE

    
    # APPENDIX
    # plot_dipole_moments_p_j() # DONE
    # plot_rpa_sweep() # DONE
    # plot_dipole_moments_broken_symmetry() # DONE
