"""common utilities"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import jax.numpy as jnp

from granad import *

### UTILITIES ###
def load_data(results_file, keys):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")
    return omegas, data, data.keys() if keys is None else keys    

def to_helicity(mat):
    """converts mat to helicity basis"""
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    trafo_inv = jnp.linalg.inv(trafo)
    return jnp.einsum('ij,jmk,ml->ilk', trafo, mat, trafo_inv)

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

    cond, pol = {}, {}
    omegas = jnp.linspace(0, 6, 200)    
    for (flake, name) in args_list:        
        v, p = flake.velocity_operator_e, flake.dipole_operator_e
        cond[name] = get_correlator(v[:2])
        pol[name] = get_correlator(p[:2])

        trivial = jnp.abs(flake.energies) > 0.1
        print("edge states for", name, len(flake) - trivial.sum())

        mask = jnp.logical_and(trivial[:, None], trivial)
        
        cond["topological." + name] = get_correlator(v[:2], mask)
        pol["topological." + name] = get_correlator(p[:2], mask)
        
    cond["omegas"], pol["omegas"] = omegas, omegas
    jnp.savez("cond_" + results_file, **cond)
    jnp.savez("pol_" + results_file, **pol)

def plot_response_functions(results_file):
    """Plots j-j response directly and as obtained from p-p response with enhanced visuals."""
    with jnp.load("cond_" + results_file) as data:
        cond = dict(data)
        cond_omegas = cond.pop("omegas")        
    with jnp.load("pol_" + results_file) as data:
        pol = dict(data)
        pol_omegas = pol.pop("omegas")

    def loop_plot(func, name, title):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16, weight='bold')

        keys = cond.keys()
        for k in keys:
            if 'topo' in k:
                continue
            for i in range(2):
                for j in range(2):
                    ax = axs[i, j]
                    if i == j:
                        offset = cond[k][i, j, 0]
                    else:
                        offset = 0
                    ax.plot(
                        cond_omegas, func(cond[k][i, j] - offset),
                        label=f'cond_{k}', alpha=0.8, linewidth=2
                    )
                    ax.plot(
                        pol_omegas, pol_omegas**2 * func(pol[k][i, j]),
                        '--', label=f'pol_{k}', alpha=0.8, linewidth=2
                    )
                    ax.set_title(f'i, j = {i, j}', fontsize=12)
                    ax.grid(alpha=0.3)
                    if i == 1:
                        ax.set_xlabel("Frequency (ω)", fontsize=10)
                    if j == 0:
                        ax.set_ylabel("Response", fontsize=10)

        axs[0, 0].legend(loc="best", fontsize=8, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for the main title
        plt.savefig(name, dpi=300)  # Save with higher resolution
        plt.close()

    loop_plot(lambda x: x.imag, "imag_cond_pol_comparison.pdf", "Imaginary Part Comparison")
    loop_plot(lambda x: x.real, "real_cond_pol_comparison.pdf", "Real Part Comparison")
    loop_plot(lambda x: jnp.abs(x), "abs_cond_pol_comparison.pdf", "Absolute Value Comparison")
    
def plot_chirality(results_file, keys=None, name="chirality.pdf"):
    """
    Plots the chirality of the total response with enhanced visuals.
    
    Parameters:
    - results_file: str, path to the file containing the results.
    - keys: list of str, specific keys to plot. If None, all keys are used.
    """
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)

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

        # Set up the main plot with a larger figure size
        fig, ax = plt.subplots(figsize=(10, 7))

        # Filter out 'topological' keys
        keys = [k for k in keys if 'topological' not in k]

        # Define a custom color palette
        colors = plt.cm.tab10(np.arange(len(keys)) % 10)
        
        # Iterate over each key to plot the corresponding data
        for i, key in enumerate(keys):
            mat = data[key]
            mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
            mat = to_helicity(mat)        

            # Compute the real and imaginary parts
            mat_real, mat_imag = mat.real, mat.imag

            # Calculate chirality        
            left = np.abs(mat[0, :, :])
            right = np.abs(mat[1, ::-1, :])

            # Normalize and compute chirality
            norm = lambda x: np.linalg.norm(x, axis=0)
            chi = norm(left - right) / np.sqrt(norm(left)**2 + norm(right)**2)

            # change linestyle depending on topological or not        
            ls = '-' if float(key.split('_')[-1]) < get_threshold(1.) else '--'

            # Plot the chirality with custom color and line style
            ax.plot(
                omegas, chi,
                label=r'$t_2 =$ ' + key.split("_")[-1] + ' eV',
                color=colors[i],
                linewidth=2,
                alpha=0.85,
                ls = ls
            )

        # Add axis labels with larger fonts
        ax.set_xlabel(r'$\omega$ (eV)', weight='bold')
        ax.set_ylabel(r'$\chi$', weight='bold')


        # ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')
        # plt.tight_layout()
        # plt.subplots_adjust(right=0.75)  # Adjust space for legend

        # Modify legend placement and add a white background for readability
        ax.legend(loc="upper right", frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

        # Optionally, increase legend spacing
        ax.legend(loc="upper right", frameon=True, framealpha=0.8, edgecolor='black', facecolor='white', handlelength=2.5, labelspacing=0.5)

        ax.set_ylim(-0.05, 1)


        # Add a grid for better readability
        ax.grid(alpha=0.4)

        # Adjust tick parameters for consistency
        ax.tick_params(axis='both', which='major')

        # Optimize layout for better spacing
        plt.tight_layout()

        # Save the plot as a high-resolution PDF
        plt.savefig(name, dpi=300)
        plt.close()


def plot_chirality_topo(results_file, keys=None, name="chirality_topo.pdf"):
    """
    Plots the chirality of the total and topological response with enhanced visuals.
    
    Parameters:
    - results_file: str, path to the file containing the results.
    - keys: list of str, specific keys to plot. If None, all keys are used.
    """
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)

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
        
        # Set up the main plot with a larger figure size
        fig, ax = plt.subplots(figsize=(10, 7))

        # Iterate over each key to plot the corresponding data
        for i, key in enumerate(keys):
            mat = data[key]
            mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
            mat = to_helicity(mat)

            # Compute the real and imaginary parts
            mat_real, mat_imag = mat.real, mat.imag

            # Set line style and label
            linestyle = '-'
            label = "Total"
            if "topo" in key:
                linestyle = '--'
                label = "Topological"

            # Calculate chirality
            left = np.abs(mat[0, :, :])
            right = np.abs(mat[1, ::-1, :])

            # Normalize and compute chirality
            norm = lambda x: np.linalg.norm(x, axis=0)
            chi = norm(left - right) / np.sqrt(norm(left)**2 + norm(right)**2)

            # Plot the chirality with custom color and line style
            ax.plot(
                omegas, chi,
                label=label,
                linestyle=linestyle,
                linewidth=2,
                alpha=0.8
            )

        # Add axis labels with larger fonts
        ax.set_xlabel(r'$\omega$ (eV)', weight='bold')
        ax.set_ylabel(r'$\chi$', weight='bold')

        # Modify legend placement and add a white background for readability
        ax.legend(loc="upper right", frameon=True, framealpha=0.8, edgecolor='black', facecolor='white')

        # Optionally, increase legend spacing
        ax.legend(loc="upper right", frameon=True, framealpha=0.8, edgecolor='black', facecolor='white', handlelength=2.5, labelspacing=0.5)

        ax.set_ylim(-0.05, 1)

        # Add gridlines for better readability
        ax.grid(alpha=0.4)

        # Adjust tick parameters for consistency
        ax.tick_params(axis='both', which='major')

        # Optimize layout for better spacing
        plt.tight_layout()

        # Save the plot as a high-resolution PDF
        plt.savefig(name, dpi=300)
        plt.close()


def plot_power(results_file, keys=None):
    """
    Plots the ratio of scattered to incoming power for different EM-field helicities with enhanced visuals.
    
    Parameters:
    - results_file: str, path to the file containing the results.
    - keys: list of str, specific keys to plot. If None, all keys are used.
    """
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)
    
    # Set up the plot with 2x2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs_flat = axs.flat

    # Define custom color palette and line styles
    colors = plt.cm.plasma(np.linspace(0, 0.7, len(keys)))    
    line_styles = ['-', '--', '-.', ':']

    # Loop through each key to plot the data
    for i, key in enumerate(keys):
        mat = data[key]
        mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
        mat = to_helicity(mat)
        
        # Compute the power in x channel due to y illumination
        p = np.abs(mat)**2

        # Skip 'topological' entries
        if 'topological' in key:
            continue

        # Select line style and color
        line_style = line_styles[i % len(line_styles)]        
        color = colors[i % len(colors)]        
        
        # Normalize power by the maximum value for consistent scaling
        normalized_p = p / p.max()
        
        # Plot on corresponding subplots
        axs_flat[0].plot(
            omegas, normalized_p[0, 0, :], label=key.split("_")[-1],
            linestyle=line_style, linewidth=2, color=color, alpha=0.85
        )
        axs_flat[1].plot(
            omegas, normalized_p[1, 1, :],
            linestyle=line_style, linewidth=2, color=color, alpha=0.85
        )

    # Adding labels and titles to subplots
    ylabel = r"$\frac{P_{%s%s}}{P_{\text{max}}}$"
    titles = [
        r"Scattered power from + to +",
        r"Scattered power from - to -"
    ]
    for idx, ax in enumerate(axs_flat):
        ax.set_ylabel(ylabel % ("+" if idx < 2 else "-", "+" if idx % 2 == 0 else "-"), fontsize=14)
        ax.set_xlabel(r'$\omega$ (eV)', fontsize=14)
        ax.set_title(titles[idx], fontsize=14, pad=10)
        ax.grid(alpha=0.4)
        ax.tick_params(axis='both', which='major')

    # Add legend to the first subplot
    axs_flat[0].legend(loc="best", fontsize=12, frameon=False)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Add a main title to the entire figure
    fig.suptitle("Normalized Scattered Power Ratios", fontsize=16, weight="bold")

    # Save the plot as a PDF with high resolution
    plt.savefig("power.pdf", dpi=300)
    plt.close()


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

def plot_rpa_response(results_file):
    """
    Plots the RPA response with enhanced aesthetics and visual clarity.
    
    Parameters:
    - results_file: str, path to the file containing the RPA response data.
    """
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data["omegas"]
        cond = data["cond"][:, :2, :2, :]
        cs = data["cs"]
        

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
        
        # Set up the figure
        plt.figure(figsize=(10, 7))
        
        # Loop through each Coulomb strength and plot the response
        for i, coulomb_strength in enumerate(cs):
            mat = to_helicity(cond[i])

            # Calculate left and right-handed responses
            left = np.abs(mat[0, :, :])
            right = np.abs(mat[1, ::-1, :])

            # Compute chirality
            n = lambda x: jnp.linalg.norm(x, axis=0)
            chi = n(left - right) / jnp.sqrt(n(left)**2 + n(right)**2)

            # Plot the response
            plt.plot(
                omegas, chi, label=fr'$\lambda = {coulomb_strength}$',
                linewidth=2, alpha=0.85
            )

        # Add axis labels
        plt.xlabel(r'$\omega$ (eV)', weight='bold')
        plt.ylabel(r'$\chi$', weight='bold')

        # Add a legend
        plt.legend(loc="best", frameon=False)

        # Add gridlines for clarity
        plt.grid(alpha=0.4)

        # Optimize layout and save the plot as a high-resolution PDF
        plt.tight_layout()
        plt.savefig("rpa.pdf", dpi=300)
        plt.close()

if __name__ == '__main__':
    
    LRT_FILE = 'lrt.npz'
    RPA_FILE = 'rpa_triangle_2.npz'

    # figure chirality
    plot_rpa_response(RPA_FILE)
    plot_chirality_topo("cond_" + LRT_FILE, keys = ['topological.haldane_graphene_0.4', 'haldane_graphene_0.4'] )
    plot_chirality("cond_" + LRT_FILE)
    1/0
    
    IP_ARGS = []
    for (t2, delta) in [(0.0, 0.0), (0.01, 1), (0.05, 1), (0.2, 1), (0.4, 1)] :
        flake = get_haldane_graphene(-2.66, -1j*t2, delta).cut_flake(Triangle(42, armchair = True))
        flake.t2 = t2
        flake.trivial = bool(flake.t2 < get_threshold(delta))
        print(len(flake))
        name = f"haldane_graphene_{t2}"
        IP_ARGS.append( (flake, name) )

    print(IP_ARGS[0][0])

    RPA_FLAKE = get_haldane_graphene(-2.66, -0.5j, 0.3).cut_flake(Triangle(42))
    RPA_VALS = [0, 0.1, 0.5, 1.0]

    ip_response(IP_ARGS, LRT_FILE)

    # figure chirality
    plot_chirality("cond_" + LRT_FILE)

    # figure example geometry
    flake = IP_ARGS[-1][0]
    idx = jnp.abs(flake.energies).argmin().item()
    flake.show_2d(display = flake.eigenvectors[:, idx], scale = True, name = 'geometry.pdf')

    # figure contribution of topological state
    plot_chirality_topo("cond_" + LRT_FILE, keys = ['topological.haldane_graphene_0.4', 'haldane_graphene_0.4'] )

    # fig RPA
    rpa_response(RPA_FLAKE, RPA_FILE, RPA_VALS)
    plot_rpa_response(RPA_FILE)

    # fig sensor
    plot_power("cond_" + LRT_FILE)

    # fig appendix gauge invariance
    plot_response_functions(LRT_FILE)
