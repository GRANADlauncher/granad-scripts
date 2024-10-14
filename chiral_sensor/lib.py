"""common utilities"""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import jax.numpy as jnp

from granad import *


### UTILITIES ###
def localization(positions, states, energies, uniform = False):
    """Compute eigenstates edge localization"""
    # edges => neighboring unit cells are incomplete => all points that are not inside a "big hexagon" made up of nearest neighbors 
    distances = jnp.round(jnp.linalg.norm(positions - positions[:, None], axis = -1), 4)
    nnn = jnp.unique(distances)[2]
    mask = (distances == nnn).sum(axis=0) < 6

    # localization => how much eingenstate 
    l = (jnp.abs(states[mask, :]).sum(axis = 0) / jnp.abs(states).sum(axis = 0))**2

    if uniform:
        return l, mask.nonzero()[0].size / mask.size

    return l    

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


### RESPONSE ###
def ip_response(args_list, results_file):
    """computes IP j-j and p-p response. saves j-j, pp in "cond_" + results_file, "pol_" + results_file.
    """

    def get_correlator(operators, mask = None):
        return jnp.array([[flake.get_ip_green_function(o1, o2, omegas, relaxation_rate = 0.1, mask = mask) for o1 in operators] for o2 in operators])

    cond, pol = {}, {}
    omegas = jnp.linspace(0, 20, 200)    
    for (flake, name) in args_list:        
        v, p = flake.velocity_operator_e, flake.dipole_operator_e
        cond[name] = get_correlator(v[:2])
        pol[name] = get_correlator(p[:2])

        # compute only topological sector => max localization
        trivial = jnp.abs(localization(flake.positions, flake.eigenvectors, flake.energies)) < 0.4
        mask = jnp.logical_and(trivial[:, None], trivial)
        
        cond["topological." + name] = get_correlator(v[:2], mask)
        pol["topological." + name] = get_correlator(p[:2], mask)
        
    cond["omegas"], pol["omegas"] = omegas, omegas
    jnp.savez("cond_" + results_file, **cond)
    jnp.savez("pol_" + results_file, **pol)


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
        if 'topo' in k:
            continue
        for i in range(2):
            for j in range(2):
                if i == j:
                    offset = cond[k][i,j, 0].imag
                else:
                    offset = 0
                axs[i, j].plot(cond_omegas, cond[k][i, j].imag - offset, label='cond_' + k)
                axs[i, j].plot(pol_omegas, pol_omegas**2 * pol[k][i, j].imag, '--', label='pol_' + k)
                axs[i, j].set_title(f'i,j = {i,j}')
                
    axs[0, 0].legend(loc="upper left")
    plt.savefig("cond_pol_comparison_imag.pdf")
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    keys = cond.keys()
    for k in keys:
        if 'topo' in k:
            continue
        for i in range(2):
            for j in range(2):
                if i == j:
                    offset = cond[k][i,j, 0].real
                else:
                    offset = 0
                axs[i, j].plot(cond_omegas, cond[k][i, j].real - offset, label='cond_' + k)
                axs[i, j].plot(pol_omegas, pol_omegas**2 * pol[k][i, j].real, '--', label='pol_' + k)
                axs[i, j].set_title(f'i,j = {i,j}')
                
    axs[0, 0].legend(loc="upper left")
    plt.savefig("cond_pol_comparison_real.pdf")
    plt.close()
    
def plot_chirality_difference(results_file, keys = None):
    """plots excess chirality of the total response"""
    omegas, data, keys = load_data(results_file, keys)

    fig, axs = plt.subplots(2,1)    
    axs_flat = list(axs.flat)

    # Loop through each key to plot the data
    for i, key in enumerate(keys):
        mat = data[key]
        mat -= jnp.diag(mat[:, :, 0].diagonal())[:, :, None]
        mat = to_helicity(mat)
        mat_real, mat_imag = mat.real, mat.imag

        ls = '-'
        if 'topological' in key:            
            ls = '--'
        
        idx = 0
        left, right = jnp.array([[0, 0], [0, 1]]), jnp.array([[1, 0], [0, 0]])
        norms = jnp.linalg.norm(mat, axis = (0,1))
        
        diff_left = jnp.linalg.norm(jnp.einsum('ij,jlk->ilk', left, mat), axis = (0,1)) / norms 
        diff_right = jnp.linalg.norm(jnp.einsum('ij,jlk->ilk', right, mat), axis = (0,1)) / norms
                
        axs_flat[0].plot(omegas[idx:], diff_left[idx:], label=key.split("_")[-1], ls = ls)
        axs_flat[1].plot(omegas[idx:], diff_right[idx:], label=key.split("_")[-1], ls = ls)

    # Adding titles and labels to make it clear
    axs_flat[0].set_ylabel(r'$\chi_{jj, +}$')
    axs_flat[1].set_xlabel(r'$\omega (eV)$')
    axs_flat[1].set_ylabel(r'$\chi_{jj, -}$')    
    plt.legend(loc="upper left")

    # Adjusting layout and saving
    plt.tight_layout()
    plt.savefig("chirality_difference.pdf")
    plt.close()    


def plot_chirality(results_file, flake, display, keys=None, name = "chirality.pdf"):
    """
    Plots the chirality of the total response with an inset plot corresponding to the topological state in `flake`.
    
    Parameters:
    - results_file: str, path to the file containing the results.
    - keys: list of str, specific keys to plot. If None, all keys are used.
    """
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)
    
    # Set up the main plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.style.use('seaborn-darkgrid')  # Optional: use a specific style for better aesthetics

    keys = [k for k in keys if not 'topological' in k]

    # Custom color palette and line styles
    colors = plt.cm.plasma(np.linspace(0, 0.7, len(keys)))

    # Iterate over each key to plot the corresponding data
    for i, key in enumerate(keys):
        
        mat = data[key]
        mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
        mat = to_helicity(mat)
        
        # Compute the real and imaginary parts
        mat_real, mat_imag = mat.real, mat.imag
        
        # Select line style based on the presence of 'topological' in the key
        line_style = ['-', '--', '-.', ':'][i % 4]
            
        # Calculate chirality
        left = np.sort(np.abs(mat[0, :, :]), axis=0)[::-1, :]
        right = np.sort(np.abs(mat[1, :, :]), axis=0)[::-1, :]
        
        # Normalize and compute chirality
        norm = lambda x: np.linalg.norm(x, axis=0)
        chi = norm(left - right) / np.sqrt(norm(left)**2 + norm(right)**2)
        
        # Plot the chirality with a custom color and line style
        ax.plot(omegas, chi, label=key.split("_")[-1] + app, color=colors[i], linestyle=line_style, linewidth=2)

    # Adding axis labels with larger fonts for readability
    ax.set_xlabel(r'$\omega$ (eV)', fontsize=18)
    ax.set_ylabel(r'$\chi$', fontsize=18)
    
    # Adding a legend with larger font size
    ax.legend(loc="upper left", fontsize=14)
    
    # Adjusting tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Create an inset axis
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper right")  # Adjust size and position
    show_2d(ax_inset, flake, display = display) 

    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot as a PDF with a higher DPI for publication quality
    plt.savefig(name, dpi=300)
    plt.close()


def plot_chirality_topo(results_file, flake, display, keys=None, name = "chirality_topo.pdf"):
    """
    Plots the chirality of the total response with an inset plot corresponding to the topological state in `flake`.
    
    Parameters:
    - results_file: str, path to the file containing the results.
    - keys: list of str, specific keys to plot. If None, all keys are used.
    """
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)
    
    # Set up the main plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # plt.style.use('seaborn-darkgrid')  # Optional: use a specific style for better aesthetics

    # Iterate over each key to plot the corresponding data
    for i, key in enumerate(keys):
        
        mat = data[key]
        mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
        mat = to_helicity(mat)
        
        # Compute the real and imaginary parts
        mat_real, mat_imag = mat.real, mat.imag
        
        line_style = '--' if 'topo' in key else '--'
        app = '_topo' if 'topo' in key else ''
            
        # Calculate chirality
        left = np.sort(np.abs(mat[0, :, :]), axis=0)[::-1, :]
        right = np.sort(np.abs(mat[1, :, :]), axis=0)[::-1, :]
        
        # Normalize and compute chirality
        norm = lambda x: np.linalg.norm(x, axis=0)
        chi = norm(left - right) / np.sqrt(norm(left)**2 + norm(right)**2)
        
        # Plot the chirality with a custom color and line style
        ax.plot(omegas, chi, label=key.split("_")[-1] + app, linestyle=line_style, linewidth=2)

    # Adding axis labels with larger fonts for readability
    ax.set_xlabel(r'$\omega$ (eV)', fontsize=18)
    ax.set_ylabel(r'$\chi$', fontsize=18)
    
    # Adding a legend with larger font size
    ax.legend(loc="upper left", fontsize=14)
    
    # Adjusting tick parameters for better readability
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Create an inset axis
    ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper right")  # Adjust size and position
    show_2d(ax_inset, flake, display = display) 

    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot as a PDF with a higher DPI for publication quality
    plt.savefig(name, dpi=300)
    plt.close()


def plot_power(results_file, keys=None):
    """
    Plots the ratio of scattered to incoming power for different EM-field helicities.
    
    Parameters:
    - results_file: str, path to the file containing the results.
    - keys: list of str, specific keys to plot. If None, all keys are used.
    """
    
    # Load data
    omegas, data, keys = load_data(results_file, keys)
    
    # Set up the plot with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs_flat = axs.flat
    
    colors = plt.cm.plasma(np.linspace(0, 0.7, len(keys)))    
    line_styles = ['-', '--', '-.', ':']
    
    # plt.style.use('seaborn-darkgrid')  # Optional: use a specific style for better aesthetics

    # Loop through each key to plot the data
    for i, key in enumerate(keys):
        mat = data[key]
        mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
        mat = to_helicity(mat)
        
        # Compute the power in x channel due to y illumination
        p = np.abs(mat)**2

        # Line style based on the presence of 'topological' in the key
        if 'topological' in key:
            continue
        
        # Select line style based on the presence of 'topological' in the key
        line_style = line_styles[i % len(line_styles)]        
        color = colors[i % len(colors)]        
        
        # Normalizing power by the maximum value for plotting
        normalized_p = p / p.max()
        
        # Plot on corresponding subplot
        axs_flat[0].plot(omegas, normalized_p[0, 0, :], label=key.split("_")[-1], linestyle=line_style, linewidth=2, color = color)
        axs_flat[1].plot(omegas, normalized_p[0, 1, :], linestyle=line_style, linewidth=2, color = color)
        axs_flat[2].plot(omegas, normalized_p[1, 0, :], linestyle=line_style, linewidth=2, color = color)
        axs_flat[3].plot(omegas, normalized_p[1, 1, :], linestyle=line_style, linewidth=2, color = color)

    # Adding labels to each subplot
    axs_flat[0].set_ylabel(r"$\frac{P_{++}}{P_{\text{max}}}$", fontsize=14)
    axs_flat[1].set_ylabel(r"$\frac{P_{+-}}{P_{\text{max}}}$", fontsize=14)
    axs_flat[2].set_ylabel(r"$\frac{P_{-+}}{P_{\text{max}}}$", fontsize=14)
    axs_flat[3].set_ylabel(r"$\frac{P_{--}}{P_{\text{max}}}$", fontsize=14)

    axs_flat[0].set_xlabel(r'$\omega$ (eV)', fontsize=14)
    axs_flat[1].set_xlabel(r'$\omega$ (eV)', fontsize=14)
    axs_flat[2].set_xlabel(r'$\omega$ (eV)', fontsize=14)
    axs_flat[3].set_xlabel(r'$\omega$ (eV)', fontsize=14)
    
    # Set titles for each subplot (optional, if needed for clarity)
    axs_flat[0].set_title(r'Scattered power from + to +', fontsize=12)
    axs_flat[1].set_title(r'Scattered power from + to -', fontsize=12)
    axs_flat[2].set_title(r'Scattered power from - to +', fontsize=12)
    axs_flat[3].set_title(r'Scattered power from - to -', fontsize=12)

    # Only add the legend to the first subplot
    axs_flat[0].legend(loc="upper right", fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the plot as a PDF with high resolution
    plt.savefig("power.pdf", dpi=300)
    plt.close()


### RPA ###
def rpa_sus(evs, omegas, occupations, energies, coulomb, electrons, relaxation_rate = 1e-1):
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
       
    omegas =  jnp.linspace(0, 15, 150)
    res = []
    
    for c in cs:        
        sus = rpa_sus(flake.eigenvectors, omegas, flake.initial_density_matrix_e.diagonal(), flake.energies, c*flake.coulomb, flake.electrons)        
        p = flake.positions.T        
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)        
        res.append(omegas[None, None, :]**2 * ref)
        
    jnp.savez(results_file, cond = res, omegas = omegas, cs = cs)

def plot_rpa_response(results_file):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data["omegas"]
        cond = data["cond"][:, :2, :2, :]
        cs = data["cs"]
        
    for i, coulomb_strength in enumerate(cs):
        mat = to_helicity(cond[i])

        left = jnp.abs(mat[0, :, :]).sort(axis = 0)[::-1, :]
        right = jnp.abs(mat[1, :, :]).sort(axis = 0)[::-1, :]

        n = lambda x : jnp.linalg.norm(x, axis = 0) 
        chi = n(left - right) / jnp.sqrt(n(left)**2 + n(right)**2)

        plt.plot(omegas, chi, label = fr'$\lambda$ = {coulomb_strength}')

    plt.xlabel(r'$\omega (eV)$')
    plt.ylabel(r'$\delta_{+-}$')
    plt.legend(loc = "upper left")
    plt.savefig("rpa.pdf")
    plt.close()


### MEAN FIELD ###
def rho(es, vecs, thresh):
    """constructs the open-shell density matrix"""
    d = jnp.where(es <= thresh, 1, 0)
    return jnp.einsum('ij,j,kj->ik', vecs, d, vecs.conj())

def scf_loop(flake, U, mixing, limit, max_steps):
    """performs open-shell scf calculation

    Returns:
        rho_up, rho_dow, ham_eff_up, ham_eff_down
    """
    
    def update(arg):
        """scf update"""
        
        rho_old_up, rho_old_down, step, error = arg

        # H = H_+ + H_-
        ham_eff_up =  ham_0 + U * jnp.diag(jnp.diag(rho_old_down))        
        ham_eff_down =  ham_0 + U * jnp.diag(jnp.diag(rho_old_up))

        # diagonalize
        vals_up, vecs_up = jnp.linalg.eigh(ham_eff_up)
        vals_down, vecs_down = jnp.linalg.eigh(ham_eff_down)    

        # build new density matrices
        thresh = jnp.concatenate([vals_up, vals_down]).sort()[N]
        rho_up = rho(vals_up, vecs_up, thresh) + mixing * rho_old_up
        rho_down = rho(vals_down, vecs_down, thresh) + mixing * rho_old_down

        # update breaks
        error = ( jnp.linalg.norm(rho_up - rho_old_up) +  jnp.linalg.norm(rho_down - rho_old_down) ) / 2

        step = jax.lax.cond(error <= limit, lambda x: step, lambda x: step + 1, step)

        return rho_up, rho_down, step, error
    
    def step(idx, res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= limit, lambda x: res, update, res)

    ham_0 = flake.hamiltonian
    
    # GRANAD gives a closed-shell hamiltonian => for hubbard model, we split it into 2 NxN matrices, one for each spin component
    N, _ = ham_0.shape

    # initial guess for the density matrices
    rho_old_up = jnp.zeros_like(ham_0)
    rho_old_down = jnp.zeros_like(ham_0)

    # scf loop
    rho_up, rho_down, steps, error = jax.lax.fori_loop(0, max_steps, step, (rho_old_up, rho_old_down, 0, jnp.inf))
    
    print(f"{steps} / {max_steps}")

    return (rho_up,
            rho_down,
            ham_0 + U * jnp.diag(jnp.diag(rho_down)),
            ham_0 + U * jnp.diag(jnp.diag(rho_up)))

def plot_stability(results_file):
    """loads scf results, plots energy landscape, current directionality"""
    
    with jnp.load(results_file) as data:
        data = dict(data)
        res = data["res"]
        Us = data["Us"]
        pos = data["pos"]

    # energy landscapes
    for i, U in enumerate(Us):
        rho_up, rho_down, h_up, h_down = res[i]        
        v_up, vecs_up = jnp.linalg.eigh(h_up)
        v_down, vecs_down = jnp.linalg.eigh(h_down)        
        c_up = jnp.diagonal(vecs_up.conj().T @ rho_up @ vecs_up)
        c_down = jnp.diagonal(vecs_down.conj().T @ rho_down @ vecs_down)
        
        # plot_localization(pos, vecs_up, v_up, f"{U}_up.pdf")
        # plot_localization(pos, vecs_down, v_down, f"{U}_down.pdf")

    # current
    l = []
    for i, U in enumerate(Us):
        rho_up, rho_down, h_up, h_down = res[i]        
        v_up, vecs_up = jnp.linalg.eigh(h_up)
        
        l_up = localization(pos, vecs_up, v_up).max()
        l_down = localization(pos, vecs_down, v_down).max()

        l.append( (l_up + l_down) / 2)
            
    plt.plot(Us, l, '.')
    plt.xlabel('U (eV)')
    plt.ylabel(r"$\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$")
    plt.savefig("scf_localization.pdf")
    plt.close()      


### GEOMETRIES ###
def show_2d(ax, orbs, show_tags=None, show_index=False, display = None, scale = False, cmap = None, circle_scale : float = 2*1e2, title = None):
    # decider whether to take abs val and normalize 
    def scale_vals( vals ):
        return jnp.abs(vals) / jnp.abs(vals).max() if scale else vals
    
    # Determine which tags to display
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    # Prepare data structures for plotting
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for orb in orbs:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(orbs.index(orb))

    cmap = plt.cm.bwr if cmap is None else cmap
    colors = scale_vals(display)
    scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], c=colors, edgecolor='black', cmap=cmap, s = circle_scale*jnp.abs(display) )
    # ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], color='black', s=1, marker='o')
    # cbar = fig.colorbar(scatter, ax=ax)

    # Optionally annotate points with their indexes
    if show_index:
        for orb in [orb for orb in orbs if orb.tag in show_tags]:
            pos = orb.position
            idx = orbs.index(orb)
            ax.annotate(str(idx), (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Finalize plot settings
    # plt.title('Orbital positions in the xy-plane' if title is None else title)
    ax.grid(True)
    ax.axis('equal')
    ax.axis('off')
    ax.text(0.7, 0.9, r'$|\psi_{\text{edge}}|^2, t_2 = -\frac{i}{2}$', fontsize=10, ha='center', transform=ax.transAxes)
    return scatter
        

def plot_edge_states_energy_landscape(setups):

    fig, axs = plt.subplots(3,2)
    axs_flat = list(axs.flat)
    
    for i, s in enumerate(setups):        
        flake = get_haldane_graphene(*s[1:4]).cut_flake(s[0])
        
        loc = localization(flake.positions, flake.eigenvectors, flake.energies)

        sc1 = show_2d(axs_flat[2*i], flake, display = jnp.abs(flake.eigenvectors[:, loc.argmax()]) )
        axs_flat[2*i].set_xlabel('X')
        axs_flat[2*i].set_ylabel('Y')

        cb1 = fig.colorbar(sc1, ax=axs_flat[2*i])
        cb1.set_label(r'$|\psi|^2$')


        sc2 = axs_flat[2*i + 1].scatter(
            jnp.arange(flake.energies.size),
            flake.energies,
            c=loc)
        axs_flat[2*i+1].set_xlabel('# eigenstate')
        axs_flat[2*i+1].set_ylabel('E (eV)')

        cb2 = fig.colorbar(sc2, ax=axs_flat[2*i + 1])
        cb2.set_label(r"$\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$")  


    plt.tight_layout()
    plt.savefig("edge_states_energy_landscape.pdf")
    plt.close()

def plot_localization_varying_hopping(setups):
    
    fig, axs = plt.subplots(3,1)
    axs_flat = list(axs.flat)

    def loc(shape, n, nn, delta):
        flake = get_haldane_graphene(n, nn, delta).cut_flake(shape)
        l = localization(flake.positions, flake.eigenvectors, flake.energies)
        return jnp.max(l)

    nns = jnp.linspace(0, 0.5, 10)
    for i, s in enumerate(setups):        
        locs = [loc(s[0], s[1], 1j*nn, s[3]) for nn in nns]
        axs[i].plot(nns, locs)

        flake = get_haldane_graphene(*s[1:4]).cut_flake(s[0])
        _, v = localization(flake.positions, flake.eigenvectors, flake.energies, uniform = True)
        
        axs[i].axhline(y=v, ls='--', label = 'uniform')
        axs[i].axvline(x=0.03, c = 'r', label = 'bulk transition')

        axs[i].set_xlabel(r'$t_2$')
        axs[i].set_ylabel(r'$\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$')

    axs[i].legend()
    plt.tight_layout()
    plt.savefig("localization_varying_hopping.pdf")
    plt.close()
