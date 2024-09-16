import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *

from lib import *

def plot_chirality(results_file, flake, display, keys=None):
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
    line_styles = ['-', '--', '-.', ':']

    # Iterate over each key to plot the corresponding data
    for i, key in enumerate(keys):
        mat = data[key]
        mat -= np.diag(mat[:, :, 0].diagonal())[:, :, None]
        mat = to_helicity(mat)
        
        # Compute the real and imaginary parts
        mat_real, mat_imag = mat.real, mat.imag
        
        # Select line style based on the presence of 'topological' in the key
        line_style = line_styles[i % len(line_styles)]        
                
        # Calculate chirality
        left = np.sort(np.abs(mat[0, :, :]), axis=0)[::-1, :]
        right = np.sort(np.abs(mat[1, :, :]), axis=0)[::-1, :]
        
        # Normalize and compute chirality
        norm = lambda x: np.linalg.norm(x, axis=0)
        chi = norm(left - right) / np.sqrt(norm(left)**2 + norm(right)**2)
        
        # Plot the chirality with a custom color and line style
        ax.plot(omegas, chi, label=key.split("_")[-1], color=colors[i], linestyle=line_style, linewidth=2)

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
    
    l = localization(flake.positions, flake.eigenvectors, flake.energies)
    m = l.argmax()
    ms = l.argsort()[::-1][:8]
    delta_e = (flake.energies[:, None] - flake.energies)
    for m in ms:
        for e in delta_e[m,:]:
            ax.axvline(jnp.abs(e), linestyle = '--')
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot as a PDF with a higher DPI for publication quality
    plt.show()

def ip(flake, A, B, omegas, mask = None):
    def inner(omega):
        mat = delta_occ / (omega + delta_e + 1j*0.1)
        return jnp.trace( mat @ operator_product)
        # mat = 1 / (omega + delta_e + 1j*0.01)
        # return jnp.trace( mat @ operator_product)

    print("Computing Greens function. Remember we default to site basis")

    operator_product =  A.T * B
    occupations = flake.initial_density_matrix_e.diagonal() * flake.electrons 
    energies = flake.energies 
    delta_occ = (occupations[:, None] - occupations)
    delta_e = flake.energies[:, None] - flake.energies

    if mask is not None:        
        delta_occ = delta_occ.at[mask].set(0) 

    return jax.lax.map(jax.jit(inner), omegas)

def to_helicity(mat):
    """converts mat to helicity basis"""
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    trafo_inv = jnp.linalg.inv(trafo)
    return jnp.einsum('ij,jmk,ml->ilk', trafo, mat, trafo_inv)


flake = get_haldane_graphene(1, -0, 0).cut_flake(Rectangle(10,20))

omegas = jnp.linspace(0, 50, 100)
delta_e = flake.energies[:, None] - flake.energies
l = localization(flake.positions, flake.eigenvectors, flake.energies)
m = l.argmax()
jx, jy = flake.velocity_operator_e[0], flake.velocity_operator_e[1]
jp = 1/jnp.sqrt(2) * (jx + 1j * jy)
jm = 1/jnp.sqrt(2) * (jx - 1j * jy)

# compute only topological sector
trivial = l < 0.5
mask = None #jnp.logical_and(trivial[:, None], trivial)
mat = [ [ ip(flake, o1, o2, omegas, mask) for o1 in [jp, jm]] for o2 in [jp, jm] ]
mat = jnp.abs(jnp.array(mat))

plt.plot(omegas, mat[0,0])
plt.plot(omegas, mat[1,1], '--')
plt.plot(omegas, mat[1,0])
plt.plot(omegas, mat[0,1], '--')
plt.show()

# mat = [ [ ip(flake, o1, o2, omegas) for o1 in [jx, jy]] for o2 in [jx, jy] ]
# mat = jnp.abs(to_helicity(jnp.array(mat)))
# plt.plot(omegas, mat[0,0])
# plt.plot(omegas, mat[1,1], '--')
# plt.plot(omegas, mat[1,0])
# plt.plot(omegas, mat[0,1], '--')
# plt.show()

# occupations = flake.initial_density_matrix_e.diagonal()
# plt.plot(jnp.arange(len(flake)), jnp.abs(jp[m,:]) * occupations)
# plt.plot(jnp.arange(len(flake)), jnp.abs(jm[m,:]) * occupations)
# plt.show()

plot_chirality("cond_" + LRT_FILE, flake, display = jnp.abs(flake.eigenvectors[:, m]) )
