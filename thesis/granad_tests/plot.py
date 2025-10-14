import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from granad import *

def plot_orbital_layout_2d(orbs, show_tags=None, circle_scale: float = 1e3, inset_ax=None):
    """
    Displays a 2D orbital layout with tagging, either as a standalone plot or an inset.

    Parameters:
        orbs: list of orbital objects with `.tag` and `.position`.
        show_tags: list or set of tags to include (defaults to all present).
        circle_scale: float scaling factor for marker size.
        inset_ax: optional matplotlib axis object (for embedding as inset).
    """
    # Select tags to show
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    # Group positions by tag
    tags_to_pos = defaultdict(list)
    for orb in orbs:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)

    # Prepare colors
    unique_tags = sorted(tags_to_pos.keys())
    cmap = plt.get_cmap('tab10')
    color_map = {
        tag: cmap(i / max(len(unique_tags) - 1, 1))
        for i, tag in enumerate(unique_tags)
    }

    # Choose axis: standalone or inset
    ax = inset_ax if inset_ax else plt.subplots(figsize=(5, 4))[1]

    # Plot
    for tag, pos_list in tags_to_pos.items():
        positions = np.array(pos_list)
        ax.scatter(
            positions[:, 0], positions[:, 1],
            s=circle_scale * 0.05,
            color=color_map[tag],
            edgecolors='k',
            alpha=0.8,
            label=str(tag)
        )

    # Format
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if not inset_ax:
        ax.set_xlabel("X (Å)")
        ax.set_ylabel("Y (Å)")
        ax.legend(title="Orbital Tags", fontsize='small', frameon=False)
        plt.tight_layout()
        plt.show()

data = jnp.load("acsnano.npz")
acsnano_absorption_ref = data["ref_data"]   # shape (N,)
acsnano_omegas_ref    = data["omegas_rpa"]      # shape (N,)
acsnano_absorption_granad    = data["absorption_td"]      # shape (N,)
acsnano_omegas_granad = data["omegas_td"]   # shape (N,)

plt.plot(acsnano_omegas_granad, acsnano_absorption_granad / jnp.max(ascnano_absorption_granad), linewidth=2, ls = '--', label = 'GRANAD' )
plt.plot(ascnano_omegas_ref, acsnano_absorption_ref, 'o', label='Experiment')
plt.xlabel(r'$\hbar\omega$', fontsize=20)
plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
plt.title('Absorption Spectrum as a Function of Photon Energy', fontsize=15)
plt.legend()


fig, ax = plt.subplots(figsize=(10, 6))
data = jnp.load("rpa_vs_td.npz")
absorption_td = data["absorption_td"]   # shape (N,)
omegas_td    = data["omegas_td"]      # shape (N,)
absorption_rpa    = data["absorption_rpa"]      # shape (N,)
omegas_rpa= data["omegas_rpa"]   # shape (N,)

# Plot RPA
ax.plot(omegas_rpa, absorption_rpa / jnp.max(absorption_rpa),
        'o', markersize=6, color='C0', alpha=0.8, 
        label='RPA')

# Plot TD
ax.plot(omegas_td, absorption_td / jnp.max(absorption_td),
        linestyle='--', linewidth=2.5, color='C1',
        label='TD')

# Labels and title
ax.set_xlabel(r'$\hbar\omega$ (eV)', fontsize=18)
ax.set_ylabel(r'$\sigma(\omega)$ (normalized)', fontsize=18)
ax.set_title('Absorption Spectrum vs Photon Energy', fontsize=20, pad=15)

# Grid & ticks
ax.grid(True, linestyle='--', alpha=0.6)
ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2)
ax.tick_params(axis='both', which='minor', length=3, width=1)

# Legend
ax.legend(fontsize=14, loc='best', frameon=True)

# Layout
plt.tight_layout()
plt.show()
