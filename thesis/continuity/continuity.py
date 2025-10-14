import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from granad import *

import jax
import jax.numpy as jnp
from granad import *

import matplotlib.pyplot as plt

plt.style.use("../thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9  

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
        


flake = MaterialCatalog.get("hBN").cut_flake( Rectangle(10, 10) )
print(len(flake))
# flake.show_2d()

pulse = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)

operators = [flake.dipole_operator, flake.velocity_operator]

result = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = pulse,
    expectation_values = operators,
    end_time = 40,
     )

omega_min, omega_max = 0, 5
omegas, pulse_omega = result.ft_illumination( omega_min = omega_min, omega_max = omega_max )
output_omega = result.ft_output( omega_min = omega_min, omega_max = omega_max )[0]

# Data
p = -(omegas * output_omega[:, 0]).imag
j = output_omega[:, 3].real

# Figure
fig, ax = plt.subplots(figsize=(LATEX_TEXTWIDTH_IN * 0.55, LATEX_TEXTWIDTH_IN * 0.55))

# Plot curves
ax.plot(omegas, p / p.max(), linewidth=2.5,
        label=r'$-\,\mathrm{Im}[\omega p_x]$')
ax.plot(omegas, j / p.max(), linestyle='--', linewidth=2.5,
        label=r'$\mathrm{Re}[j_x]$')

# Labels
ax.set_xlabel(r'$\omega$ (eV)')
ax.set_ylabel(r'$j$ (a.u.)')  # added units placeholder

# Grid and ticks
# ax.tick_params(axis='both', which='major', labelsize=14, length=6, width=1.2)
# ax.tick_params(axis='both', which='minor', length=3, width=1)

# Legend
ax.legend(loc='best', frameon=True)

inset_ax = inset_axes(ax, width="35%", height="35%", loc='upper left', borderpad=1)
plot_orbital_layout_2d(flake, circle_scale=1e2, inset_ax=inset_ax)

# Layout
plt.tight_layout()
plt.savefig("continuity.pdf")
