from granad import *

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
        
flake = MaterialCatalog.get("graphene").cut_flake( Hexagon(10) )

pulse = Pulse(
    amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2
)

operators = [flake.velocity_operator]

result = flake.master_equation(
    relaxation_rate = 1/10,
    illumination = pulse,
    expectation_values = operators,
    end_time = 40,
     )

# Main figure
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(result.time_axis, result.output[0][:, 0], linewidth=2, color='tab:blue')
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Current (a.u.)")
ax.set_title("Time-Dependent Current")
ax.grid(True, linestyle=':', alpha=0.6)

# Inset for 2D layout
inset_ax = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=1)
plot_orbital_layout_2d(flake, circle_scale=1e3, inset_ax=inset_ax)

# Save or show
plt.tight_layout()
plt.savefig("example_current_with_inset.pdf", dpi=300)
