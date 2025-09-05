from granad import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def show_energies_with_inset_2d(orbs, display1=None, display2=None, label=None, e_max=None, e_min=None,
                                 show_tags=None, show_index=False, scale=False, cmap=None,
                                 circle_scale: float = 1e-3, inset_loc='upper left',
                                 inset_size='35%', name : str = None):
    """
    Shows the energy spectrum with occupation and an inset showing 2D orbital layout.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from collections import defaultdict
    import numpy as np
    import jax.numpy as jnp

    # --- Setup Energy Spectrum Plot (Main Axes) ---
    energies = orbs.energies
    e_max = e_max if e_max is not None else energies.max()
    e_min = e_min if e_min is not None else energies.min()

    padding = (e_max - e_min) * 0.01
    e_max += padding
    e_min -= padding

    mask = (energies >= e_min) & (energies <= e_max)
    state_indices = jnp.arange(len(energies))[mask]
    filtered_energies = energies[mask]

    if display1 is None:
        display1 = jnp.diag(orbs.electrons * orbs.initial_density_matrix_e)
    colors = display1[mask]
    label = label or "Initial State Occupation"

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        state_indices,
        filtered_energies,
        c=colors,
        cmap="viridis",
        edgecolors='k',
        alpha=0.9
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(label, fontsize=10)

    ax.set_xlabel("Eigenstate Index", fontsize=11)
    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_ylim(e_min, e_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title("Energy Spectrum with Occupation Weight", fontsize=13)

    # --- Create Inset Axes ---
    inset_ax = ax.inset_axes([0.0, 0.6, 0.35, 0.35])  # upper left


    # --- Plot 2D Orbital Layout in Inset Axes ---
    # Same logic as show_2d, but plotting into `inset_ax`
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    tags_to_pos = defaultdict(list)
    tags_to_vals = defaultdict(list)
    tags_to_idx = defaultdict(list)

    for idx, orb in enumerate(orbs):
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idx[orb.tag].append(idx)
            if display2 is not None:
                tags_to_vals[orb.tag].append(display2[idx])

    unique_tags = sorted(tags_to_pos.keys())
    base_cmap = plt.get_cmap(cmap or 'tab10')
    tag_colors = {
        tag: base_cmap(i / max(len(unique_tags) - 1, 1))
        for i, tag in enumerate(unique_tags)
    }

    for tag in unique_tags:
        positions = np.array(tags_to_pos[tag])
        color = tag_colors[tag]

        if display2 is not None:
            values = np.array(tags_to_vals[tag])
            sizes = circle_scale * values if scale else circle_scale * jnp.ones_like(values)

            inset_ax.scatter(
                positions[:, 0], positions[:, 1],
                s=values.real * 1e2 * 5,
                c=values.real,
                cmap=cmap or 'viridis',
                alpha=0.8,
                edgecolors='k'
            )
        else:
            inset_ax.scatter(
                positions[:, 0], positions[:, 1],
                s=50,
                color=color,
                alpha=0.8,
                edgecolors='k'
            )

        if show_index:
            for idx, (x, y) in zip(tags_to_idx[tag], positions):
                inset_ax.text(x, y, str(idx), fontsize=6, ha='center', va='center')

    inset_ax.tick_params(labelsize=6)
    inset_ax.set_aspect('equal')
    inset_ax.set_xticklabels([])
    inset_ax.set_xticks([])
    inset_ax.set_yticklabels([])
    inset_ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(name)

triangle = Triangle(15)
flake = MaterialCatalog.get("graphene").cut_flake( triangle )

show_energies_with_inset_2d(
    flake,
    display1=jnp.diag(flake.electrons * flake.initial_density_matrix_e),
    display2=jnp.abs(flake.eigenvectors[:, 0]),
    circle_scale=500,  # scale for inset markers
    name = "triangle_energy_landscape.pdf"

)

triangle = Triangle(15)
del flake[flake.center_index]
show_energies_with_inset_2d(
    flake,
    display1=jnp.diag(flake.electrons * flake.initial_density_matrix_e),
    display2=jnp.abs(flake.eigenvectors[:, 0]),
    circle_scale=500,  # scale for inset markers
    name = "triangle_defect_energy_landscape.pdf"
)
