import matplotlib.pyplot as plt

import jax.numpy as jnp

from granad import *
from granad._numerics import bare_susceptibility_function
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from collections import defaultdict
import jax.numpy as jnp

plt.style.use("../thesis.mplstyle")
LATEX_TEXTWIDTH_IN = 5.9  

# --- your existing dependency ---
# get_dip(name, omega_max, omega_min, omega_0) -> (omegas, p)

def show_2d(
    flake,
    show_tags=None,
    show_index=False,
    display=None,
    scale=False,
    cmap=None,
    circle_scale: float = 1e3,
    title=None,
    mode=None,
    indicate_atoms=False,
    grid=False,
    ax=None,
    fig=None,
):
    """
    Same as your original show_2d, but will draw onto a provided Axes (ax) if given.
    This makes it composable for multi-panel layouts.
    """
    # decider whether to take abs val and normalize
    def scale_vals(vals):
        return jnp.abs(vals) / jnp.abs(vals).max() if scale else vals

    # Determine which tags to display
    if show_tags is None:
        show_tags = {orb.tag for orb in flake}
    else:
        show_tags = set(show_tags)

    # Prepare data structures for plotting
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for orb in flake:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(flake.index(orb))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True
    if fig is None:
        fig = ax.figure

    if display is not None:
        cmap = plt.cm.bwr if cmap is None else cmap
        if mode == 'two-signed':
            display = display.real
            dmax = jnp.max(jnp.abs(display))
            scatter = ax.scatter(
                [orb.position[0] for orb in flake],
                [orb.position[1] for orb in flake],
                c=display,
                edgecolor='black',
                cmap=cmap,
                s=circle_scale / 10,
            )
            scatter.set_clim(-dmax, dmax)
        elif mode == 'one-signed':
            cmap = plt.cm.Reds
            display = display.real
            dmax = display[jnp.argmax(jnp.abs(display))]
            scatter = ax.scatter(
                [orb.position[0] for orb in flake],
                [orb.position[1] for orb in flake],
                c=display,
                edgecolor='black',
                cmap=cmap,
                s=circle_scale / 10,
            )
            if dmax < 0:
                scatter.set_clim(dmax, 0)
            else:
                scatter.set_clim(0, dmax)
        else:
            colors = scale_vals(display)
            scatter = ax.scatter(
                [orb.position[0] for orb in flake],
                [orb.position[1] for orb in flake],
                c=colors,
                edgecolor='black',
                cmap=cmap,
                s=circle_scale * jnp.abs(display),
            )
        if indicate_atoms:
            ax.scatter(
                [orb.position[0] for orb in flake],
                [orb.position[1] for orb in flake],
                color='black',
                s=10,
                marker='o',
            )
        fig.colorbar(scatter, ax=ax)
    else:
        # Color by tags if no display is given
        unique_tags = list(set(orb.tag for orb in flake))
        color_map = {
            tag: plt.cm.get_cmap('tab10')(i / len(unique_tags))
            for i, tag in enumerate(unique_tags)
        }
        for tag, positions in tags_to_pos.items():
            positions = jnp.array(positions)
            ax.scatter(
                positions[:, 0],
                positions[:, 1],
                label=tag,
                color=color_map[tag],
                edgecolor='white',
                alpha=0.7,
            )
        # ax.legend(title='Orbital Tags')

    # Optionally annotate points with their indexes
    if show_index:
        for orb in [orb for orb in flake if orb.tag in show_tags]:
            pos = orb.position
            idx = flake.index(orb)
            ax.annotate(
                str(idx),
                (pos[0], pos[1]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
            )

    # Finalize plot settings
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.grid(grid)
    ax.axis('equal')

    # Return fig/ax for composition
    return fig, ax


def plot_omega_dipole_ax(name, omega_max, omega_min, omega_0, ax=None):
    """
    Spectrum plotter that draws onto a provided Axes.
    """
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)

    created_fig = False
    fig = ax.figure

    ax.semilogy(omegas / omega_0, p, linewidth=2.0)

    ax.set_xlabel(r'$\omega / \omega_0$')
    ax.set_ylabel('Dipole Strength (a.u.)')
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Add dashed vlines at 1, 2, 3, 4, 5
    for xpos in [1, 2, 3, 4, 5]:
        ax.axvline(x=xpos, color="brown", linestyle="--", linewidth=1)

    if created_fig:
        fig.tight_layout()

    return fig, ax


def plot_geometry_and_spectrum(
    flake,
    name,
    omega_max,
    omega_min,
    omega_0,
    *,
    show_tags=None,
    show_index=False,
    display=None,
    scale=False,
    cmap=None,
    circle_scale: float = 1e3,
    geom_mode=None,
    indicate_atoms=False,
    grid=False,
    figsize=(12, 5),
    savepath=None,
):
    """
    Create a side-by-side figure:
      A: geometry (from show_2d)
      B: spectrum (from plot_omega_dipole), with panel labels.

    Args mirror your originals with a few renames:
      - geom_mode corresponds to `mode` in show_2d (to avoid name clash with matplotlib).
      - savepath: if provided, saves the combined figure (e.g., 'figure.pdf' or 'figure.png').
    """
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(LATEX_TEXTWIDTH_IN, LATEX_TEXTWIDTH_IN * 0.45))

    # Left panel: geometry
    show_2d(
        flake,
        show_tags=show_tags,
        show_index=show_index,
        display=display,
        scale=scale,
        cmap=cmap,
        circle_scale=circle_scale,
        mode=geom_mode,
        indicate_atoms=indicate_atoms,
        grid=grid,
        ax=axA,
        fig=fig,
    )

    # Right panel: spectrum
    plot_omega_dipole_ax(name, omega_max, omega_min, omega_0, ax=axB)

    fig.tight_layout(w_pad=2.0)

    fig.canvas.draw_idle()  # make sure positions are up-to-date

    pos_a = axA.get_position()
    pos_b = axB.get_position()

    common_y = max(pos_a.y1, pos_b.y1) + 0.01  # a little above the taller axes
    offset = 0.01  # leftward offset

    fig.text(pos_a.x0 - offset, common_y, "(a)",
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)

    fig.text(pos_b.x0 - offset, common_y, "(b)",
             ha="right", va="bottom", fontsize=11, fontweight="bold",
             transform=fig.transFigure)


    if savepath:
        fig.savefig(savepath, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm):
    time = jnp.linspace(0, end_time, 1000)
    pulse = Pulse(amplitudes, omega, peak, fwhm)
    e_field = jax.vmap(pulse)(time)
    
    result = TDResult.load(name)        

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, e_field.real, linewidth=2.0, label='Driving Field (Re)')
    # ax.plot(time, e_field.imag, '--', linewidth=2.0, label='Driving Field (Im)')
    ax.plot(result.time_axis, result.output[0], '--', linewidth=2.0, label='Dipole Response')

    ax.set_xlabel('Time (a.u.)', fontsize=18)
    ax.set_ylabel('Amplitude (a.u.)', fontsize=18)
    ax.set_title('Time-Domain Dipole Moment', fontsize=20, pad=15)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14, loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(f"t_dipole_moment_{name}.pdf")
    plt.close()

def get_dip(name, omega_max, omega_min, omega_0):
    """returns induced dipole moment, normalized to its value at omega_0
    and omega axis.
    """
    
    result = TDResult.load(name)
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas, _ = result.ft_illumination( omega_max, omega_min )
    closest_index = jnp.argmin(jnp.abs(omegas - omega_0))
    p_0 = 1.0#jnp.linalg.norm(p_omega[closest_index])
    p_normalized = jnp.linalg.norm(p_omega, axis = -1) / p_0
    return omegas, p_normalized

flake = MaterialCatalog.get("graphene").cut_flake(Triangle(45, armchair = True))
flake.set_electrons(flake.electrons + 2)
flake.show_energies(name = "energies")

name, end_time, amplitudes, omega, peak, fwhm = "cox_50_1e-4_new", 700, [0.03, 0, 0], 0.68, 0.659 * 200, 0.659 * 166
    
# result = flake.master_equation(
#     dt = 1e-4,
#     end_time = end_time,
#     relaxation_rate = 1/10,
#     expectation_values = [ flake.dipole_operator ],
#     illumination = Pulse(amplitudes, omega, peak, fwhm),
#     max_mem_gb = 50,
#     grid = 100
# )
# result.save(name)        
# plot_omega_dipole_ax(name, 6*omega, 0, omega)
fig = plot_geometry_and_spectrum(
    flake,
    name=name,
    omega_max=6*omega, omega_min=0, omega_0=omega,
    display=None, geom_mode='two-signed', indicate_atoms=True,
    savepath="hhg.pdf"
)
# plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm)
