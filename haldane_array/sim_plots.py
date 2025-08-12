# relevant scales:
# length in nm, energies in eV, hbar = 1
import jax
import jax.numpy as jnp

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm, SymLogNorm

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
                [0, 0, 0, delta/2], # onsite                
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
                [0, 0, 0, -delta/2], # onsite                
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

def rpa_polarizability(flake, omegas, cs, relaxation_rate, results_file = None):
    """computes RPA polarizability, following https://pubs.acs.org/doi/10.1021/nn204780e"""
    pol = []    
    for c in cs:
        # sus is sandwiched like x * sus * x
        sus = rpa_susceptibility(flake, c, omegas, relaxation_rate)
        
        p = flake.positions.T
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)

        # TODO: check if this is right, maybe missing omegas?
        pol.append(ref)

    if results_file is not None:
        jnp.savez(results_file, pol = pol, omegas = omegas, cs = cs)
        
    return pol

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

### PLOTTING ###    
# ==== THESIS PLOTTING THEME (new) ====
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm, SymLogNorm
import jax.numpy as jnp

# Small helper: consistent, publication-friendly style without requiring LaTeX.
_THESE_RC = {
    "text.usetex": False,          # use MathText (avoids TeX dependency)
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "axes.linewidth": 1.2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,            # editable text in vector outputs
    "figure.dpi": 120,
}

def _theme():
    return mpl.rc_context(rc=_THESE_RC)

def _beautify_axes(ax, xlabel=None, ylabel=None):
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.tick_params(direction="in", length=5, width=1, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_alpha(0.7)

def _add_cbar(fig, ax, mappable, where="right", size="4%", pad=0.15, label=None, orientation=None):
    divider = make_axes_locatable(ax)
    if where in ("right", "left"):
        cax = divider.append_axes(where, size=size, pad=pad)
        orientation = orientation or "vertical"
    else:
        cax = divider.append_axes(where, size=size, pad=pad)
        orientation = orientation or "horizontal"
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    if label:
        cbar.set_label(label)
    return cbar

# ==== REFACTORED / RESTYLED PLOTTING FUNCTIONS ONLY ====

def plot_2d_geometry(show_tags=None, show_index=False, scale=False, cmap=None,
                     circle_scale: float = 1e3, title=None, mode=None,
                     indicate_atoms=False, grid=False):
    """Distinct style: soft grid, inset colorbar, layered scatter with alpha."""
    # local import dependencies (assumed available in caller context)
    from granad import Rhomboid
    # uses: get_haldane_graphene, localization (already defined by you)

    # decider whether to take abs val and normalize 
    def scale_vals(vals):
        return jnp.abs(vals) / (jnp.abs(vals).max() + 1e-15) if scale else vals

    shape = Rhomboid(40, 40, armchair=False)
    orbs = get_haldane_graphene(1.0, 1j*0.3, 1.0).cut_flake(shape)
    l = localization(orbs)
    display = jnp.abs(orbs.eigenvectors[:, l.argsort()[-4]])**2

    xs = jnp.array([orb.position[0] for orb in orbs])
    ys = jnp.array([orb.position[1] for orb in orbs])
    colors = scale_vals(display)
    cmap = cmap or "plasma"

    with _theme():
        fig, ax = plt.subplots(figsize=(6, 5.8))
        # base points (faint)
        ax.scatter(xs, ys, s=8, color="0.15", zorder=1)
        # magnitude overlay
        s = circle_scale * (display / (display.max() + 1e-15)) * 6.0
        sc = ax.scatter(xs, ys, s=s, c=colors, cmap=cmap, ec="none", alpha=0.9, zorder=2)

        _beautify_axes(ax, xlabel="x (Å)", ylabel="y (Å)")
        ax.set_aspect("equal", adjustable="box")

        if title:
            ax.set_title(title, pad=8)

        cbar = _add_cbar(fig, ax, sc, where="right", label=r"$|\Psi|^2$")
        plt.tight_layout()
        plt.savefig("geometry.pdf", bbox_inches="tight")
        plt.close(fig)

def get_projection(dip):
    """(unchanged API) projection to circular (helicity) basis."""
    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    return jnp.einsum('ij, jkl -> ikl', trafo, dip)

def plot_projected_polarization():
    """Matrix elements in circular basis, shared colorbar across panels."""
    from granad import Rhomboid
    delta, t_nn = 1.0, 1.0
    ts = [0.0, 0.1, 0.2]
    labels = ["(a)", "(b)", "(c)"]

    with _theme():
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
        vmin = None; vmax = None
        ims = []
        for i, t in enumerate(ts):
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(Rhomboid(20, 20, armchair=False))
            dip = flake.velocity_operator_e[:2]
            proj = get_projection(dip)
            A = jnp.abs(proj[0])**2
            im = axes[i].imshow(A, cmap="inferno", origin="lower", interpolation="nearest")
            ims.append(im)
            axes[i].set_xlabel(r"$m$")
            if i == 0:
                axes[i].set_ylabel(r"$n$")
            _beautify_axes(axes[i])
            axes[i].text(0.02, 0.98, labels[i], transform=axes[i].transAxes,
                         ha="left", va="top", fontweight="bold")

        # one shared colorbar
        cbar = fig.colorbar(ims[0], ax=axes, shrink=0.85, location="top", pad=0.08)
        cbar.set_label(r"$|J_{+}|^2$ (a.u.)")
        plt.savefig("projected_polarizations.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_dipole_moments():
    """|p_+| and |p_-| vs ω with peak annotations; clean log scale layout."""
    from granad import Rhomboid
    shape = Rhomboid(40, 40, armchair=False)
    delta, t_nn = 1.0, 1.0
    ts = [0.4]        # keep your selection
    omegas = jnp.linspace(0., 0.5, 300)

    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))

    with _theme():
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
            alpha_cart = ip_response(flake, omegas, relaxation_rate=1e-3)["total"]
            dip = f_dip(alpha_cart)

            ln1, = ax.plot(omegas, dip[0], label=r"$|p_+|$", linewidth=2.0)
            ln2, = ax.plot(omegas, dip[1], label=r"$|p_-|$", linestyle="--", linewidth=2.0)

        ax.set_yscale("log")
        _beautify_axes(ax, xlabel=r"$\omega/t$", ylabel=r"$|p|$ (a.u.)")
        ax.legend(frameon=False, ncol=2, loc="upper right")

        # annotate a couple of peaks (robust to edges)
        pp = find_peaks(dip[0])[0]
        pm = find_peaks(dip[1])[2] if len(find_peaks(dip[1])) > 2 else find_peaks(dip[1])[-1]
        for idx, lab in [(pp, r"$J_{+}$-dom."), (pm, r"$J_{-}$-dom.")]:
            x0 = float(omegas[idx]); y0 = float(dip[0][idx] if lab.startswith(r"$J_{+}$") else dip[1][idx])
            ax.annotate(lab, xy=(x0, y0),
                        xytext=(x0 * 1.08, y0 * 1.5),
                        arrowprops=dict(arrowstyle="->", lw=1.2),
                        fontsize=12)

        plt.tight_layout()
        plt.savefig("p.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_dipole_moments_sweep():
    """Heatmap of |p_+| - |p_-| across λ and ω (pcolormesh for crisp axes)."""
    from granad import Rhomboid
    shape = Rhomboid(40, 40, armchair=False)
    delta, t_nn = 1.0, 1.0
    ts = jnp.linspace(0, 0.4, 40)
    omegas = jnp.linspace(0., 0.5, 300)

    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))

    res = []
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
        alpha_cart = ip_response(flake, omegas, relaxation_rate=1e-3)["total"]
        d = f_dip(alpha_cart)
        res.append(d[0] - d[1])
    Z = jnp.array(res).T  # shape (W, T)

    with _theme():
        fig, ax = plt.subplots(figsize=(6.6, 5.5))
        norm = SymLogNorm(linthresh=1, linscale=1.0, base=10)
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="coolwarm", norm=norm,
                       extent=[float(ts.min()), float(ts.max()),
                               float(omegas.min()), float(omegas.max())])
        ax.axvline(float(get_threshold(delta)), color="k", linestyle="--", linewidth=1.5, alpha=0.8)
        _beautify_axes(ax, xlabel=r"$\lambda/t$", ylabel=r"$\omega/t$")
        _add_cbar(fig, ax, im, where="right", label=r"$|p_+|-|p_-|$ (a.u.)")
        plt.tight_layout()
        plt.savefig("p_sweep.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_noise_dipole_moments_sweep():
    """Same as above, but with diagonal noise; distinct visual style preserved."""
    # uses your inner helpers exactly, but applies the new theme and colorbar style
    import jax
    import jax.numpy as jnp
    import jax.random as random
    from granad import Rhomboid

    def _get_ip_green_function(A, B, energies, occupations, omegas, relaxation_rate):
        def inner(omega):
            return jnp.trace((delta_occ / (omega + delta_e + 1j*relaxation_rate)) @ operator_product)
        operator_product = A.T * B
        delta_occ = (occupations[:, None] - occupations)
        delta_e = energies[:, None] - energies
        return jax.lax.map(jax.jit(inner), omegas)

    def _get_correlator(flake, scale, omegas):
        occupations = jnp.ones(len(flake)) * (jnp.arange(len(flake)) < len(flake) // 2)
        N = flake.positions.shape[0]
        dipole_operator = jnp.zeros((3, N, N)).astype(complex)
        for i in range(3):
            dipole_operator = dipole_operator.at[i, :, :].set(jnp.diag(flake.positions[:, i] / 2))
        dipole_operator = dipole_operator + jnp.transpose(dipole_operator, (0, 2, 1)).conj()

        key = random.PRNGKey(42)
        noise = random.normal(key, shape=(len(flake), len(flake)))
        noise = jnp.diag(jnp.diag(noise))
        hamiltonian = flake.hamiltonian + scale * noise
        energies, vecs = jnp.linalg.eigh(hamiltonian)
        vecs = vecs.conj().T
        dipole_operator = jnp.einsum("ij,mjk,lk->mil", vecs, dipole_operator, vecs.conj())[:2]
        return jnp.array([[_get_ip_green_function(o1, o2, energies, occupations, omegas, 1e-3)
                            for o1 in dipole_operator] for o2 in dipole_operator])

    def single_plot(noise):
        delta, t_nn = 1.0, 1.0
        ts = jnp.linspace(0, 0.4, 40)
        shape = Rhomboid(40, 40, armchair=False)
        omegas = jnp.linspace(0., 0.5, 300)

        trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
        f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))
        res = []
        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
            alpha_cart = _get_correlator(flake, noise, omegas)
            dip = f_dip(alpha_cart)
            res.append(dip[0] - dip[1])
        Z = jnp.array(res).T

        with _theme():
            fig, ax = plt.subplots(figsize=(6.6, 5.5))
            norm = SymLogNorm(linthresh=1, linscale=1.0, base=10)
            im = ax.imshow(Z, origin="lower", aspect="auto", cmap="coolwarm", norm=norm,
                           extent=[float(ts.min()), float(ts.max()),
                                   float(omegas.min()), float(omegas.max())])
            ax.axvline(float(get_threshold(delta)), color="k", linestyle="--", linewidth=1.5, alpha=0.8)
            _beautify_axes(ax, xlabel=r"$\lambda/t$", ylabel=r"$\omega/t$")
            _add_cbar(fig, ax, im, where="right", label=r"$|p_+|-|p_-|$ (a.u.)")
            plt.savefig(f"p_sweep_noise_{noise}.pdf", bbox_inches="tight")
            plt.close(fig)

    for noise in [0, 1]:
        single_plot(noise)

def plot_flake_ip_cd():
    """Circular dichroism on a triangular flake; distinct palette/legend."""
    from granad import Triangle
    shape = Triangle(20, armchair=False)
    delta, t_nn = 1.0, -2.66
    ts = [0, 1e-5, 0.4]
    omegas = jnp.linspace(0, 8, 100)

    f_cd = lambda pph: (jnp.abs(pph[0, 0].imag) - jnp.abs(pph[1, 1].imag)) / \
                       (jnp.abs(pph[0, 0].imag) + jnp.abs(pph[1, 1].imag) + 1e-15)

    with _theme():
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
            absp = get_ip_abs(flake, omegas, 0, relaxation_rate=1e-2)
            absm = get_ip_abs(flake, omegas, 1, relaxation_rate=1e-2)
            cd = (absp.imag - absm.imag) / (absp.imag + absm.imag + 1e-15)

            ls = "-" if t > get_threshold(delta) else ":"
            ax.plot(omegas, cd, label=fr"$\lambda/t={t:.2f}$", linestyle=ls, linewidth=2)

        _beautify_axes(ax, xlabel=r"$\omega/t$", ylabel=r"$s$")
        ax.legend(frameon=False, ncol=3, loc="upper right")
        plt.tight_layout()
        plt.savefig("ip_cd.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_selectivity_sweep():
    """Heatmap of selectivity s across λ and ω with a dashed threshold line."""
    from granad import Rhomboid
    shape = Rhomboid(40, 40, armchair=False)
    delta, t_nn = 1.0, 1.0
    ts = jnp.linspace(0, 0.4, 40)
    omegas = jnp.linspace(0., 0.5, 300)

    f_cd = lambda pph: (jnp.abs(pph[0, 0].imag) - jnp.abs(pph[1, 1].imag)) / \
                       (jnp.abs(pph[0, 0].imag) + jnp.abs(pph[1, 1].imag) + 1e-15)

    res = []
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
        alpha_cart = ip_response(flake, omegas, relaxation_rate=1e-3)["total"]
        alpha_circ = to_helicity(alpha_cart)
        res.append(f_cd(alpha_circ))
    Z = jnp.array(res).T

    with _theme():
        fig, ax = plt.subplots(figsize=(6.6, 5.5))
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="coolwarm",
                       extent=[float(ts.min()), float(ts.max()),
                               float(omegas.min()), float(omegas.max())])
        ax.axvline(float(get_threshold(delta)), color="k", linestyle="--", linewidth=1.5, alpha=0.8)
        _beautify_axes(ax, xlabel=r"$\lambda/t$", ylabel=r"$\omega/t$")
        _add_cbar(fig, ax, im, where="right", label=r"$s$")
        plt.tight_layout()
        plt.savefig("selectivity_sweep.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_energy_localization():
    """Energy spectrum colored by edge localization; shared horizontal colorbar."""
    from granad import Rhomboid
    shape = Rhomboid(20, 20, armchair=False)
    ts = [0.05, 0.4]

    with _theme():
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
        sc_last = None
        for i, t in enumerate(ts):
            flake = get_haldane_graphene(1., 1j * t, 1.).cut_flake(shape)
            e_max, e_min = float(flake.energies.max()), float(flake.energies.min())
            pad = 0.01 * (e_max - e_min + 1e-12)
            e_max += pad; e_min -= pad

            loc = localization(flake)
            idx = jnp.where((flake.energies >= e_min) & (flake.energies <= e_max))[0]
            sc = axs[i].scatter(idx, flake.energies[idx], c=loc, vmin=0, vmax=1, cmap="viridis", s=14)
            sc_last = sc

            axs[i].set_xlabel("state index")
            if i == 0:
                axs[i].set_ylabel(r"$E/t$")
            axs[i].axhline((e_max + e_min)/2, color="0.5", linestyle="--", linewidth=1, alpha=0.6)
            _beautify_axes(axs[i])
            axs[i].text(0.02, 0.98, f"({chr(97+i)})", transform=axs[i].transAxes,
                        ha="left", va="top", fontweight="bold")

        # shared horizontal colorbar on top
        cbar = fig.colorbar(sc_last, ax=axs, location="top", fraction=0.06, pad=0.08, shrink=0.9, orientation="horizontal")
        cbar.set_label(r"$\mathcal{L}$")
        plt.savefig("energy_localization.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_size_sweep():
    """Heatmap of |p_+| - |p_-| vs system size N and ω."""
    from granad import Rhomboid
    delta, t_nn = 1.0, 1.0
    omegas = jnp.linspace(0., 0.5, 300)

    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))

    sizes = jnp.arange(20, 100, 2)
    res = []
    Ns = []
    for s in sizes:
        shape = Rhomboid(s, s, armchair=False)
        flake = get_haldane_graphene(t_nn, 1j*0.4, delta).cut_flake(shape)
        alpha_cart = ip_response(flake, omegas, relaxation_rate=1e-3)["total"]
        d = f_dip(alpha_cart)
        res.append(d[0] - d[1])
        Ns.append(len(flake))
    Z = jnp.array(res).T
    Ns = jnp.array(Ns)

    with _theme():
        fig, ax = plt.subplots(figsize=(6.6, 5.5))
        norm = SymLogNorm(linthresh=1, linscale=1.0, base=10)
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="coolwarm", norm=norm,
                       extent=[float(Ns.min()), float(Ns.max()),
                               float(omegas.min()), float(omegas.max())])
        _beautify_axes(ax, xlabel=r"$N$", ylabel=r"$\omega/t$")
        _add_cbar(fig, ax, im, where="right", label=r"$|p_+|-|p_-|$ (a.u.)")
        plt.tight_layout()
        plt.savefig("size_sweep.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_rpa_sweep():
    """RPA |p_+|-|p_-| vs Coulomb strength c and ω."""
    from granad import Rhomboid
    shape = Rhomboid(40, 40, armchair=False)
    delta, t_nn = 1.0, 1.0
    omegas = jnp.linspace(0., 1, 300)
    cs = jnp.linspace(0, 1, 40)

    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))

    res = []
    for c in cs:
        flake = get_haldane_graphene(t_nn, 1j*0.4, delta).cut_flake(shape)
        alpha_cart = rpa_polarizability(flake, omegas, [c], relaxation_rate=1e-3)[0][:2, :2]
        d = f_dip(alpha_cart)
        res.append(d[0] - d[1])
    Z = jnp.array(res).T

    with _theme():
        fig, ax = plt.subplots(figsize=(6.6, 5.5))
        norm = SymLogNorm(linthresh=1, linscale=1.0, base=10)
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="coolwarm", norm=norm,
                       extent=[float(cs.min()), float(cs.max()),
                               float(omegas.min()), float(omegas.max())])
        _beautify_axes(ax, xlabel=r"$c$", ylabel=r"$\omega/t$")
        _add_cbar(fig, ax, im, where="right", label=r"$|p_+|-|p_-|$ (a.u.)")
        plt.tight_layout()
        plt.savefig("rpa_sweep.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_dipole_moments_p_j():
    """Compare dipoles from xpp vs. xjj (TRK-consistent view)."""
    from granad import Rhomboid
    shape = Rhomboid(20, 20, armchair=False)
    delta, t_nn = 1.0, 1.0
    ts = [0, 0.15, 0.4]
    omegas = jnp.linspace(0., 0.5, 300)

    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))
    f_dip_j = lambda jj: f_dip(jj - jj[..., 0][:, :, None])

    with _theme():
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
            alpha_cart = ip_response(flake, omegas, relaxation_rate=1e-3)["total"]
            dip = f_dip(alpha_cart)

            chi_cart = ip_response(flake, omegas, relaxation_rate=1e-3,
                                   os1=flake.velocity_operator_e[:2],
                                   os2=flake.velocity_operator_e[:2])["total"]
            dip2 = f_dip_j(chi_cart) / (omegas**2 + 1e-15)

            ax.plot(omegas, dip[0] - dip[1], label=fr"$\lambda/t={t:.2f}$ (x)", linewidth=2.0)
            ax.plot(omegas, dip2[0] - dip2[1], linestyle="--",
                    label=fr"$\lambda/t={t:.2f}$ (j)", linewidth=2.0)

        _beautify_axes(ax, xlabel=r"$\omega/t$", ylabel=r"$|p_+|-|p_-|$ (a.u.)")
        ax.legend(frameon=False, ncol=2)
        plt.tight_layout()
        plt.savefig("p_trk.pdf", bbox_inches="tight")
        plt.close(fig)

def plot_dipole_moments_broken_symmetry():
    """|p_+| and |p_-| on triangular flake with broken symmetry."""
    from granad import Triangle
    shape = Triangle(20, armchair=False)
    delta, t_nn = 1.0, 1.0
    ts = [0.5]   # keep your selection
    omegas = jnp.linspace(0., 0.8, 300)

    trafo = 1 / jnp.sqrt(2) * jnp.array([[1, -1j], [1, 1j]])
    f_dip = lambda xx: jnp.abs(jnp.einsum('ij, jk -> ik', trafo, xx.sum(axis=1)))

    with _theme():
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for t in ts:
            flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
            alpha_cart = ip_response(flake, omegas, relaxation_rate=1e-3)["total"]
            dip = f_dip(alpha_cart)
            ax.plot(omegas, dip[0], label=r"$|p_+|$", linewidth=2.0)
            ax.plot(omegas, dip[1], label=r"$|p_-|$", linestyle="--", linewidth=2.0)

        ax.set_yscale("log")
        _beautify_axes(ax, xlabel=r"$\omega/t$", ylabel=r"$|p|$ (a.u.)")
        ax.legend(frameon=False, ncol=2, loc="upper right")
        plt.tight_layout()
        plt.savefig("p_broken.pdf", bbox_inches="tight")
        plt.close(fig)

        
if __name__ == '__main__':
    plot_2d_geometry() # DONE
    plot_projected_polarization() # DONE
    plot_dipole_moments() # DONE
    plot_dipole_moments_sweep() # DONE
    plot_energy_localization() # DONE
    plot_selectivity_sweep() # DONE
    plot_size_sweep()  # DONE
    
    plot_noise_dipole_moments_sweep() # DONE
    
    # APPENDIX
    plot_dipole_moments_p_j() # DONE
    plot_rpa_sweep() # DONE
    plot_dipole_moments_broken_symmetry() # DONE
