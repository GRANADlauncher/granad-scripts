# TODO: present circular dichroism in flakes, reference DOI: 10.1103/PhysRevB.99.161404
# TODO: tell nice story going from condensed bulk to optical bulk, interacting via dipole stuff
# TODO: no analytical lattice, use treams

# TODO: there is sth wrong if eps1 != eps2, but im not gonna fix that
# TODO: rot average / use geometry with xx = yy, think about s / p implication if alpha_zz = 0

# relevant scales:
# length in nm, energies in eV, hbar = 1
from typing import Any

import jax
import jax.numpy as jnp
from dataclasses import dataclass

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from granad import *


### UTILITIES ###
def load_data(results_file, keys):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")
    return omegas, data, data.keys() if keys is None else keys    

def get_threshold(delta):
    """threshold for topological nontriviality for t_2"""
    return delta / (3 * jnp.sqrt(3) )

LIGHT = 299.8
def wavelength(omega):
    return LIGHT / (omega / 2*jnp.pi)

def omega(wavelength):
    return 2*jnp.pi * LIGHT / wavelength
    
def to_helicity(mat):
    """converts mat to helicity basis"""
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    trafo_inv = jnp.linalg.inv(trafo)
    return jnp.einsum('ij,jmk,ml->ilk', trafo, mat, trafo_inv)

### MATERIAL ###
def sample_brillouin_zone(num_kpoints=100):
    """
    Samples the hexagonal Brillouin zone along the high-symmetry path Γ-K-M-K'-Γ.
    
    Parameters:
    num_kpoints : int
        Number of k-points to sample along the path.
    
    Returns:
    kx, ky : ndarray
        Arrays of kx and ky values along the high-symmetry path.
    """
    # High-symmetry points in the hexagonal Brillouin zone
    Gamma = np.array([0, 0])
    K = np.array([4*np.pi/3, 0])
    M = np.array([2*np.pi/3, 2*np.pi/3*np.sqrt(3)])
    Kp = np.array([-4*np.pi/3, 0])
    
    # Define path segments
    segments = [Gamma, K, M, Kp, Gamma]
    
    # Generate k-point path
    kx, ky = [], []
    for i in range(len(segments)-1):
        k_start, k_end = segments[i], segments[i+1]
        k_linspace = np.linspace(k_start, k_end, num_kpoints // (len(segments)-1))
        kx.extend(k_linspace[:, 0])
        ky.extend(k_linspace[:, 1])
    
    return np.array(kx), np.array(ky)

def haldane_hamiltonian(k, t1=1.0, t2=0.2, phi=jnp.pi/2, M=0.0):
    """
    Computes the Haldane model Hamiltonian in momentum space.
    
    Parameters:
    k : array-like
        momentum in the Brillouin zone.
    t1 : float
        Nearest-neighbor hopping amplitude.
    t2 : float
        Next-nearest-neighbor hopping amplitude.
    phi : float
        Phase associated with the complex hopping (breaks time-reversal symmetry).
    M : float
        Sublattice potential term (breaks inversion symmetry).
    
    Returns:
    H : ndarray
        Hamiltonian matrix of shape (2, 2, len(kx))
    """    
    sigma_0 = jnp.eye(2)
    sigma_x = jnp.array( [ [0, 1], [1, 0] ] )
    sigma_y = 1j * jnp.array( [ [0, 1], [-1, 0] ] )
    sigma_z = jnp.array( [ [1, 0], [0, -1] ] )

    # nearest-neighbor vectors
    a_vecs = jnp.array( [
        [1, 0],
        [-1/2, jnp.sqrt(3)/2],
        [-1/2, -jnp.sqrt(3)/2]        
        ]
    )
    
    # Next-nearest-neighbor vectors
    b_vecs = jnp.array([
        [0, jnp.sqrt(3)],
        [-3/2, -jnp.sqrt(3)/2],
        [3/2, -jnp.sqrt(3)/2]
        ]
    )

    # wave numbers
    ka = a_vecs @ k 
    kb = b_vecs @ k

    # Hamiltonian components
    A0 = 2 * t2 * jnp.cos(phi) * jnp.cos(kb).sum()
    A1 = t1 * jnp.cos(ka).sum()
    A2 = t1 * jnp.sin(ka).sum()
    A3 = -2 * t2 * jnp.sin(phi) * jnp.sin(kb).sum() + M

    # hamiltonian matrix
    H = A0 * sigma_0 + A1 * sigma_x + A2 * sigma_y + A3 * sigma_z
    
    return H

def get_haldane_graphene(t1, t2, delta):
    """Constructs a graphene model with onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2

    threshold is at $t_2 > \\frac{\\delta}{3 \\sqrt{3}}$
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
    """computes Wx3x3 IP polarizability according to usual lehman representation"""
    corr = {}
    os1 = os1 if os1 is not None else flake.dipole_operator_e[:2]
    os2 = os2 if os2 is not None else flake.dipole_operator_e[:2]
    
    corr["total"] =  get_correlator(flake, omegas, os1, os2, relaxation_rate = relaxation_rate)

    if topology == True:
        trivial = jnp.abs(flake.energies) > 0.1
        mask = jnp.logical_and(trivial[:, None], trivial)        
        corr["topological"] = get_correlator(flake, omegas, os1, os2, relaxation_rate = relaxation_rate, mask = mask)

    corr["omegas"] = omegas
    
    if results_file is not None:
        jnp.savez(results_file, **corr)
        
    return corr

### RPA ###
def rpa_susceptibility(flake, c, relaxation_rate):
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
        sus = rpa_susceptibility(flake, c, relaxation_rate = 0.05)
        
        p = flake.positions.T
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)

        # TODO: check if this is right, maybe missing omegas?
        pol.append(ref)

    if results_file is not None:
        jnp.savez(results_file, pol = pol, omegas = omegas, cs = cs)
        
    return res

### OPTICAL SIMS ###
@dataclass
class Params:
    omegas : Any
    alpha : Any
    theta : Any
    
@dataclass
class Lattice:
    eps1: float
    eps2: float
    a: float
    is_square: bool = True
    
    @property
    def area(self):
        return jax.lax.cond(
            self.is_square,
            lambda: self.a**2,
            lambda: (jnp.sqrt(3) / 2) * self.a**2
        )

    @property
    def g(self):
        return jax.lax.cond(
            self.is_square,
            lambda: 4.52,
            lambda: 5.52
        )

    def subwavelength_omegas():
        lambda_min = self.a * 1.1 # nm, 10 % larger than lattice, just to make sure
        omega_min = 1240 / lambda_min # eV
        return jnp.linspace(omega_min, 10*omega_max, 100)
    
    def _snell_angle(self, theta_i):
        """Compute transmitted angle using Snell's Law."""
        n1 = jnp.sqrt(self.eps1)
        n2 = jnp.sqrt(self.eps2)
        sin_theta_t = n1 * jnp.sin(theta_i) / n2
        # Ensure the angle is valid (total internal reflection case)
        return jnp.arcsin(jax.lax.clamp(-1.0, sin_theta_t, 1.0))
    
    def r0(self, params):
        """Compute Fresnel reflection coefficient (r) for s and p polarization."""
        theta_i = params.theta
        theta_t = self._snell_angle(theta_i)

        n1 = jnp.sqrt(self.eps1)
        n2 = jnp.sqrt(self.eps2)

        # Compute reflection coefficients
        rs = (n1 * jnp.cos(theta_i) - n2 * jnp.cos(theta_t)) / (n1 * jnp.cos(theta_i) + n2 * jnp.cos(theta_t))
        rp = (n2 * jnp.cos(theta_i) - n1 * jnp.cos(theta_t)) / (n2 * jnp.cos(theta_i) + n1 * jnp.cos(theta_t))

        return {"s": rs, "p": rp}

    def t0(self, params):
        """Compute Fresnel transmission coefficient (t) for s and p polarization."""
        theta_i = params.theta
        theta_t = self._snell_angle(theta_i)

        n1 = jnp.sqrt(self.eps1)
        n2 = jnp.sqrt(self.eps2)

        # Compute transmission coefficients
        ts = (2 * n1 * jnp.cos(theta_i)) / (n1 * jnp.cos(theta_i) + n2 * jnp.cos(theta_t))
        tp = (2 * n1 * jnp.cos(theta_i)) / (n2 * jnp.cos(theta_i) + n1 * jnp.cos(theta_t))

        return {"s": ts, "p": tp}

    # fine
    def reflection_layer(self, params, polarization):
        """Generalized reflection layer calculation for s and p polarization."""
        omegas, alpha, theta = params.omegas, params.alpha, params.theta
        S = 2 * jnp.pi * omegas * (1/jnp.cos(theta) if polarization == "s" else jnp.cos(theta)) / (LIGHT * self.area)
        G_im = S - 2 * (omegas / LIGHT) ** 3 / 3
        G_re = self.g / self.a ** 3
        G = G_re + 1j * G_im
        factor = 1j if polarization == "s" else -1j
        return factor * S / (1/alpha - G)

    # fine
    def amplitude(self, params, polarization):
        """Generalized amplitude calculation for s and p polarization."""
        r = self.reflection_layer(params, polarization)
        r0 = self.r0(params)[polarization]
        factor = 1 if polarization == "s" else -1
        return r * (1 + factor * r0) / (1 - r0 * r)

    # fine
    def total_reflection(self, params, polarization):
        """Generalized total reflection calculation for s and p polarization."""
        eta = self.amplitude(params, polarization)
        r0 = self.r0(params)[polarization]
        factor = 1 if polarization == "s" else -1
        return r0 + (1 + factor * r0) * eta

    # fine
    def total_transmission(self, params, polarization):
        """Generalized total transmission calculation for s and p polarization."""
        eta = self.amplitude(params, polarization)
        t0 = self.t0(params)[polarization]
        factor = 1 if polarization == "s" else -1
        return t0 + factor * t0 * eta

    def coefficients(self, params):
        t_s = l.total_transmission(p, "s")
        r_s = l.total_reflection(p, "s")    
        t_p = l.total_transmission(p, "p")
        r_p = l.total_reflection(p, "p")
        return t_s, r_s, t_p, r_p


### PLOTTING ###    
def show_2d(orbs, show_tags=None, show_index=False, display = None, scale = False, cmap = None, circle_scale : float = 1e3, title = None, mode = None, indicate_atoms = False, grid = False):

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

    # Apply settings only for this block
    with mpl.rc_context(rc=custom_params):

        # Create plot
        fig, ax = plt.subplots()
        cmap = plt.cm.bwr if cmap is None else cmap
        colors = scale_vals(display)            
        scatter = ax.scatter([orb.position[0] * scale for orb in orbs], [orb.position[1]  * scale for orb in orbs], c=colors, edgecolor='none', cmap=cmap, s = circle_scale*jnp.abs(display) )
        ax.scatter([orb.position[0] * scale  for orb in orbs], [orb.position[1]  * scale  for orb in orbs], color='black', s=5, marker='o')            
        cbar = fig.colorbar(scatter, ax=ax)

        # Finalize plot settings
        if title is not None:
            plt.title(title)
            
        plt.xlabel('X (Å)')
        plt.ylabel('Y (Å)')    
        ax.grid(True)
        ax.axis('equal')
        plt.savefig('geometry.pdf')
    
def plot_absorption(omegas, t, r, name):
    a = 1 - (jnp.abs(t)**2 + jnp.abs(r)**2)
    plt.plot(omegas, a)
    plt.show()
    plt.savefig(name)    

def check_prl_figures():    
    # params paper:
    # eps1 = 1, eps2 = 10
    # disk 60nm, Ef = 0.4 eV
    # lambda >> lattice const 72 - 126 nm, 

    # selling point: there is analytical upper limit on absorption, eqn. 3, how to reach? can with graphene and eqn. 5

    def alpha_ref(omegas):
        omega_p = 0.195 # eV

        kappa = 0.196 - 0.194 # eV, eye-balled

        sigma_ext = 6 * jnp.pi*(60/2)**2 # dimensionless peak in fig 1 a times approximate disk area (nm)

        res_wv = 1240 / omega_p # convert to nm
        sigma_prefac = 3*res_wv**2 / (2 * jnp.pi) # nm
        kappa_r = sigma_ext / sigma_prefac * kappa # eV

        c = 299.8 # nm / fs, since hbar = 1

        prefac = 3 * c**3 * kappa_r / (2 * omega_p**2)
        freq = 1/(omega_p**2 - omegas**2 - 1j * kappa * omegas**3/omega_p**2)
        return prefac * freq    

    # fig 1 a
    omegas = jnp.linspace(0.18, 0.21, 100)
    ext = alpha_ref(omegas).imag * 4 * jnp.pi * omegas / 300 / (jnp.pi * 30**2)
    plt.plot(omegas, ext)
    plt.savefig("extinction_prl.pdf")

    # fig 2 normal incidence
    eps1 = 1
    eps2 = 1
    l = Lattice(eps1 = eps1, eps2 = eps2, a = 99.0, is_square = False)

    # freqs dont match entirely but thats okay given i had to eyeball everything
    omegas = jnp.linspace(0.14, 0.2, 100)
    alpha = alpha_ref(omegas)

    p = Params(omegas = omegas, alpha = alpha, theta = 0) #jnp.pi / 180 * 35)
    
    t_s, r_s, t_p, r_p = l.coefficients(p)

    plot_absorption(omegas, t_s, r_s, "normal_s_prl.pdf")
    plot_absorption(omegas, t_p, r_p, "normal_p_prl.pdf")

    # fig 2 oblique, there is sth wrong if eps1 != eps2, but im not gonna fix that
    eps1 = 1
    eps2 = 1
    l = Lattice(eps1 = eps1, eps2 = eps2, a = 78.0, is_square = False)

    omegas = jnp.linspace(0.0, 0.2, 100)
    alpha = alpha_ref(omegas)

    p = Params(omegas = omegas, alpha = alpha, theta = jnp.pi / 180 * 35)

    t_s, r_s, t_p, r_p = l.coefficients(p)

    plot_absorption(omegas, t_s, r_s, "oblique_s_prl.pdf")
    plot_absorption(omegas, t_p, r_p, "oblique_p_prl.pdf")

# TODO: identify transitions by states    
def plot_dipole_operator_components():
    # look if single flake is interesting    
    shape = Triangle(42, armchair = False)
    
    # why do peak happen? => dipole transitions between edge states become allowed
    material = get_haldane_graphene(t_nn, 0, delta)
    flake_triv = material.cut_flake(shape)

    material = get_haldane_graphene(t_nn, 1j*0.005, delta)
    flake_topo = material.cut_flake(shape)

    flake_triv.dipole_operator_e

    lower = 0#flake_triv.homo - 2
    upper = -1#flake_triv.homo + 2

    dip_triv = jnp.abs(flake_triv.dipole_operator_e[1, lower:upper, lower:upper])
    dip_triv /= dip_triv.max()
    dip_topo = jnp.abs(flake_topo.dipole_operator_e[1, lower:upper, lower:upper])
    dip_topo /= dip_topo.max()
    vmax = jnp.concatenate([dip_triv, dip_topo]).max()
    vmin = jnp.concatenate([dip_triv, dip_topo]).min()
    
    plt.matshow(dip_triv, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.savefig("triv.pdf")
    plt.close()
    
    plt.matshow(dip_topo, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.savefig("topo.pdf")
    plt.close()

# extinction shows large values towards lower freqs associated with edge states
def plot_single_cross_sections():    
    # vary shape?
    shape = Triangle(20, armchair = False)
    
    # vary?
    delta = 0.0
    t_nn = -2.66
    
    ts = jnp.linspace(0, 0.05, 10)

    # omegas
    omegas = jnp.linspace(0, 4, 100)
    
    for t in ts:
        material = get_haldane_graphene(t_nn, 1j*t, delta)
        flake = material.cut_flake(shape)

        flake.show_energies(name = f'{t:.2f}.pdf')
        
        alpha = jnp.trace(ip_response(flake, omegas, relaxation_rate = 0.001)["total"], axis1=0, axis2=1)
        
        k = omegas / LIGHT

        extinction = -k * alpha.imag
        scattering = k**2 / (6 * jnp.pi) * jnp.abs(alpha)**2
        absorption = extinction - scattering

        ls = '-' if t > get_threshold(delta) else '--'
        plt.plot(omegas, extinction, label = f'{t:.2f}', ls = ls)

    plt.legend()
    plt.savefig("cross_sections.pdf")
    plt.close()

def plot_flake_cd():
    # vary shape?
    shape = Triangle(20, armchair = True)
    
    # vary?    
    delta = 1.0
    t_nn = -2.66
    
    ts = jnp.linspace(0, 0.1, 10)
    
    # omegas
    omegas = jnp.linspace(0.2, 6, 100)
    
    # # uses 9 trillion ways to compute cd
    # flake = get_haldane_graphene(t_nn, 0.5j, delta).cut_flake(shape)
    # jj = ip_response(flake, omegas, os1 = flake.velocity_operator_e[:2], os2 = flake.velocity_operator_e[:2], relaxation_rate = 0.01)["total"]
    # xj = jj[1, 0, :].imag / (2 * jj[0,0,:].real)
    # print(xj)
    # xj2 = jj[1, 0, :].imag / (2 * jnp.trace(jj).real) * omegas / LIGHT
    # print(xj2)
    
    # # "canonical" way
    # pp = ip_response(flake, omegas, relaxation_rate = 0.01)["total"]
    # pph = to_helicity(pp)
    # xp = (pph[0, 0].imag - pph[1, 1].imag) / (2*(pph[0, 0].imag + pph[1, 1].imag))
    # print(xp)

    f_cd = lambda pph: (pph[0, 0].imag - pph[1, 1].imag) / (2*(pph[0, 0].imag + pph[1, 1].imag))    

    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)

        alpha = to_helicity(ip_response(flake, omegas, relaxation_rate = 0.01)["total"])
        cd = f_cd(alpha)
        
        ls = '-' if t > get_threshold(delta) else '--'
        plt.plot(omegas, cd, label = f'{t:.2f}', ls = ls)

    plt.legend()
    plt.savefig("cd.pdf")
    plt.close()
    
# wavelength resolution resonance at a glance
def plot_flake_alpha():
    # vary shape?
    shape = Triangle(20, armchair = False)
    
    # vary?
    delta = 0.01
    t_nn = -2.66
    
    ts = [0, 0.05]# jnp.linspace(0, 0.05, 10)
    
    # omegas
    omegas = jnp.linspace(0.01, 1., 100)
    
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
                
        # alpha = jnp.trace(ip_response(flake, omegas, relaxation_rate = 0.01)["total"], axis1=0, axis2=1)
        alpha = jnp.diagonal(ip_response(flake, omegas, relaxation_rate = 0.01)["total"])[:, 1]
        
        k = omegas / LIGHT
        alpha = alpha.imag
        ls = '-' if t > get_threshold(delta) else '--'
        plt.plot(1240 / omegas, alpha, label = f'{t:.2f}', ls = ls)

    plt.legend()
    plt.savefig("alphas.pdf")
    plt.close()

# TODO: unfug yet   
def plot_lattice():
        
    # vary shape?
    shape = Triangle(20, armchair = False)
    
    # vary?
    delta = 1
    t_nn = -2.66
    
    ts = [0.0, 0.5]# jnp.linspace(0, 0.05, 10)

    # fig 2 normal incidence
    eps1 = 1
    eps2 = 1
    l = Lattice(eps1 = eps1, eps2 = eps2, a = 95.0, is_square = False)

    # omegas
    omegas = jnp.linspace(0, 20, 400)
    
    for t in ts:
        flake = get_haldane_graphene(t_nn, 1j*t, delta).cut_flake(shape)
                
        alpha = jnp.trace(ip_response(flake, omegas, relaxation_rate = 0.01)["total"], axis1=0, axis2=1)                

        p = Params(omegas = omegas, alpha = alpha, theta = 0)
        t_s, r_s, t_p, r_p = l.coefficients(p)

        ls = '-' if t > get_threshold(delta) else '--'

        plt.plot(omegas, 1 - jnp.abs(r_s)**2 - jnp.abs(t_s)**2, label = f's_{t:.2f}', ls = ls)
        plt.plot(omegas, 1 - jnp.abs(r_p)**2 - jnp.abs(t_p)**2, label = f'p_{t:.2f}', ls = ls)
        
        # plt.plot(omegas, jnp.angle(t_s), label = f's_{t:.2f}', ls = ls)
        # plt.plot(omegas, jnp.angle(t_p), label = f'p_{t:.2f}', ls = ls)
        
        # plt.plot(t_s.real, t_s.imag, label = f's_{t:.2f}', ls = ls)
        # plt.plot(t_p.real, t_p.imag, label = f'p_{t:.2f}', ls = ls)

    plt.legend()
    plt.savefig("coefficients.pdf")
    plt.close()


def plot_dispersion():
    # Create kx, ky meshgrid
    kx_vals = jnp.linspace(-np.pi, np.pi, 100)
    ky_vals = jnp.linspace(-np.pi, np.pi, 100)
    kx_grid, ky_grid = jnp.meshgrid(kx_vals, ky_vals)    
    ks = jnp.stack([kx_grid.ravel(), ky_grid.ravel()])    
    
    # JIT-compiled function mapping
    h_map = jax.jit(jax.vmap(haldane_hamiltonian, in_axes=1))
    h = h_map(ks)  # (Nk, 2, 2)
        
    # Compute eigenvalues
    vals, vecs = jnp.linalg.eigh(h)  # (Nk, 2)    

    # Reshape back to 2D grid shape
    vals = vals.reshape(kx_grid.shape + (-1,))  # (100, 100, 2)

    # # Plot the eigenvalues as a colormap
    # plt.figure(figsize=(6, 5))
    # plt.contourf(kx_vals, ky_vals, vals[..., 0], levels=50, cmap='viridis')  # First band
    # plt.colorbar(label="Eigenvalue 1")
    # plt.title("Band Structure")
    # plt.xlabel(r"$k_x$")
    # plt.ylabel(r"$k_y$")
    # plt.show()

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx_grid, ky_grid, vals[..., 0], cmap='viridis', edgecolor='k')
    ax.plot_surface(kx_grid, ky_grid, vals[..., 1], cmap='viridis', edgecolor='k')
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel("Eigenvalue 1")
    ax.set_title("3D Band Structure")
    plt.show()


if __name__ == '__main__':
    # Create kx, ky meshgrid
    kx_vals = jnp.linspace(-np.pi, np.pi, 100)
    ky_vals = jnp.linspace(-np.pi, np.pi, 100)
    kx_grid, ky_grid = jnp.meshgrid(kx_vals, ky_vals)    

    # Flatten the grids and stack to create input shape (2, Nk)
    ks = jnp.stack([kx_grid.ravel(), ky_grid.ravel()])    

    # CD transition matrix elements
    f_state = lambda k : jnp.linalg.eigh(haldane_hamiltonian(k))[1]
    f_state_prime = jax.jacfwd(f_state)
    state_prime = f_state_prime(ks) # 2 x 2 x 2 x Nk; last two dims are derivs
    
    # derivative of valence band
    v_prime = state_prime[:, 0, :, :]

    # derivative of conduction band
    c_prime = state_prime[:, 1, :, :]    

    # conduction band
    c = jax.vmap(f_state, in_axes = 1)(ks)[:, 0, :]

    # bulk polarization
    pol = jnp.einsum('kc, cdk->dk', c, v_prime)

    # circular electric field
    circ = 1 / jnp.sqrt(2) * jnp.array([ [1, 1], [1j, -1j] ] )

    # projection => transition matrix element
    element = jnp.abs(circ.conj().T @ pol)**2    

    # Reshape back to 2D grid shape
    element = element.reshape(kx_grid.shape + (-1,))  # (100, 100, 2)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx_grid, ky_grid, element[..., 0], cmap='viridis', edgecolor='k')
    # ax.plot_surface(kx_grid, ky_grid, element[..., 1], cmap='viridis', edgecolor='k')
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel("Eigenvalue 1")
    ax.set_title("3D Band Structure")
    plt.savefig("cd+.pdf")
    plt.close()
    
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx_grid, ky_grid, element[..., 1], cmap='viridis', edgecolor='k')
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_zlabel("Eigenvalue 1")
    ax.set_title("3D Band Structure")
    plt.savefig("cd-.pdf")
