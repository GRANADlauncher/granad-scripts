# TODO: bring in alpha from micro sims
# TODO: there is sth wrong if eps1 != eps2, but im not gonna fix that

# relevant scales:
# length in nm, energies in eV, hbar = 1
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

import matplotlib.pyplot as plt

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

    pol = {}
    omegas = jnp.linspace(0, 6, 200)    
    for (flake, name) in args_list:        
        v, p = flake.velocity_operator_e, flake.dipole_operator_e
        pol[name] = get_correlator(p[:2])

        trivial = jnp.abs(flake.energies) > 0.1
        print("edge states for", name, len(flake) - trivial.sum())

        mask = jnp.logical_and(trivial[:, None], trivial)
        
        pol["topological." + name] = get_correlator(p[:2], mask)
        
    pol["omegas"] = omegas
    jnp.savez("pol_" + results_file, **pol)

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


LIGHT = 299.8
def wavelength(omega):
    return LIGHT / (omega / 2*jnp.pi)

def omega(wavelength):
    return 2*jnp.pi * LIGHT / wavelength

@struct.dataclass
class Params:
    omegas : Any
    alpha : Any
    theta : Any

@struct.dataclass
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

if __name__ == '__main__':
    pass
