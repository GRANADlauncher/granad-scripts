import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from granad import *

# couplings from 10.1103/PhysRevMaterials.2.034004

INTERLAYER_DISTANCE = 3.35

def coupling(dvec):
    d0 = INTERLAYER_DISTANCE
    a0 = 1.42
    delta = 0.3187 * a0
    
    d = jnp.linalg.norm(dvec)
    dz = dvec[-1]
    exp1 = jnp.exp(-(d - a0) / delta)
    exp2 = jnp.exp(-(d - d0) / delta)
    # f = jax.lax.cond( d >= 0.5 * d0,
    #                   lambda d : jnp.heaviside(d - 1.5*d0, 0.),
    #                   lambda d : jnp.heaviside(d - 1.5*jnp.sqrt(3)*a0, 0.),
    #                   d
    #                  )
    f = 1
    
    V1 = -2.7 * exp1 * f
    V2 = 0.48 * exp2 * f

    return jnp.nan_to_num(V1*(1 - (dz/d)**2) + V2*(dz/d)**2, posinf=0, neginf = 0)


def cut_energies(energies, e_max = None, e_min = None):
    e_max = (e_max or energies.max()) 
    e_min = (e_min or energies.min())
    widening = (e_max - e_min) * 0.01 # 1% larger in each direction
    e_max += widening
    e_min -= widening
    energies_filtered_idxs = jnp.argwhere( jnp.logical_and(energies <= e_max, energies >= e_min))
    state_numbers = energies_filtered_idxs[:, 0]
    energies_filtered = energies[energies_filtered_idxs]
    return state_numbers, energies_filtered

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

print(coupling(jnp.array([0, 0, INTERLAYER_DISTANCE])))

triangle = Triangle(15)
flake = MaterialCatalog.get("graphene").cut_flake(triangle)
flake.shift_by_vector( [0,0,INTERLAYER_DISTANCE]  )
second_flake = MaterialCatalog.get("graphene").cut_flake(triangle)
stack = flake + second_flake

# Set up the plot
plt.figure(figsize=(8, 5))

angles = jnp.array([0, 1/4, 1/3]) * jnp.pi
labels = [r"$\phi = 0$", r"$\phi = \frac{\pi}{4}$", r"$\phi = \frac{\pi}{3}$"]
stack.set_hamiltonian_groups(stack, stack, coupling)

# Plot energy spectra for each coupling
for i, angle in enumerate(angles):
    flake.rotate(flake.positions[flake.center_index], angle)
    stack._recompute = True
    e_max, e_min = 2.5, -1.5
    
    state_numbers, energies = cut_energies(stack.energies, e_max, e_min)
    plt.axhline()
    plt.plot(
        state_numbers,
        energies,
        'o',
        label=labels[i],
        alpha=0.8,
        markersize=4
    )
    flake.rotate(flake.positions[flake.center_index], -angle)


# Final plot tweaks
plt.xlabel("Energy Eigenstate")
plt.ylabel("Energy (eV)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="Rotation angle", fontsize='small', frameon=False)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(e_min, e_max)    
plt.tight_layout()
plt.savefig("bilayer_energy_landscape.pdf")
