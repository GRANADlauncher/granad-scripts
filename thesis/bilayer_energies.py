import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    
    energies = stack.energies  # Assuming 1D array or list
    plt.plot(
        range(len(energies)),
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
plt.tight_layout()
plt.savefig("bilayer_energy_landscape.pdf")
