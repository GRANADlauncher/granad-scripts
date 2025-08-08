import jax.numpy as jnp
import matplotlib.pyplot as plt

from granad import *

def interlayer_hopping_factory( coupling ):
    def interlayer_hopping( distance ):
        return coupling * jnp.exp( -100*(jnp.linalg.norm(distance) - 1.0)**2 )    
    return interlayer_hopping


triangle = Triangle(15)

flake = MaterialCatalog.get("graphene").cut_flake(triangle)
flake.shift_by_vector( [0,0,1]  )
print(flake.positions)
second_flake = MaterialCatalog.get("graphene").cut_flake(triangle)
stack = flake + second_flake

# Set up the plot
plt.figure(figsize=(8, 5))

# Coupling values to sweep
couplings = [0, 1.0, 1.5, 2.0, 3.0]

# Plot energy spectra for each coupling
for coupling in couplings:
    interlayer_hopping = interlayer_hopping_factory(-coupling * 2.66)
    stack.set_hamiltonian_groups(flake, second_flake, interlayer_hopping)
    
    energies = stack.energies  # Assuming 1D array or list
    plt.plot(
        range(len(energies)),
        energies,
        'o',
        label=fr"$t / t_0$ = {coupling}",
        alpha=0.8,
        markersize=4
    )

# Final plot tweaks
plt.title("Energy Spectrum vs. Interlayer Coupling", fontsize=13)
plt.xlabel("Energy Eigenstate")
plt.ylabel("Energy (eV)")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(title="Coupling Strength", fontsize='small', frameon=False)
plt.tight_layout()
plt.savefig("bilayer_energy_landscape.pdf")
