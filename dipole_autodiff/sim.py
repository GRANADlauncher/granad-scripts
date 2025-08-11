import jax
import jax.numpy as jnp
from granad import *

import matplotlib.pyplot as plt

sizes = [5, 10, 15, 20, 22, 23, 25, 27, 29, 30, 32, 33, 35]
ad = []
ip = []
atoms = []

for size in sizes:
    idx1, idx2 = 0, 0
    flake = get_graphene().cut_flake(Triangle(size))
    sus = -flake.get_ip_green_function(flake.dipole_operator_e[idx1], flake.dipole_operator_e[idx2], jnp.array([0])).real
    ip.append(sus)

    atoms.append(len(flake))
    
    def gs(field):
        ham =  flake.hamiltonian + jnp.diag(flake.positions[:, 0] *  field)
        es, _ = jnp.linalg.eigh(ham)
        # spin degeneracy => mult by 2
        return -2*es[:(flake.electrons // 2)].sum()

    sus_fun = jax.jacrev(jax.jacrev(gs))
    field = 1e-7 # zero => nans
    sus_autodiff = sus_fun(field)
    ad.append(sus_autodiff)
    
    print(sus, sus_autodiff)

# Create figure and axis
fig, ax = plt.subplots(figsize=(7, 5))

# Plot IP and autodiff susceptibilities
ax.plot(atoms, ip, marker='o', linestyle='-', color='tab:blue', label='IP')
ax.plot(atoms, ad, marker='s', linestyle='--', color='tab:orange', label='AD')

# Axis labels
ax.set_xlabel("Number of Atoms", fontsize=11)
ax.set_ylabel("Static Polarizability (a.u.)", fontsize=11)

# Title and grid
ax.set_title("Static Polarizability: AD vs IP", fontsize=13)
ax.grid(True, linestyle=':', alpha=0.6)

# Legend
ax.legend(loc='best', fontsize='small', frameon=False)

# Save high-quality PDF
plt.tight_layout()
plt.savefig("static_pol_ad_vs_ip.pdf", dpi=300, bbox_inches='tight')
plt.show()
