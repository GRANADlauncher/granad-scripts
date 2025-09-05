import matplotlib.pyplot as plt
from granad import *


flake = MaterialCatalog.get("graphene").cut_flake( Triangle(10) )


diff = flake.energies - flake.energies[:, None]
m = jnp.abs(diff).max()
omegas = jnp.linspace(0, m + 1, 100)

ip = flake.get_ip_green_function(flake.dipole_operator_e[0], flake.dipole_operator_e[0], omegas = omegas)

cs = jnp.linspace(0, 1, 5)
rpa = []
for c in cs:
    rpa.append(flake.get_polarizability_rpa(omegas = omegas, polarization = 0, coulomb_strength = c, hungry = 1))
    
# Create figure
fig, ax = plt.subplots(figsize=(7, 5))

# Plot -RPA spectra
for i, r in enumerate(rpa):
    ax.plot(omegas, r.imag, label=r"RPA, $\lambda=$" f"{cs[i]}", alpha=0.7, linewidth=1.5)

# Plot IP spectrum
ax.plot(omegas, -ip.imag, label="Independent Particles (IP)", color='black', linewidth=2, linestyle='--')

# Labels and formatting
ax.set_xlabel("Energy (eV)", fontsize=11)
ax.set_ylabel("Absorption (arb. units)", fontsize=11)
ax.set_title("RPA vs Independent Particles Spectrum", fontsize=13)
ax.grid(True, linestyle=':', alpha=0.6)

# Legend
ax.legend(fontsize='small', frameon=False, loc='upper right')

# Tight layout and save
plt.tight_layout()
plt.savefig("rpa_vs_ip.pdf", dpi=300, bbox_inches='tight')
plt.show()
