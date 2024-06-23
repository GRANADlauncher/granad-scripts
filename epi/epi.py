import jax.numpy as jnp
import matplotlib.pyplot as plt
from granad import *

n, m = 30, 30
flake = MaterialCatalog.get("graphene").cut_flake( Rhomboid(n,m) )

fs = jnp.linspace(0, flake.energies.max(), 100)
epis = []
for f in fs:
    res = flake.master_equation(
        relaxation_rate = 1/10,
        illumination = Wave(frequency = f, amplitudes = [1e-5, 0, 0]),
        end_time = 40)
    epis.append(flake.get_epi(res.final_density_matrix, f))

plt.plot(fs, epis)
plt.xlabel(r'$\hbar \omega$ (eV)')
plt.ylabel('EPI')
plt.savefig("epi.pdf")
