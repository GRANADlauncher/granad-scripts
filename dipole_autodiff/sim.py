import jax
import jax.numpy as jnp
from granad import *

for size in [10, 15, 20, 25]:
    idx1, idx2 = 0, 0
    flake = get_graphene().cut_flake(Triangle(size))
    sus = flake.get_ip_green_function(flake.dipole_operator_e[idx1], flake.dipole_operator_e[idx2], jnp.array([0])).real

    def gs(field):
        ham =  flake.hamiltonian + jnp.diag(flake.positions[:, 0] *  field)
        es, _ = jnp.linalg.eigh(ham)
        return 2*es[:(flake.electrons // 2)].sum()

    sus_fun = jax.jacrev(jax.jacrev(gs))
    field = jnp.array([1e-9, 1e-9, 1e-9])
    field = 1e-7
    sus_autodiff = sus_fun(field)
    print(sus, sus_autodiff)
