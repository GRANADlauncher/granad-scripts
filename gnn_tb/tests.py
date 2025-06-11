from main import *

# TODO: visualization
# import jax
# import jax.numpy as jnp  # JAX NumPy
# https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.tabulate
# print(ggnn.tabulate(jax.random.key(0), jnp.ones((1, 28, 28, 1)), compute_flops=True, compute_vjp_flops=True))

def test_ggnn():
    n_nodes = 4 # 4 atoms
    n_nodes_large = 100
    n_batch = 3
    n_feats = 1

    # init random number generator
    rng = jax.random.PRNGKey(0)

    # macro hoppings and extensions
    hoppings = jnp.stack([jnp.zeros(n_batch), jnp.linspace(3, 10, n_batch)]).T
    lengths = jax.random.randint(rng, (n_batch,), 80, 100)

    # generate tranining batch
    batch = generate_batch(n_nodes, lengths, hoppings)

    # initialize single layer
    ggnn = GGNNLayer(n_nodes, n_feats, n_batch)

    # REMEMBER: flax magic inserts rng / variables before all other args in init / apply
    variables = ggnn.init(rng, batch["node_features"][:, :, None], batch["hamiltonian"], rng, carry = None)
    output = ggnn.apply(variables, batch["node_features"][:, :, None], batch["hamiltonian"], rng, carry = None)

    assert output.shape == (n_batch, n_nodes, n_feats)

