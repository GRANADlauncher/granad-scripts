from main import *

# TODO: visualization
# import jax
# import jax.numpy as jnp  # JAX NumPy
# https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.tabulate
# print(ggnn.tabulate(jax.random.key(0), jnp.ones((1, 28, 28, 1)), compute_flops=True, compute_vjp_flops=True))

def test_ggnn():
    n_nodes = 4 # atoms
    n_feats = 1
    n_batch = 3

    min_cells = 4
    max_cells = 5

    # init random number generator
    rng = jax.random.PRNGKey(0)

    # batch
    batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)

    # initialize single layer
    ggnn = GGNNLayer(n_nodes, n_feats, n_batch)

    # REMEMBER: flax magic inserts rng / variables before all other args in init / apply
    variables = ggnn.init(rng, batch["node_features"][:, :, None], batch["hamiltonian"], rng, carry = None)
    output = ggnn.apply(variables, batch["node_features"][:, :, None], batch["hamiltonian"], rng, carry = None)

    assert output.shape == (n_batch, n_nodes, n_feats)

    

def test_spectral_vanilla():
    from vanilla import FusionMLP, generate_batch
    
    n_batch = 3
    n_nodes = 4
    min_cells = 1
    max_cells = 100 # 400 atoms
    
    # Model    
    rng = jax.random.PRNGKey(42)
    
    # batch
    batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)

    # setup model
    fusion = FusionMLP(1, 10, 20, 10, 20)
    params = fusion.init(rng, batch["energies"], batch["cell_arr"])
    output = fusion.apply(params, batch["energies"], batch["cell_arr"])

    print("# params ", sum(x.size for x in jax.tree_leaves(params)))

    assert output.shape == (n_batch,1)
