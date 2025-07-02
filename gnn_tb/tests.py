from main import *

# TODO: visualization
# import jax
# import jax.numpy as jnp  # JAX NumPy
# https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html#flax.linen.Module.tabulate
# print(ggnn.tabulate(jax.random.key(0), jnp.ones((1, 28, 28, 1)), compute_flops=True, compute_vjp_flops=True))


def test_fusion():    
    n_batch = 3
    max_atoms = 4
    rng = jax.random.PRNGKey(42)
    max_supercells = 3

    batch = generate_batch(n_batch, rng, max_atoms, max_supercells)

    spectral_config = {"n_hidden" : 8, "n_out" : 10}
    ggnn_stack_config = {"n_nodes" : max_atoms**2, "n_feats" : 1, "n_batch" : n_batch, "n_hidden_dims" : 4, "n_dense_dim" : 10, "use_residual" : False}
    cnn_config = {"features" : [32, 64], "kernels" : [(3,3), (3,3)], "windows" : [(2,2), (2,2)], "strides" : [(2,2), (2,2)], "n_hidden_features" : 256, "n_out_features" : 10}    
    fusion = FusionMLP(spectral_config, ggnn_stack_config, cnn_config, n_hidden = 10, n_out = 1)
    variables = fusion.init(rng, batch, rng)
    output = fusion.apply(variables, batch, rng)
    assert output.shape == (3,1)


def test_cnn():
    cnn_config = {"features" : [32, 64], "kernels" : [(3,3), (3,3)], "windows" : [(2,2), (2,2)], "strides" : [(2,2), (2,2)], "n_hidden_features" : 256, "n_out_features" : 10}

    cnn = CNN(**cnn_config)

    rng = jax.random.PRNGKey(42)
    img = jax.random.randint(rng, (2, 10,10, 1), 0, 2)
    variables = cnn.init(rng, img)
    output = cnn.apply(variables, img)
    
    assert output.shape == (2, 10)

def test_spectral():
    rng = jax.random.PRNGKey(0)    
    spectral = SpectralMLP(n_out = 10, n_hidden = 32)
    variables = spectral.init(rng, jnp.arange(10).reshape(2,5))
    output = spectral.apply(variables, jnp.arange(10).reshape(2,5))
    assert output.shape == (2, 10)

def test_ggnn_stack():
    n_batch = 4
    max_atoms = 5
    rng = jax.random.PRNGKey(42)
    max_supercells = 3
    n_feats = 1
    n_hidden_dims = 8
    n_dense_dim = 4
    n_nodes = max_atoms**2

    batch = generate_batch(n_batch, rng, max_atoms, max_supercells)
    
    stack = GGNNStack(n_nodes, n_feats, n_batch, n_hidden_dims, n_dense_dim, use_residual = False)

    variables = stack.init(rng, batch, rng)
    output = stack.apply(variables, batch, rng)

    assert output.shape == (n_batch, n_nodes)


def test_ggnn():
    n_batch = 3
    max_atoms = 4
    rng = jax.random.PRNGKey(42)
    max_supercells = 3
    n_feats = 1
    n_hidden_dims = 8
    n_dense_dim = 4
    n_nodes = max_atoms**2

    batch = generate_batch(n_batch, rng, max_atoms, max_supercells)

    # initialize single layer
    ggnn = GGNNLayer(n_nodes, n_feats, n_batch)

    # REMEMBER: flax magic inserts rng / variables before all other args in init / apply
    variables = ggnn.init(rng, batch["node_features"], batch["hamiltonian"], rng, carry = None)
    output, _ = ggnn.apply(variables, batch["node_features"], batch["hamiltonian"], rng, carry = None)

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
