import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional
import optax
from flax.serialization import to_state_dict, from_state_dict
import pickle

import matplotlib.pyplot as plt

## GEOMETRIES ##
def chain(n, electrons, ts):
    """features of simple metal chain 

    n : length
    electrons : total number of electrons put in the system
    ts : hoppings [onsite, nn, nnn, nnnn, ...]
    """
    
    pos = jnp.arange(n)

    # initialize
    hamiltonian = jnp.zeros((n, n))

    # sum over neighborhoods in square lattice
    delta_p = jnp.abs(pos[:, None] - pos)
    for i, t in enumerate(ts):
        hamiltonian += (jnp.abs(delta_p - i) < 0.1) * t
        
    energies, _ = jnp.linalg.eigh(hamiltonian)

    # fill energies
    even = 2 * energies[:electrons//2].sum()
    
    #  uneven number of electrons
    odd = energies[(electrons // 2) + (electrons % 2)] * (electrons % 2)

    # each node is characterized by its atom kind
    node_features = jnp.array([1 for i in range(n)])

    return {
        "ground_state" : even + odd,
        "energies" : energies,
        "hamiltonian" : hamiltonian,
        "node_features" : node_features
    }

def generate_batch(rng, min_cells, max_cells, n_nodes, n_batch):    
    """Generate a training batch.

    Arguments:
        rng: PRNG
        min_cells : min_number of units cells
        max_cells : max_number of units cells
        n_nodes : number of nodes / atoms in supercell
        n_batch : batch size
       
    Returns: Dict with keys
       ground_state : gs energy of the structure
       cell_arr : boolean array indicating presence / absence of supercell
       energies : IP energies of supercell
       hamiltonian : IP hamiltonian of supercell

       array: new PRNG key
    
    """
    
    # chains up to max_cells * n_nodes atoms
    ns = jax.random.randint(rng, (n_batch,), min_cells, max_cells)

    # scale nn hopping rates array to floats
    ts = jax.random.randint(rng, (n_batch,), 1, 100)
    ts /= ts.max() * 4
    ts = jnp.vstack([jnp.zeros_like(ts), ts]).T

    # TARGET: metallic chain gs energies 
    energies = jnp.array([chain(n * n_nodes, n * n_nodes, ts[i])["ground_state"] for i, n in enumerate(ns)])

    # MICROSCOPIC PREDICTOR: hamiltonians and energies of supercell
    f_chains_train = lambda t: chain(n_nodes, n_nodes, t)
    chains_train = jax.vmap(f_chains_train, in_axes = 0, out_axes = 0)(ts)

    # MACROSCOPIC PREDICTOR: boolean array indicating presence / absence of supercells
    cell_arr = ns[:, None] <= jnp.arange(max_cells * n_nodes)

    rng, _ = jax.random.split(rng)
    
    # put all together in dict
    return chains_train | {"ground_state" : energies, "cell_arr" : cell_arr}, rng

## SPECTRAL EMBEDDING ##
class SpectralMLP(nn.Module):
    n_out : int
    n_hidden : int

    @nn.compact
    def __call__(self, energies):
        x = energies
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.n_hidden)(x)                 # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(self.n_hidden)(x)                 # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(self.n_out)(x)       # shape inference
        return x
    
## GEOMETRY EMBEDING ##
class GeometryMLP(nn.Module):
    n_out : int
    n_hidden : int

    @nn.compact
    def __call__(self, cell_arr):
        x = cell_arr
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.n_hidden)(x)                 # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(self.n_hidden)(x)                 # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(self.n_out)(x)       # shape inference
        return x
    
## FEATURE FUSION ##
class FusionMLP(nn.Module):
    n_out : int
    spectrum_n_out : int
    spectrum_n_hidden : int
    geometry_n_out : int
    geometry_n_hidden : int

    @nn.compact
    def __call__(self, energies, cell_arr):
        spectrum = SpectralMLP(self.spectrum_n_out, self.spectrum_n_hidden)(energies)        
        geometry = GeometryMLP(self.geometry_n_out, self.geometry_n_hidden)(cell_arr)
        x = jnp.concatenate([geometry, spectrum], axis = 1)        
        x = nn.Dense(10)(x)                 # create inline Flax Module submodules
        x = nn.relu(x)
        x = nn.Dense(self.n_out)(x)       # shape inference
        return x           

## STRUCTURE EMBEDDING ##
class GGNNLayer(nn.Module):
    """Gated Graph Neural Network.

    Each orbital is defined by a feature list. This layer takes the list n^{t} at t and returns list at t+1 according to

    m^{t+1}_i = F[H] \cdot n^{t}
    n^{t+1}_i = G(n^{t}, m^{t+1})

    where F = id, and G is a Gated Recurrent Unit.
    
    TODO: gain anything from making F MLP? where F = \sigma(W H + B) with W, B regression tensors. G is a Gated Recurrent Unit.
    TODO: generalize edge_tensor to more feats
    """
    
    n_nodes : int
    n_feats : int
    n_batch : int
    
    @nn.compact
    def __call__(self, node_feats, edge_tensor, rng, carry = None):
        """
        
        node_feats: n_batch x n_nodes x n_feats tensor for tb orbitals
        edge_tensor: n_batch x n_nodes x n_nodes for tb hamiltonian
        rng: prng
        carry: n_batch x n_nodes x n_feats tensor of previous layer
        """
        ## MAYBE: regression embedding of TB Hamiltonian ##

        # message : n_batch x n_nodes x n_feats
        message = jnp.einsum('bij,bjf->bif', edge_tensor, node_feats)

        # GRUCell: "All dimensions except the final are considered batch dimensions."
        size_flat = self.n_nodes * self.n_feats
        message = message.reshape((self.n_batch, size_flat))
        if carry is None:
            carry = nn.GRUCell(features = size_flat).initialize_carry(rng, (self.n_batch, size_flat) )
        node_feats, _ = nn.GRUCell(features = size_flat)(carry, message)

        return node_feats.reshape(self.n_batch, self.n_nodes, self.n_feats)

# TODO: implement
class GGNNStack(nn.Module):
    """Stack of N GGNNs. Takes in orbital feature list and extracts the final feature via    

    R = \sum \sigma(F(n^{N}, n^{0})) \odot G(n^{N})

    where F, G are regression layers and the sum runs over the neighbors of each node.
    """
    @nn.compact
    def __call__(self):
        return
    
def train():
    @jax.jit
    def loss_fn(params, energies, cell_arr, targets):
        # batch is dict containing supercell energies and geometry representation
        preds = model.apply(params, energies, cell_arr)
        
        # remove singleton dimension
        preds = jnp.squeeze(preds)

        # compute loss
        loss = jnp.mean((preds - targets) ** 2)
        return loss

    @jax.jit
    def train_step(params, energies, cell_arr, targets, opt_state):
        # gradients
        grads = jax.grad(loss_fn)(params, energies, cell_arr, targets)
        
        # update params
        updates, opt_state = optimizer.update(grads, opt_state)        
        new_params = optax.apply_updates(params, updates)

        # loss with new params
        loss = loss_fn(new_params, energies, cell_arr, targets)

        # return info
        return new_params, opt_state, loss
    
    # Some hyperparameters
    n_batch = 32
    n_nodes = 4
    min_cells = 1
    max_cells = 100 # 400 atoms
    lr = 1e-3
    num_epochs = 505
    
    # Model    
    rng = jax.random.PRNGKey(42)
    
    # batch
    batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)

    # setup model
    model = FusionMLP(1, 10, 20, 10, 20)
    params = model.init(rng, batch["energies"], batch["cell_arr"])
    
    # Optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # Training loop
    loss_arr = []
    for epoch in range(num_epochs):
        batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)
        energies, cell_arr, targets = batch["energies"], batch["cell_arr"], batch["ground_state"]
        params, opt_state, loss = train_step(params, energies, cell_arr, targets, opt_state)
        loss_arr.append(loss)
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

    # save params and loss
    with open('params.pkl', 'wb') as f:
        pickle.dump(params, f)
    jnp.savez("loss.npz", loss=jnp.array(loss_arr))
            
    return model, params

# Plot loss
def plot_loss(filename='loss.npz'):
    with jnp.load(filename) as data:
        loss_arr = data['loss']

    plt.plot(loss_arr)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig("loss.pdf")
    plt.close()

def validate():
    # Some hyperparameters
    n_batch = 32
    n_nodes = 4
    min_cells = 1
    max_cells = 100 # 400 atoms
    lr = 1e-3
    num_epochs = 500
    
    # Model    
    rng = jax.random.PRNGKey(42)
    
    # batch
    batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)

    # setup model
    model = FusionMLP(1, 10, 20, 10, 20)
    model.init(rng, batch["energies"], batch["cell_arr"])

    with open('params.pkl', 'rb') as f:        
        params = pickle.load(f)

    # new batch of larger dims
    batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)
    preds = model.apply(params, batch["energies"], batch["cell_arr"])
    preds = jnp.squeeze(preds)
    targets = batch["ground_state"]
    loss = jnp.mean((preds - targets) ** 2)
    
    print(loss)

    plt.plot(batch["cell_arr"].sum(axis = 1), targets, 'o', label = "data")
    plt.plot(batch["cell_arr"].sum(axis = 1), preds, 'o', label = "prediction")
    plt.xlabel("Structure Size")
    plt.ylabel("Ground State Energy")
    plt.legend()
    plt.savefig(f"pred.pdf")


if __name__ == '__main__':
    # train()
    plot_loss()
    validate()
    
