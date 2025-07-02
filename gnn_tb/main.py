# TODO: refactor batch to return sample of geometries / structure as an NWHC image
# TODO: batch with SK integrals / realistic structures
# TODO: train, test, validate
# TODO: cross validation
# TODO: think of sensible node feats in tb model
# TODO: global refactor of __call__ to accept batch
# TODO: how to cleanly handly hyperparameters?

import flax.linen as nn
import jax
import jax.numpy as jnp
from typing import Sequence, Optional
import optax
from flax.serialization import to_state_dict, from_state_dict
import pickle

import matplotlib.pyplot as plt

def generate_batch(n_batch, rng, max_atoms, max_supercells):
    """Generates training batch. Random 1D or 2D structure, characterized by supercell and TB Hamiltonian.

    Args:
        n_batch : batch size
        rng : prng key
        max_atoms : max number of atoms in supercell
        max_supercells : max number of overall supercells

    Returns:
        Dictionary with keys
    
        ground_state : energy of structure
        image : supercell boolean
        node_features : N_atoms x N_features
        energies : supercell energies
        hamiltonian : supercell hamiltonian
    """
    def get_hamiltonian(pos, ts):
        d = jnp.linalg.norm(pos - pos[:, None], axis = -1)
        ds = jnp.unique(d)

        ham = jnp.zeros_like(d)
        for i, t in enumerate(ts):
            ham += t * (d == ds[i])
            
        return ham

    # supercell
    grid = jnp.array([[i, j] for i in range(0, max_atoms) for j in range(0, max_atoms)])
    masks = jax.random.randint(rng, (n_batch, max_atoms**2), 0, 2).astype(bool)  # randomly delete atoms in supercell
    rng, _ = jax.random.split(rng)
    no_neighbors = jax.random.randint(rng, (n_batch,), 1, 10)  # randomly assign hopping range
    rng, _ = jax.random.split(rng)
    

    # return vals
    imgs = []
    supercell_ham = []
    supercell_energies = []    
    ground_states = []
    node_features = []
    
    ## structure
    coefficients = jax.random.randint(rng, (n_batch, 2), 1, max_supercells)
    for batch_idx, (x,y) in enumerate(coefficients):
        from itertools import product
        
        # displacement vectors for positions in supercell
        prod = list(product(range(x+1), range(y+1)))
        displacements = jnp.array(prod) * (max_atoms)

        # final positions of structure
        supercell_masked = grid[masks[batch_idx]]
        positions = (supercell_masked + displacements[:, None, :]).reshape(supercell_masked.shape[0] * displacements.shape[0], 2)

        # hopping rates 
        ts = jax.random.randint(rng, (no_neighbors[batch_idx],), 1, 100) / 100
        rng, _ = jax.random.split(rng)
        
        # hamiltonian of supercell
        ham_sup = get_hamiltonian(supercell_masked, ts)
        supercell_ham.append(jnp.pad(ham_sup, (0, max_atoms**2 - ham_sup.shape[0])))

        # energies
        vals, _ = jnp.linalg.eigh(ham_sup)
        supercell_energies.append(jnp.pad(vals, (0, max_atoms**2 - vals.size)))

        # node features
        node_features.append( jnp.pad(jnp.ones(vals.size), (0, max_atoms**2 - vals.size)) )
            
        # ground state energy of structure
        ham = get_hamiltonian(positions, ts)        
        electrons = ham.shape[0] // 2
        even = 2 * vals[:electrons//2].sum()
        odd = vals[(electrons // 2) + (electrons % 2)] * (electrons % 2)
        ground_states.append(even + odd)

        # boolean mask representing displacement
        img = [[1 if (i,j) in prod else 0 for i in range(max_atoms)] for j in range(max_atoms)]
        imgs.append(img)

        # print(jnp.array(img))
        # plt.scatter(positions[:, 0], positions[:, 1])
        # plt.scatter(grid[masks[batch_idx]][:, 0], grid[masks[batch_idx]][:, 1])
        # for idx, pos in enumerate(positions):
        #     plt.annotate(str(idx), (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')
        # plt.show()


    imgs = jnp.array(imgs)[..., None]
    node_features = jnp.array(node_features)
    energies = jnp.array(supercell_energies)
    hamiltonians = jnp.array(supercell_ham)
    ground_states = jnp.array(ground_states)
    
    return {
        "ground_state" : ground_states,
        "node_features" : node_features,
        "energies" : energies,
        "hamiltonian" : hamiltonians,
        "image" : imgs    
    }

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
    node_features = jnp.array([1 for i in range(n)])[:, None]

    return {
        "ground_state" : even + odd,
        "energies" : energies,
        "hamiltonian" : hamiltonian,
        "node_features" : node_features
    }

## GEOMETRY EMBEDDING ##
# input data with dimensions (batch, spatial_dims…, features) => (batch, N, N, 1)
# learns representation of supercell geometry from cnns
class CNN(nn.Module):
    """A simple CNN model. Input: 2D boolean image indicating supercell position in structure. Output: Latent Rep."""

    features : list[int]
    kernels : list[tuple]
    windows : list[tuple]
    strides : list[tuple]
    n_hidden_features : int
    n_out_features : int
  

    @nn.compact
    def __call__(self, x):
        for i, feats in enumerate(self.features):
            x = nn.Conv(features=feats, kernel_size=self.kernels[i])(x)
            x = nn.relu(x)
            x = nn.avg_pool(x, window_shape=self.windows[i], strides=self.strides[i])

        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=self.n_hidden_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.n_out_features)(x)
    
        return x

## SUPERCELL EMBEDDING ##
# energy / spectral representation
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
    
# geometry of microscopic cell
class GGNNLayer(nn.Module):
    """Gated Graph Neural Network.

    Each orbital is defined by a feature list. This layer takes the list n^{t} at t and returns list at t+1 according to

    m^{t+1}_i = F[H] \cdot n^{t}
    n^{t+1}_i = G(n^{t}, m^{t+1})

    where F = id, and G is a Gated Recurrent Unit.

    With B batch dim, N number of orbitals, F number of features per orbital, this is a function F

    F : X, H -> Y

    where X represents the structure of the supercell and is of dimension B x N x F
    and H is the Hamiltonian of dimension B x N x N
    
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
        rng, _ = jax.random.split(rng)

        # message : n_batch x n_nodes x n_feats
        message = jnp.einsum('bij,bjf->bif', edge_tensor, node_feats)

        # TODO: if this sucks, check if one grucell per node performs better
        # GRUCell: "All dimensions except the final are considered batch dimensions."    
        size_flat = self.n_nodes * self.n_feats
        message = message.reshape((self.n_batch, size_flat))

        # TODO: push up reshape op
        # TODO: one GRU cell per node
        if carry is None:
            carry = nn.GRUCell(features = size_flat).initialize_carry(rng, (self.n_batch, size_flat) )
        else:
            carry = carry.reshape((self.n_batch, size_flat))
        node_feats, _ = nn.GRUCell(features = size_flat)(carry, message)

        return node_feats.reshape(self.n_batch, self.n_nodes, self.n_feats), rng


class GGNNStack(nn.Module):
    """Stack of N GGNNs. Takes in orbital feature list and extracts the final feature via    

    R = \sum \sigma(F(n^{N}, n^{0})) \odot G(n^{N})

    where F, G are regression layers and the sum runs over the neighbors of each node.
    """
    n_nodes : int
    n_feats : int
    n_batch : int
    n_hidden_dims : int
    n_dense_dim : int
    use_residual : bool
    
    @nn.compact
    def __call__(self, batch, rng):
        node_feats, edge_tensor = batch["node_features"], batch["hamiltonian"]

        # carry for gru
        carry = None
        for _ in range(self.n_hidden_dims):
            ggnn = GGNNLayer(self.n_nodes, self.n_feats, self.n_batch)
            carry, rng = ggnn(node_feats, edge_tensor, rng = rng, carry = carry)

            # TODO: makes sense / skip layers?
            # residual network => add old and new node_feats
            if self.use_residual:
                carry += node_feats
            node_feats = carry

        # readout :  R = sigma(nn(h_f, h_0)) \odot nn(h_f)
        feats = jnp.stack([node_feats, batch["node_features"]], axis = 1)
        stage1 = nn.Dense(node_feats.shape[1])(feats.reshape(self.n_batch, self.n_feats * self.n_nodes * 2))
        stage2 = nn.Dense(node_feats.shape[1])(node_feats.reshape(self.n_batch, self.n_feats * self.n_nodes))
        readout = nn.sigmoid(stage1) * stage2
        return readout

## FEATURE FUSION ##
class FusionMLP(nn.Module):
    spectral_config : dict    
    ggnn_stack_config : dict    
    cnn_config : dict

    n_hidden : int
    n_out : int

    @nn.compact
    def __call__(self, energies, batch, rng, structure):
        spectrum = SpectralMLP(**self.spectral_config)(energies)
        
        stack = GGNNStack(**self.ggnn_stack_config)(batch, rng)
        
        geometry = CNN(**self.cnn_config)(structure)

        x = jnp.concatenate([geometry, spectrum, stack], axis = 1)        
        x = nn.Dense(self.n_hidden)(x)        
        x = nn.relu(x)
        x = nn.Dense(self.n_out)(x)
        
        return x           

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
    
    # Model    
    rng = jax.random.PRNGKey(42)
    
    # batch
    batch, rng = generate_batch(rng, min_cells, max_cells, n_nodes, n_batch)

    # setup model
    model = FusionMLP(1, 10, 20, 10, 20)
    model.init(rng, batch["energies"], batch["cell_arr"])
    # print(model.tabulate(rng, (batch["energies"], batch["cell_arr"]), compute_flops=True, compute_vjp_flops=True))

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
    plt.ylabel("Ground State Energy (eV)")
    plt.legend()
    plt.savefig(f"pred.pdf")
    plt.close()
    

if __name__ == '__main__':
    # train()
    # plot_loss()
    # validate()
    
    n_batch = 3
    max_atoms = 4
    min_atoms = 1
    rng = jax.random.PRNGKey(42)
    max_supercells = 3    
    batch = generate_batch(n_batch, rng, max_atoms, max_supercells)
