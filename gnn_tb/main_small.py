from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence, Optional, Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
import flax.struct

import pickle
import matplotlib.pyplot as plt

# --------------- Utils ---------------
Array = jnp.ndarray

EPS = 1e-8

def split_many(key: Array, n: int) -> list[Array]:
    keys = jax.random.split(key, n + 1)
    return list(keys[:-1]) + [keys[-1]]  # last returned for chaining if needed


# --------------- Data ---------------
def get_hamiltonian(positions: Array, ts: Array, tol: float = 1e-6, include_self: bool = True) -> Array:
    """Construct a TB Hamiltonian with radial shells.

    positions: [N, D]
    ts: [K] hopping strengths: t0 (onsite), then shells 1..K-1 by increasing distance.
    tol: distance equality tolerance.
    include_self: include onsite on the diagonal using t0.
    """
    N = positions.shape[0]
    d = jnp.linalg.norm(positions - positions[:, None, :], axis=-1)  # [N,N]
    # Unique sorted distances (excluding 0 for shells)
    unique_d = jnp.unique(jnp.round(d / tol) * tol)
    # ensure 0 is present
    unique_d = jnp.sort(unique_d)

    # Build adjacency by shells using isclose instead of exact equality
    H = jnp.zeros_like(d)
    # Onsite
    t0 = ts[0]
    if include_self:
        H = H + jnp.eye(N) * t0
    # Offsite shells
    def body(i, H):
        # i starts from 1 (skip 0 distance)
        di = unique_d[i]
        mask = jnp.isclose(d, di, rtol=0.0, atol=tol) * (1.0 - jnp.eye(N))
        t = jnp.where(i < ts.shape[0], ts[i], 0.0)
        return H + t * mask

    n_shells = jnp.minimum(ts.shape[0] - 1, unique_d.shape[0] - 1)
    H = jax.lax.fori_loop(1, n_shells + 1, body, H)
    # Symmetrize for numerical safety
    H = 0.5 * (H + H.T)
    return H

@flax.struct.dataclass
class Batch:
    ground_state_per_atom: jnp.ndarray
    image: jnp.ndarray
    node_features: jnp.ndarray
    energies: jnp.ndarray
    energies_mask: jnp.ndarray
    hamiltonian: jnp.ndarray
    n_atoms: jnp.ndarray          # Nt = total atoms in the supercell
    xy: jnp.ndarray               # shape [B, 2] with (x, y)
    n_cell_atoms: jnp.ndarray     # Nc = atoms in microscopic cell

def generate_batch(n_batch: int, key: Array, max_atoms: int, max_supercells: int) -> Batch:
    """Generates a batch of random 2D supercell structures and their TB properties.

    Returns arrays padded to N=max_atoms**2, with masks for padded entries.
    Targets are energy per atom (ground-state / N_atoms) for stability.
    """
    Nmax = max_atoms ** 2
    grid = jnp.stack(jnp.meshgrid(jnp.arange(max_atoms), jnp.arange(max_atoms), indexing="ij"), axis=-1).reshape(-1, 2)
    
    k_masks, k_range, k_coeffs, k_ts, k_noise = jax.random.split(key, 5)
    # random supercell membership mask
    masks = jax.random.bernoulli(k_masks, 0.5, (n_batch, Nmax))
    # make sure at least one true per row
    rand_cols = jax.random.randint(k_masks, (n_batch,), 0, Nmax)
    masks = masks.at[jnp.arange(n_batch), rand_cols].set(True)
    
    # hopping range (number of shells)
    no_neighbors = jax.random.randint(k_range, (n_batch,), 2, 6)  # 2..5 shells is plenty
    # supercell multipliers (x,y)
    coeffs = jax.random.randint(k_coeffs, (n_batch, 2), 1, max_supercells + 1)

    images = []
    supercell_ham = []
    supercell_eigs = []
    supercell_eigmask = []
    node_feats = []
    ground_per_atom = []
    n_atoms_list = []
    xys = []
    n_cell_atoms = []

    # per-sample rngs
    sample_keys = jax.random.split(k_ts, n_batch)

    for b in range(n_batch):
        k_b = sample_keys[b]
        x, y = coeffs[b]
        # positions inside the microscopic cell
        super_idx = jnp.where(masks[b])[0]
        pos_cell = grid[super_idx]  # [Nc,2], Nc>=1

        # displacements
        disp = jnp.stack(jnp.meshgrid(jnp.arange(x), jnp.arange(y), indexing="ij"), axis=-1).reshape(-1, 2)
        disp = disp * max_atoms
        # final positions of structure
        positions = (pos_cell[None, :, :] + disp[:, None, :]).reshape(-1, 2)

        # check layout
        # plt.scatter(positions[:, 0], positions[:, 1])
        # plt.scatter(pos_cell[:, 0], pos_cell[:, 1])
        # plt.show()
        # plt.close()

        # hoppings: onsite + shells
        K = int(no_neighbors[b])
        # smaller magnitudes -> more stable spectra
        tvals = jax.random.uniform(k_b, (K,), minval=-0.5, maxval=0.5)
        # sort by size to make "feu-physical"
        tvals = tvals[jnp.argsort(jnp.abs(tvals))[::-1]]
        tvals = tvals.at[0].set(jax.random.uniform(k_b, (), minval=-0.2, maxval=0.2))  # onsite smaller
        
        # Hamiltonians
        H_cell = get_hamiltonian(pos_cell.astype(jnp.float32), tvals)        
        # spectrum for the cell (pad to Nmax)
        evals_cell, _ = jnp.linalg.eigh(H_cell)
        nc = evals_cell.shape[0]
        pad = Nmax - nc
        evals_padded = jnp.pad(evals_cell, (0, pad))
        mask = jnp.pad(jnp.ones((nc,), dtype=jnp.float32), (0, pad))

        # Ground-state energy for the full structure with positions
        H_full = get_hamiltonian(positions.astype(jnp.float32), tvals)
        evals_full, _ = jnp.linalg.eigh(H_full)        
        # structures are always half-filled
        electrons = H_full.shape[0] // 2
        # fill two per level; if odd, one more on next level
        even = 2.0 * jnp.sum(evals_full[: electrons // 2])
        odd = jnp.where(electrons % 2 == 1, evals_full[electrons // 2], 0.0)
        gs = even + odd  # total energy

        Nat = positions.shape[0]
        gs_per_atom = gs / (Nat + EPS)

        # boolean image (x by y tiled supercell indicator)
        img = jnp.zeros((max_supercells, max_supercells))
        img = img.at[:x, :y].set(1.0)
        images.append(img[..., None])

        # node features (1 feature per potential site). Here: occupancy mask as a feature.
        nf = jnp.zeros((Nmax, 1), dtype=jnp.float32)
        nf = nf.at[super_idx, 0].set(1.0)
        node_feats.append(nf)

        # pad microscopic H to Nmax x Nmax (for GGNN)
        occ_mask = masks[b].astype(jnp.float32)  # shape (Nmax,)
        # start from full-size zero Hamiltonian
        H_pad = jnp.zeros((Nmax, Nmax), dtype=jnp.float32)
        # get indices of occupied sites
        occ_idx = jnp.where(occ_mask == 1)[0]
        # fill the block for the occupied sites
        H_pad = H_pad.at[occ_idx[:, None], occ_idx[None, :]].set(H_cell)

        supercell_ham.append(H_pad)
        supercell_eigs.append(evals_padded)
        supercell_eigmask.append(mask)
        ground_per_atom.append(gs_per_atom)
        n_atoms_list.append(float(Nat))

        nc = evals_cell.shape[0]      # atoms in microscopic cell
        Nat = positions.shape[0]      # total atoms in full supercell
        xys.append(jnp.array([x, y], dtype=jnp.float32))
        n_cell_atoms.append(float(nc))

    batch = Batch(
        ground_state_per_atom=jnp.asarray(ground_per_atom, dtype=jnp.float32),
        image=jnp.asarray(images, dtype=jnp.float32),
        node_features=jnp.asarray(node_feats, dtype=jnp.float32),
        energies=jnp.asarray(supercell_eigs, dtype=jnp.float32),
        energies_mask=jnp.asarray(supercell_eigmask, dtype=jnp.float32),
        hamiltonian=jnp.asarray(supercell_ham, dtype=jnp.float32),
        n_atoms=jnp.asarray(n_atoms_list, dtype=jnp.float32),
        xy=jnp.asarray(xys, dtype=jnp.float32),
        n_cell_atoms=jnp.asarray(n_cell_atoms, dtype=jnp.float32),

    )
    return batch

# --------------- Models ---------------    
class SpectralMoments(nn.Module):
    hidden: int = 16
    out: int = 8

    @nn.compact
    def __call__(self, energies: Array, mask: Array):
        # mask
        m = jnp.clip(mask, 0.0, 1.0)
        n_float = jnp.sum(m, axis=1)                  # (B,)
        n = jnp.maximum(n_float.astype(jnp.int32), 1) # (B,)

        # basic stats
        mean = jnp.sum(energies * m, axis=1, keepdims=True) / (n_float[:, None] + 1e-8)
        var  = jnp.sum(((energies - mean) ** 2) * m, axis=1, keepdims=True) / (n_float[:, None] + 1e-8)
        std  = jnp.sqrt(var + 1e-6)

        # sort valid values to the front: invalids -> +inf so they go to the end
        big = jnp.where(m > 0.0, energies, jnp.inf)
        x_sorted = jnp.sort(big, axis=1)  # ascending

        # helper to build per-row indices for quantiles
        def q_idx(q: float) -> Array:
            r = jnp.floor((n_float - 1.0) * q).astype(jnp.int32)  # (B,)
            r = jnp.clip(r, 0, n - 1)
            return r[:, None]  # (B,1)

        idx0    = jnp.zeros_like(n)[:, None]          # (B,1)
        idxlast = (n - 1)[:, None]                    # (B,1)

        # gather per-row scalars -> shapes all (B,1)
        minv = jnp.take_along_axis(x_sorted, idx0,    axis=1)
        q25  = jnp.take_along_axis(x_sorted, q_idx(0.25), axis=1)
        q50  = jnp.take_along_axis(x_sorted, q_idx(0.50), axis=1)
        q75  = jnp.take_along_axis(x_sorted, q_idx(0.75), axis=1)
        maxv = jnp.take_along_axis(x_sorted, idxlast, axis=1)

        feats = jnp.concatenate([mean, std, minv, maxv, q25, q50, q75], axis=1)  # (B,7)

        h = nn.Dense(self.hidden)(feats); h = nn.relu(h)
        h = nn.Dense(self.out)(h)
        return h

def physics_feats(batch: Batch) -> Array:
    # x, y, Nc, Nt, 1/Nt, tile_count
    x, y = batch.xy[:, 0:1], batch.xy[:, 1:2]
    Nc = batch.n_cell_atoms[:, None]
    Nt = batch.n_atoms[:, None]
    return jnp.concatenate([x, y, Nc, Nt, 1.0 / (Nt + 1e-8), x * y], axis=1)

class LeanModel(nn.Module):
    hidden1: int = 32
    hidden2: int = 16
    @nn.compact
    def __call__(self, batch: Batch):
        spec = SpectralMoments(hidden=16, out=8)(batch.energies, batch.energies_mask)  # ~8 dims
        phy  = physics_feats(batch)                                                    # 6 dims
        x = jnp.concatenate([spec, phy], axis=1)                                       # ~14 dims
        x = nn.Dense(self.hidden1)(x); x = nn.relu(x)
        x = nn.Dense(self.hidden2)(x); x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
    
# --------------- Training ---------------
def make_optimizer(total_steps: int, base_lr: float = 3e-4, weight_decay: float = 1e-4):
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=max(10, total_steps // 20),
        decay_steps=total_steps,
        end_value=base_lr * 0.05,
    )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=sched, weight_decay=weight_decay),
    )
    return opt, sched

@dataclass
class Config:
    n_batch: int = 12
    max_atoms: int = 4
    max_supercells: int = 10
    steps: int = 2_000
    print_every: int = 50
    ema_alpha: float = 0.9

def create_state(key: Array, cfg: Config) -> tuple[TrainState, nn.Module, Batch, Any]:
    model = LeanModel(hidden1=32, hidden2=16)
    bkey, initkey = jax.random.split(key)
    batch = generate_batch(cfg.n_batch, bkey, cfg.max_atoms, cfg.max_supercells)
    params = model.init(initkey, batch)
    # quick sanity on size
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print("param_count:", int(param_count))
    opt, sched = make_optimizer(cfg.steps, base_lr=3e-4, weight_decay=1e-4)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=opt)
    return state, model, batch, sched


def loss_fn(params, apply_fn, batch: Batch):
    preds = apply_fn(params, batch)
    preds = jnp.squeeze(preds)
    # MSE on energy per atom
    loss = jnp.mean((preds - batch.ground_state_per_atom) ** 2)
    # guard
    loss = jnp.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=1e6)
    return loss


@jax.jit
def train_step(state: TrainState, batch: Batch):
    grad_fn = jax.value_and_grad(lambda p: loss_fn(p, state.apply_fn, batch))
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train(seed: int = 0, save_dir: str = "."):
    key = jax.random.PRNGKey(seed)
    cfg = Config()

    state, model, _, sched = create_state(key, cfg)

    ema_loss = None
    losses = []
    rngs = jax.random.split(key, cfg.steps)

    for step in range(cfg.steps):
        # fresh batch every step
        batch = generate_batch(cfg.n_batch, rngs[step], cfg.max_atoms, cfg.max_supercells)
        print(batch.ground_state_per_atom)
        state, loss = train_step(state, batch)
        loss_val = float(loss)
        ema_loss = loss_val if ema_loss is None else cfg.ema_alpha * ema_loss + (1 - cfg.ema_alpha) * loss_val
        losses.append(loss_val)

        if step % cfg.print_every == 0:
            print(f"step {step:04d} | loss {loss_val:.5f} | ema {ema_loss:.5f}")

        # simple early stop if loss diverges
        print(loss_val)
        if not math.isfinite(loss_val):
            print("Divergence detected; stopping early.")
            break
        # gentle early stop when very low
        if ema_loss is not None and ema_loss < 1e-4 and step > 200:
            print("Converged; stopping early.")
            break

    # Save
    with open(f"{save_dir}/params.pkl", "wb") as f:
        pickle.dump(state.params, f)
    jnp.savez(f"{save_dir}/loss.npz", loss=jnp.array(losses))

    # quick plot
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (per-atom)")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss.pdf")
    plt.close()

    return state, model


# --------------- Validation ---------------

def validate(params_path: str = "params.pkl", seed: int = 123):
    key = jax.random.PRNGKey(seed)
    cfg = Config(n_batch=128)
    state, model, batch, _ = create_state(key, cfg)
    # load params
    with open(params_path, "rb") as f:
        params = pickle.load(f)

    preds = model.apply(params, batch)
    preds = jnp.squeeze(preds)

    # compare against per-atom energy
    x = batch.n_atoms
    y = batch.ground_state_per_atom

    plt.figure()
    plt.plot(x, y, "o", label="data")
    plt.plot(x, preds, "o", label="prediction")
    plt.xlabel("Structure Size (atoms)")
    plt.ylabel("Ground State Energy / atom")
    plt.legend()
    plt.grid(True)
    plt.savefig("pred.pdf")
    plt.close()


if __name__ == "__main__":
    train()
    validate()
