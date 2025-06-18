# TODO: check numpy version shennanigans
# TODO: more geometries
# TODO: port to jax perhaps (if doesnt work: how bout non-diff linear granad?, need sugar-y abstractions for materials, but this should suffice for now)
import numpy as np
import scipy as scp
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.fft import dct

import matplotlib.pyplot as plt
import time

from granad import *

def remove_dangling_atoms(flake, last_size = np.inf):
    dist = flake.sparse_distance_matrix(flake, max_distance = 1.43)
    d = jnp.squeeze(np.sum(dist, axis = 0))
    flake =  scp.spatial.KDTree(flake.data[d > 2])

    if flake.data.size != last_size:
        return remove_dangling_atoms(flake, flake.data.size)
    
    return flake
    
def get_flake(n = 10):    
    """rectangular graphene flake"""
    grid_range = [(0, n), (0,n)]
    graphene  = MaterialCatalog.get("graphene")
    orbital_positions_uc =  graphene._get_positions_in_uc()
    grid = graphene._get_grid( grid_range )
    orbital_positions = graphene._get_positions_in_lattice( orbital_positions_uc, grid )
    orbital_positions = jnp.unique( orbital_positions, axis = 0)
    
    # no dangling atoms => nearest neighbors < 2
    flake = scp.spatial.KDTree(orbital_positions[:, :2])
    flake  = remove_dangling_atoms(flake)
    return flake

def get_hamiltonian(flake, gap = 0.):
    dist = flake.sparse_distance_matrix(flake, max_distance = 1.43)
    ham = dist.tocoo()
    gap_term = scp.sparse.spdiags( [[gap if i % 2 else 0 for i in range(ham.shape[0])]], diags = 0)    
    ham.data = np.piecewise(ham.data, [ham.data == 0, ham.data > 0], [0, -2.7])
    return ham + gap_term

def purity(rho):
    r2 =  rho @ rho 
    return 3 * r2 - 2 * rho @ r2

def get_density_matrix_gcp(ham, mu = 0, cutoff = 1e-3, max_steps = 100):
    """obtain rho from grand canonical purification"""

    # auxiliary function
    dist = lambda x, y : scp.sparse.linalg.norm(x - y)

    # parameters
    alpha = min(0.5/(ham.max() - mu), 0.5/(mu - ham.min()))
    identity = scp.sparse.identity(ham.shape[0])

    # initial guesses
    rho_0 = alpha*(mu * identity - ham) + 0.5*identity
    rho_1 = purity(rho_0)    
    step = 0
    error = dist(rho_0, rho_1)

    # sc loop    
    while error > cutoff and step < max_steps:
        print(error, type(rho_1))
        rho_0 = rho_1
        rho_1 = purity(rho_1)
        # error = dist(rho_1 @ rho_1, rho_1)
        error = dist(rho_1, rho_0)
        step += 1

    print(f"Finished after {step} / {max_steps}")

    return rho_1

def get_density_matrix_chebyshev():
    return


t = time.time()
flake = get_flake(20)
print(time.time() - t)
# plt.scatter(x = flake.data[:, 0], y = flake.data[:, 1])
# plt.axis('equal')
# plt.show()
# plt.close()

t = time.time()
ham = get_hamiltonian(flake, gap = -10)
print(time.time() - t)
print(ham.shape)

vals, _ = np.linalg.eigh(ham.toarray())
plt.plot(np.arange(ham.shape[0]), vals, 'o')
plt.show()

rho = get_density_matrix_gcp(ham, -4)

# basis vectors as sparse matrix (just permutation of identity lol)
# chebyshev loop => sparse matrix R
# density matrix = R (actually, lol)

def chebyshev_coefficients(f, N):
    # Chebyshev points
    x = np.cos(np.pi * np.arange(N) / (N - 1))
    y = f(x)

    # DCT type-I for Chebyshev coefficients
    c = dct(y, type=1) / (N - 1)
    c[0] /= 2
    c[-1] /= 2
    return c

# apply once per desired column/row neighbourhood
def chebyshev_apply(v):
    T0, T1 = v, Hsc @ v                      # CSR SpMV
    acc    = c[0]*T0 + c[1]*T1
    for k in range(2, M):
        T0, T1 = T1, 2.0 * Hsc @ T1 - T0
        acc   += c[k] * T1
    return acc

# Example
coeffs = chebyshev_coefficients(f, 10)
print("Chebyshev coefficients (via DCT):", coeffs)

# --- assemble sparse TB Hamiltonian (nearest–neighbour graphene) ---
N = 50000
row, col, val = ...                     # neighbour list
H = sp.csr_array((val, (row, col)), shape=(N, N))

# --- cheap spectral bounds ----------------
# Gershgorin row sums: one pass over non-zeros
diag  = H.diagonal()
R     = np.abs(H).sum(axis=1).A1 - np.abs(diag)
Emin  = (diag - R).min()      # safe lower bound
Emax  = (diag + R).max()      # safe upper bound

# optional: tighten upper bound with 20-step power iteration
w_max  = eigsh(H, k=1, which='LA', return_eigenvectors=False)[0]

# --- Chebyshev FOE loop (CPU) -------------
beta = 1/0.025
mu   = 0.0
M    = 200                                   # expansion order
a    = (Emax - Emin)/2
b    = (Emax + Emin)/2
Hsc  = (H - b*sp.eye(N))/a                   # scale to [-1,1]

c = precompute_coeffs(beta, mu, M)           # scalar FFT/DCT once
