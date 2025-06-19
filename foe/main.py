# TODO: make (grand) canonical purifcation fast by keeping matrices sparse
# TODO: remove hardcoding for flexible bounds
# TODO: time propagation by hamiltonian: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm.html
# TODO: check numpy version shennanigans
# TODO: more geometries
# TODO: implement chebyshev density matrix and chebyshev expansions
# TODO: how bout non-diff linear granad?, need sugar-y abstractions for materials, but this should suffice for now
import time

import numpy as np
import jax.numpy as jnp
import scipy as scp

import matplotlib.pyplot as plt

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
    orbital_positions = np.unique( orbital_positions, axis = 0)
    
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

# following PhysRevB.58.12704.pdf
def get_density_matrix_gcp(ham, mu = 0, cutoff = 1e-3, max_steps = 100):
    """obtain rho from grand canonical purification"""

    # auxiliary function
    dist = lambda x, y : scp.sparse.linalg.norm(x - y)

    # parameters
    ma = 5
    mi = -15
    alpha = min(0.5/(ma - mu), 0.5/(mu - mi))
    identity = scp.sparse.identity(ham.shape[0]) 

    # initial guesses
    rho_0 = alpha*(mu * identity - ham) + 0.5*identity
    rho_1 = purity(rho_0)    
    step = 0
    error = dist(rho_0, rho_1)

    # sc loop    
    while error > cutoff and step < max_steps:
        rho_0 = rho_1
        rho_1 = purity(rho_1)
        error = dist(rho_1, rho_0)
        step += 1

        if step % 10 == 0:
            print(error, step)


    print(f"Finished after {step} / {max_steps} with error {error}")

    return rho_1

def sparsify(r):
    r.data[np.abs(r.data) < 1e-4] = 0
    r.eliminate_zeros()
    return r

def canonical_iteration(rho_0):
    r2 = rho_0 @ rho_0
    r3 = r2 @ rho_0

    c = (r2 - r3).trace() / (rho_0 - r2).trace()

    if c >= 0.5:
        rho_1 = ((1 + c) * r2 - r3) / c
    else:
        rho_1 = ((1 - 2*c) * rho_0 + (1 + c)* r2 - r3) / (1 - c)

    return rho_1.multiply(cutoff_matrix)

# following PhysRevB.58.12704.pdf
def get_density_matrix_cp(ham, cutoff = 1e-6, max_steps = 100):
    """obtain rho from canonical purification"""

    # auxiliary function
    dist = lambda x, y : scp.sparse.linalg.norm(x - y)
    
    # parameters
    mu = ham.trace() / (ham.shape[0] // 2)
    E_max = 5
    E_min = -15
    # alpha = min(0.5/(ma - mu), 0.5/(mu - mi))
    identity = scp.sparse.identity(ham.shape[0])

    alpha = 1.0 / (E_max - E_min)          # maps spectrum to (0,1)
    rho_0 = (E_max*identity - ham) * alpha        #   ρ₀ = (E_max I - H)/(E_max-E_min)

    # initial guesses
    # rho_0 = alpha*(mu * identity - ham) + 0.5*identity
    rho_1 = canonical_iteration(rho_0)
    step = 0
    error = dist(rho_0, rho_1)

    # sc loop    
    while error > cutoff and step < max_steps:
        rho_1 = canonical_iteration(rho_0)        
        error = dist(rho_1, rho_0)
        rho_0 = rho_1
        step += 1

        if step % 10 == 0:
            print(rho_0.data.size, ham.shape[0]**2)
            print(error, step)

    print(f"Finished after {step} / {max_steps} with error {error}")

    return rho_1

def get_density_matrix_foe():
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
    return

def get_rho_exact(ham):
    energies, vecs = np.linalg.eigh(ham.toarray())
    rho_exact_energy = np.diag(np.ones_like(energies) * (energies <= -5.0))
    rho_exact = vecs @ rho_exact_energy @ vecs.conj().T
    # rho_exact = vecs.conj().T @ rho_exact_energy @ vecs
    # rho_exact_energy.diagonal() # occupations
    # plt.plot(np.arange(ham.shape[0]), energies, 'o')
    # plt.show()
    return rho_exact

def plot_flake(flake):
    plt.scatter(x = flake.data[:, 0], y = flake.data[:, 1])
    plt.axis('equal')
    plt.show()
    plt.close()

t = time.time()
flake = get_flake(100)
print(time.time() - t)


t = time.time()
ham = get_hamiltonian(flake, gap = -10)
print(time.time() - t)
print(ham.shape)

cutoff_matrix = flake.sparse_distance_matrix(flake, max_distance = 4*1.43) != 0 + scp.sparse.identity(ham.shape[0]).astype(bool)

t = time.time()
rho = get_density_matrix_cp(ham, max_steps = 400)
print("Canonical Purification ", time.time() - t)

t = time.time()
energies, vecs = np.linalg.eigh(ham.toarray())
print("Exact diagonalization ", time.time() - t)



print("idempotency error", scp.sparse.linalg.norm(rho @ rho - rho))
print("occupation ", rho.trace(), "expected: ", rho.shape[0] // 2)
