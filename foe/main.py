# TODO: this: https://github.com/flatironinstitute/sparse_dot/tree/release
# TODO: make (grand) canonical purifcation fast by keeping matrices sparse
# TODO: remove hardcoding for flexible bounds
# TODO: time propagation by hamiltonian: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.expm.html
# TODO: check numpy version shennanigans
# TODO: more geometries
# TODO: implement chebyshev density matrix and chebyshev expansions
# TODO: how bout non-diff linear granad?, need sugar-y abstractions for materials, but this should suffice for now
import time
from typing import Callable, Iterable, List, Tuple, Any

import numpy as np
import jax.numpy as jnp
import scipy as scp

import matplotlib.pyplot as plt

from granad import *

def get_fourier_transform(t_linspace, function_of_time, omega_max = np.inf, omega_min = -np.inf, return_omega_axis=True):
    # Calculate the frequency axis
    delta_t = t_linspace[1] - t_linspace[0]  # assuming uniform spacing

    # Compute the FFT along the first axis    
    function_of_omega = np.fft.fft(function_of_time, axis=0) * delta_t

    N = function_of_time.shape[0]  # number of points in t_linspace
    omega_axis = 2 * np.pi * np.fft.fftfreq(N, d=delta_t)

    mask = (omega_axis >= omega_min) & (omega_axis <= omega_max)
    
    if return_omega_axis:
        return omega_axis[mask], function_of_omega[mask]
    else:
        return function_of_omega[mask]


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

def canonical_iteration(rho_0, mask):
    r2 = rho_0 @ rho_0
    r3 = r2 @ rho_0

    c = (r2 - r3).trace() / (rho_0 - r2).trace()

    if c >= 0.5:
        rho_1 = ((1 + c) * r2 - r3) / c
    else:
        rho_1 = ((1 - 2*c) * rho_0 + (1 + c)* r2 - r3) / (1 - c)

    return rho_1.multiply(mask)

# following PhysRevB.58.12704.pdf
def spectral_bounds(H, tighten=20):
    """Cheap lower/upper bounds needed for scaling."""
    d = H.diagonal()
    R = np.abs(H).sum(axis=1).A1 - np.abs(d)
    Emin, Emax = (d - R).min(), (d + R).max()
    # tighten with a few Lanczos steps
    try:
        from scipy.sparse.linalg import eigsh
        Emax = max(Emax, eigsh(H, k=1, which='LA', return_eigenvectors=False)[0])
        Emin = min(Emin, eigsh(H, k=1, which='SA', return_eigenvectors=False)[0])
    except Exception:
        pass
    return Emin, Emax

def prune(M, mask):
    M = M.multiply(mask)
    M.eliminate_zeros()
    return M

def get_density_matrix_cp(H, mask, cutoff=1e-6, max_steps=200):
    N  = H.shape[0]
    Ne = N // 2                       # half filling for graphene
    Id = scp.sparse.identity(N, format='csr')

    Emin, Emax = spectral_bounds(H)
    alpha      = 1.0/(Emax - Emin)
    rho        = (Emax*Id - H) * alpha
    rho *= Ne / rho.trace()           # enforce Tr ρ = Ne

    mask = ((H != 0) + Id).astype(bool) # locality mask
    rho  = prune(rho, mask)

    err, step = 1.0, 0
    while err > cutoff and step < max_steps:
        # Palser–Manolopoulos canonical iteration
        r2, r3 = rho @ rho, None
        r3     = r2 @ rho
        c      = (r2.trace() - r3.trace()) / (rho.trace() - r2.trace())
        rho    = ((1 + c)*r2 - r3)/c if c >= 0.5 \
                 else ((1 - 2*c)*rho + (1 + c)*r2 - r3)/(1 - c)
        rho    = prune(rho, mask)

        step += 1
        if step % 5 == 0 or err < cutoff:
            # err = abs(rho.trace() - Ne)/Ne
            err = scp.sparse.linalg.norm(rho@rho - rho)
            # print((rho @ H).trace() / -2.7 / (H.shape[0]//2))
            print(f"step {step:3d}: ΔTrρ = {err:.2e}")
            
    err = scp.sparse.linalg.norm(rho@rho - rho)
    print(f"Converged in {step} steps, idempotency error ", f"{err}")
    return rho

def get_linear_response():
    return

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

def get_pulse(
    amplitudes: list[float],
    frequency: float,
    peak: float,
    fwhm: float,
):
    def _field(t, real = True):
        val = (
            static_part
            * np.exp(-1j * np.pi / 2 + 1j * frequency * (t - peak))
            * np.exp(-((t - peak) ** 2) / sigma**2)
        )
        return val.real if real else val
    static_part = np.array(amplitudes)
    sigma = fwhm / (2.0 * np.sqrt(np.log(2)))
    return _field
    

def get_rhs(ham, dip, rho_stat, field):
    def sparse_rhs(t, rho):
        # perturbation out of equilibrium
        delta_rho = rho - rho_stat

        f = field(t)
        field_term = scp.sparse.spdiags([dip @ f], diags = 0)

        # hermitian
        h_total = ham + field_term
        h_times_d = h_total @ rho
        comm = -1j * (h_times_d  - h_times_d.conj().T)

        # loss
        diss = -delta_rho * 1/2 * 0
        
        return comm + diss
    return sparse_rhs

def rk4_propagate(
        rhs_func,                
        t_grid,
        rho0,
        dip,
        mask
):
    """
    Sparse RK4 propagator for a density matrix.

    Parameters
    ----------
    rhs_func : callable
        drho/dt = rhs_func(t, rho, args). Must return a **sparse** matrix.
    t_grid : 1-D array, float
        Monotonic time points (len ≥ 2).
    rho0 : sparse matrix
        Initial density matrix from canonical purification.
        Will be converted to complex & CSR on entry.
    args : tuple
        Extra arguments forwarded to `rhs_func`.
    postprocesses : iterable of callables
        Each f(t, rho, args) → value.  Return values are collected
        at every step and returned in a list-of-lists.
    mask : float
        Absolute tolerance below which matrix entries are discarded
        to keep ρ sparse.

    Returns
    -------
    rho_T : sparse matrix
        Density matrix at t_grid[-1].
    measurements : list
        List of lists with post-processing outputs per time-step.
    """    
    # --- initialisation -----------------------------------------------------
    rho = rho0.astype(np.complex128, copy=True).tocsr()
    rho_stat = rho
    out: List[List[Any]] = []

    # --- main RK4 loop ------------------------------------------------------
    for i in range(len(t_grid) - 1):        
        t  = float(t_grid[i])
        dt = float(t_grid[i + 1] - t)

        # RK4 stages (all sparse)
        k1 = rhs_func(t,           rho)
        # k2 = rhs_func(t + dt / 2,  rho + (dt / 2) * k1)
        # k3 = rhs_func(t + dt / 2,  rho + (dt / 2) * k2)
        # k4 = rhs_func(t + dt,      rho + dt * k3)
        # rho = prune(rho + dt / 6 * (k1 + 2*k2 + 2*k3 + k4), mask)

        rho = prune(rho + dt * k1, mask)

        # collect x dipole moment
        delta_rho = rho - rho_stat
        out.append((delta_rho.diagonal() * dip[:, 0]).sum())

        if i % 100 == 0:
            print(f"step {i}")

    return rho, out

def sim():
    ## static
    t = time.time()
    flake = get_flake(200)
    print(time.time() - t)
    t = time.time()
    ham = get_hamiltonian(flake, gap = -10)
    print(time.time() - t)
    print(ham.shape)
    mask = (flake.sparse_distance_matrix(flake, max_distance = 20*1.43) != 0 + scp.sparse.identity(ham.shape[0])).astype(bool)
    t = time.time()
    rho = get_density_matrix_cp(ham, mask, cutoff = 1e-6, max_steps = 400)
    print("Canonical Purification ", time.time() - t)
    r = get_rho_exact(ham)
    print(np.linalg.norm(r-rho))

    ## dynamic
    t_points    = np.linspace(0.0, 30.0, 10001)           # 0 … 50 fs, 0.05 fs step
    pulse = get_pulse(amplitudes=[1e-5, 0], frequency=2.3, peak=2, fwhm=0.5)
    dip = flake.data
    rhs_func = get_rhs(ham, dip, rho, pulse)

    rho_final, dip = rk4_propagate(
        rhs_func,
        t_points,
        rho,
        dip,
        mask
    )

    scp.sparse.save_npz("rho.npz",        rho)          # ρ(t = 0)
    scp.sparse.save_npz("rho_final.npz",  rho_final)    # ρ(t = t_final)

    np.savez_compressed(
        "time_and_dip.npz",
        t_points=t_points,            # shape (N_time,)
        dip=dip                       # shape (N_time,) or (N_time, …)
    )

def plot_sim():

    rho        = scp.sparse.load_npz("rho.npz")
    rho_final  = scp.sparse.load_npz("rho_final.npz")

    data       = np.load("time_and_dip.npz")
    t_points   = data["t_points"]
    dip        = data["dip"]
    
    n = 6000
    dip = np.array(dip)
    plt.plot(t_points[:n], dip[:n])
    plt.savefig("dip.pdf")
    plt.close()

    eta = 0.5
    dip_damped = dip * np.exp(-eta * t_points)
    plt.plot(t_points[:n], dip_damped[:n])
    plt.savefig("dip_damped.pdf")
    plt.close()

    omega, dip_omega = get_fourier_transform(t_points, dip_damped, omega_min = 0, omega_max = 10)
    plt.plot(omega, dip_omega)
    plt.savefig("dip_omega.pdf")
    plt.close()


if __name__ == '__main__':
    sim()
    plot_sim()
