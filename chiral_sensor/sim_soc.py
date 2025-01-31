"""common utilities"""

# Rashba SOC λR mixes
# states of opposite spins and sublattices. The unit vector dij
# points from site j to i and ŝ is the vector of spin Pauli
# matrices. The last term, the intrinsic SOC, is a next-nearest-
# neighbor hopping. It couples same spins and depends on
# clockwise (νij ¼ −1) or counterclockwise (νij ¼ 1) paths
# along a hexagonal ring from site j to i. This term dis-
# tinguishes intrinsic SOC at different sublattices λiI , where i
# stands for A or B.

# $H = H_{staggered, onsite} + H_{nn} + i R \sum_{<i, j>, s, s'} \left(d^{ij}_x \sigma_y - d^{ij}_y \sigma_x\right)_{ss'} c^{\dagger}_{i,s} c_{j, s'} + \sum_{<<i,j>>, s} I_{ij}[s]c^{\dagger}_{i,s} c_{j, s}$, where $I[s] = -I[s]$
# $\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$
# $\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$
# $iR \sum_{<i,j>, s, s'} \left(d^{ij}_x \sigma_y - d^{ij}_y \sigma_x\right)_{ss'} c^{\dagger}_{i,s} c_{j, s'} = i R \sum_{<i,j>, s \neq s'} \left(d^{ij}_x c^{\dagger}_{i, s} c_{j,s'} + id^{ij}_y c^{\dagger}_{i, s} c_{j,s'} + h.c. \right)$
# $\sum_{<<i,j>>, s} I_{ij}c^{\dagger}_{i,s} c_{j, s}$
# $(s \times d^{ij}) = d^{ij}_x \sigma_y - d^{ij}_y \sigma_x$

# main take away:
# two edge state types: one decays over much space, the other over short space
# both combine to make the system trivial
# idea: let long range states hybridize, forming bonding-antibonding pairs, removing them from the spectrum, leaving only short-range edge states

import matplotlib.pyplot as plt
import jax.numpy as jnp
from granad import *

# all atrocities
from sim import *

# nn hopping
t = 1.0
# onsite diff
onsite = 0.1 * t

# intrinsic soc, couples same spins, is >0/<0 for (+,+) / (-,-)
# uniform, i.e. no difference between sublattices => same spin
# staggered: lambda_a = - lambda_b => opposite spin
scale = 1
lambda_a = scale * 1j * 0.06 * t
lambda_b = -scale * 1j * 0.06 * t

# rashba soc, couples opposite spins,
scale_2 = 1
lambda_r = scale_2 * 1j * 0.05 * t

# needed for rashba coupling
nn_distance_vecs_normed = jnp.array([[-0.8660254,  0.5      ,  0.       ],
                                     [ 0.8660254,  0.5      ,  0.       ],
                                     [ 0.       , -1.       ,  0.       ]], dtype=float)

nn_distance_vecs =  jnp.array([[-1.23,     0.71014083,  0.        ],
                               [ 1.23,       0.71014083,  0.        ],
                               [ 0.,         -1.42028166,  0.        ]])


material = (
    Material("soc_graphene")
    .lattice_constant(2.46)
    .lattice_basis([
        [1, 0, 0],
        [-0.5, jnp.sqrt(3)/2, 0]
    ])
    .add_orbital_species("A_up", atom='C')
    .add_orbital_species("B_up", atom='C')
    .add_orbital_species("A_down", atom='C')
    .add_orbital_species("B_down", atom='C')
    .add_orbital(position=(0, 0), tag="A_+", species="A_up")
    .add_orbital(position=(0, 0), tag="A_-", species="A_down")
    .add_orbital(position=(-1/3, -2/3), tag="B_+", species="B_up")
    .add_orbital(position=(-1/3, -2/3), tag="B_-", species="B_down")    
    .add_interaction(
        "hamiltonian",
        participants=("A_up", "B_up"),
        parameters= [t],
    )
    .add_interaction(
        "hamiltonian",
        participants=("A_down", "B_down"),
        parameters= [t],
    )
    .add_interaction(
        "hamiltonian",
        participants=("A_up", "B_down"),
        # rashba soc, couples opposite spins like \lambda_R c^{\dagger} \begin{pmatrix} 0 & d^{ij}_x - i d^{ij}_y \\ d^{ij}_x + i d^{ij}_y \end{pmatrix} c => +- couples with d^{ij}_x + i d^{ij}_y, -+ couples with d^{ij}_x - i d^{ij}_y
        parameters= [
            nn_distance_vecs[0].tolist() + [lambda_r * (nn_distance_vecs_normed[0, 0].item() - 1j*nn_distance_vecs_normed[0, 1].item())],
            nn_distance_vecs[1].tolist() + [lambda_r * (nn_distance_vecs_normed[1, 0].item() - 1j*nn_distance_vecs_normed[1, 1].item())],
            nn_distance_vecs[2].tolist() + [lambda_r * (nn_distance_vecs_normed[2, 0].item() - 1j*nn_distance_vecs_normed[2, 1].item())],
        ],
    )
    .add_interaction(
        "hamiltonian",
        participants=("A_down", "B_up"),
        # rashba soc, couples opposite spins like \lambda_R c^{\dagger} \begin{pmatrix} 0 & d^{ij}_x - i d^{ij}_y \\ d^{ij}_x + i d^{ij}_y \end{pmatrix} c => +- couples with d^{ij}_x + i d^{ij}_y, -+ couples with d^{ij}_x - i d^{ij}_y
        parameters= [
            nn_distance_vecs[0].tolist() + [lambda_r * (nn_distance_vecs_normed[0, 0].item() + 1j*nn_distance_vecs_normed[0, 1].item())],
            nn_distance_vecs[1].tolist() + [lambda_r * (nn_distance_vecs_normed[1, 0].item() + 1j*nn_distance_vecs_normed[1, 1].item())],
            nn_distance_vecs[2].tolist() + [lambda_r * (nn_distance_vecs_normed[2, 0].item() + 1j*nn_distance_vecs_normed[2, 1].item())],
        ],
    )
    .add_interaction(
        "hamiltonian",
        participants=("A_down", "A_down"),
        # this is like [vec, hopping]
        parameters=[                
            [0, 0, 0, onsite], # onsite, A gets positive
            # hoppings positive if clockwise, spin-diagonal, different spins get different signs, down gets minus
            [-2.46, 0, 0, -jnp.conj(lambda_a)], 
            [2.46, 0, 0, -lambda_a],
            [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, -jnp.conj(lambda_a)],
            [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, -lambda_a],
            [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, -jnp.conj(lambda_a)],
            [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, -lambda_a]
        ],
    )
    .add_interaction(
        "hamiltonian",
        participants=("A_up", "A_up"),
        # this is like [vec, hopping]
        parameters=[                
            [0, 0, 0, onsite], # onsite, A gets positive
            # hoppings positive if clockwise, spin-diagonal, different spins get different signs, down gets minus
            [-2.46, 0, 0, jnp.conj(lambda_a)], 
            [2.46, 0, 0, lambda_a],
            [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(lambda_a)],
            [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, lambda_a],
            [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(lambda_a)],
            [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, lambda_a]
        ],
    )
    .add_interaction(
        "hamiltonian",
        participants=("B_down", "B_down"),
        # this is like [vec, hopping]
        parameters=[                
            [0, 0, 0, -onsite], # onsite, A gets positive
            # hoppings positive if clockwise, spin-diagonal, different spins get different signs, down gets minus
            [-2.46, 0, 0, -jnp.conj(lambda_b)], 
            [2.46, 0, 0, -lambda_b],
            [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, -jnp.conj(lambda_b)],
            [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, -lambda_b],
            [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, -jnp.conj(lambda_b)],
            [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, -lambda_b]
        ],
    )
    .add_interaction(
        "hamiltonian",
        participants=("B_up", "B_up"),
        parameters=[                
            [0, 0, 0, -onsite], # onsite, A gets positive
            # hoppings positive if clockwise, spin-diagonal, different spins get different signs, down gets minus
            [-2.46, 0, 0, jnp.conj(lambda_b)], 
            [2.46, 0, 0, lambda_b],
            [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(lambda_b)],
            [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, lambda_b],
            [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(lambda_b)],
            [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, lambda_b]
        ],
    )
)

# determining nn vectors
# flake = material.cut_flake(Triangle(20, armchair = False))
# flake.show_2d(show_index = True, show_tags = ['A_+', 'B_+'], name = 'fig.pdf')
# dists = (flake.positions[(55, 58, 54), :] - flake.positions[7]) / jnp.linalg.norm(flake.positions[(55, 58, 54), :] - flake.positions[7], axis = -1)

# checking rashba soc
# def get_info(flake, from_orb, to_orbs):
#     dists = (flake.positions[to_orbs, :] - flake.positions[from_orb])
#     print("Distance vectors", dists)
#     print("H from to:", flake.hamiltonian[from_orb, to_orbs])
#     print("H to from:", flake.hamiltonian[to_orbs, from_orb])

# flake = material.cut_flake(Triangle(20, armchair = False))
# flake.show_2d(show_index = True, show_tags = ['A_+', 'B_-'], name = 'fig_+-.pdf')
# flake.show_2d(show_index = True, show_tags = ['A_-', 'B_+'], name = 'fig_-+.pdf')

# # # coupling a- b+
# get_info(flake, 43, (64, 69, 63))
# am = flake.filter_orbs('A_-', Orbital)[0].group_id
# bp = flake.filter_orbs('B_+', Orbital)[0].group_id
# cf = flake.couplings.hamiltonian[(am, bp)]
# print(cf(nn_distance_vecs[0]))

# # coupling a+ b-
# get_info(flake, 18, (85, 84, 90))
# ap = flake.filter_orbs('A_+', Orbital)[0].group_id
# bm = flake.filter_orbs('B_-', Orbital)[0].group_id
# cf = flake.couplings.hamiltonian[(ap, bm)]
# print(cf(nn_distance_vecs[0]))

# for key, f in flake.couplings.hamiltonian.group_id_items():
#     print(key, f)

def get_edge_idxs(positions):
    delta = positions - positions[:, None, :]
    distances = jnp.round(jnp.linalg.norm(delta, axis = -1), 8)

    # nn distance is 2nd smallest distance, unique sorts and makes unique
    min_d = jnp.unique(distances)[1]

    return jnp.argwhere(jnp.sum(distances == min_d, axis = 1) < 3)
    

# full model
savedir = ""
def plot_spin_polarization(flake, eps):
    """plots spin polarization of all states captured in energy window +- eps around 0"""

    # collect all spin up / down
    up_idxs = jnp.array(flake.filter_orbs("A_+", int) + flake.filter_orbs("B_+", int))
    down_idxs = jnp.array(flake.filter_orbs("A_-", int) + flake.filter_orbs("B_-", int))

    # relevant state indices
    state_idxs = jnp.argwhere(jnp.logical_and(flake.energies > -eps, flake.energies < eps))
    print(state_idxs.shape)

    # only one orbital species for plotting the atoms
    plotting_list = OrbitalList(flake.filter_orbs("A_-", Orbital) + flake.filter_orbs("B_-", Orbital))

    # edges only
    edge_idxs = get_edge_idxs(plotting_list.positions)
    for i in state_idxs:
        diff = flake.eigenvectors[up_idxs, i] - flake.eigenvectors[down_idxs, i]
        display = jnp.zeros_like(diff)
        display = display.at[edge_idxs].set(diff[edge_idxs])
        plotting_list.show_2d(display = display, name = f'{savedir}{i}.pdf', indicate_atoms = False, mode = "two-signed", circle_scale = 250, show_index = False)
    
# # check whether different spins are at same position if on same sublattice in index order in orbital list
# x = flake.filter_orbs("A_-", Orbital)
# y = flake.filter_orbs("A_+", Orbital)
# diffs = [ x[i].position - y[i].position for i in range(len(y))]
# print(jnp.abs(jnp.array(diffs)).max())

sizes = [20, 40, 80, 100, 120]

for size in sizes:
    name = f"soc_{size}"

    # lookie lookie
    flake = material.cut_flake(Rectangle(20, size, armchair = False), plot = False)
    flake.set_electrons(len(flake) // 2)
    flake.set_open_shell()
    flake.show_energies(name = f"{savedir}energies_{size}.pdf")

    # plot occupations for spin up / spin down
    plot_spin_polarization(flake, 0.1)

    # lrt
    args_list = [(flake, "soc")]
    ip_response(args_list, name + ".npz")

    # figure chirality
    plot_chirality(f"cond_{name}" + ".npz")
