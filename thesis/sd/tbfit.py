from typing import Callable, Union
import itertools as it
import dataclasses as dc

import jax
from jax import Array
import jax.numpy as jnp

import matplotlib.pyplot as plt

CouplingDict = dict[ str, list[complex] ]
Orbital = tuple[ Array, dict[str, Union[str, complex]] ]

# TODO port to struct
@dc.dataclass
class UnitCell():
    name          : str
    lattice_basis : Array    = dc.field( default_factory = lambda : jnp.array([]) )
    orbitals      : dict[str, Orbital] = dc.field( default_factory = lambda : {} )
    
# TODO: calculate this and other material parameters from hamiltonian according to RPA / lindhard theory
def susceptibility( hamiltonian ):
    pass

def show2D( unit_cell : UnitCell, n_rep : int = 4 ):
    """ Displays n_rep copies of the geometry contained in the unit_cell
    """
    points : list[Array] = []
    dim                       = unit_cell.lattice_basis.shape[0]
    candidates                = list( it.product( *[range(n_rep+1) for i in range(dim)] ) )
    
    for i, orbital in enumerate( unit_cell.orbitals ):
        points =jnp.array([unit_cell.orbitals[orbital][0] + sum(c[i]*unit_cell.lattice_basis[i] for i in range(dim)) for c in candidates])
        plt.scatter( points[:,0], points[:,1] )
    plt.axis('equal')
    plt.show()
    
# TODO: rethink this, takes sth like N^4 memory, most of which is wasted on what is essentially a sparse matrix
# furthermore, incredibly inefficient if called repeatedly, e.g. for optimization on geometry parameters
def matrix_params( unit_cell : UnitCell ) -> tuple[Array, Array, Array, Array]:
    """
    This function maps a unit cell containing N orbitals to an array representation via h, B, R such that the TB
    hamiltonian at k is given by $H_{ij}(\vec{k}) =  h_{m} B_{ijnm} exp(i R_{nl} k_l)$
    hoppings       : m-dim array, hopping rates (h)
    basis          : NxNxmxn-dim array, basis matrices (B)
    position_stack : n-dim array, unique distance vectors (R)
    """
    
    hilbert_dim    = len( unit_cell.orbitals )
    orbitals, lattice_basis, lattice_dim  = list(unit_cell.orbitals.items()), unit_cell.lattice_basis, unit_cell.lattice_basis.shape[0]
    
    # TODO: all of this should be changed
    position_stack, indices, hoppings, overlap =jnp.zeros((1,3)), [], [], []

    for i,orbital in enumerate(orbitals):
        pos = orbital[1][0]
        for j in range(i, hilbert_dim):
            partner       = orbitals[j]
            coupling      = orbital[1][1][partner[0]] if partner[0] in orbital[1][1] else partner[1][1][orbital[0]]
            ov            = orbital[1][2][partner[0]] if partner[0] in orbital[1][2] else partner[1][2][orbital[0]]
            n_max         = len(coupling)
            
            # all vectors connecting i and j
            distance_vecs =jnp.unique(
               jnp.round(
                    jnp.array(
                        [ partner[1][0] - pos + sum(c[i]*lattice_basis[i] for i in range(lattice_dim)) for c in it.product(*[range(-n_max, n_max+1) for i in range(lattice_dim)])]
                    ),
                    8),
                axis = 0 )

            # sorted, unique distances
            distances        =jnp.round(jnp.linalg.norm( distance_vecs, axis = 1 ), 8)
            distances_unique =jnp.unique( distances )

            # distance vectors relevant for nn... interaction
            distance_vecs = distance_vecs[distances < distances_unique[n_max],:]
            distances     = distances[distances < distances_unique[n_max]]

            # # keep only newly encountered distance vectors
            vecs = [vec for vec in distance_vecs if not (jnp.abs(position_stack - vec) < 1e-9).all(1).any() ]
            if vecs: position_stack =jnp.concatenate( (jnp.array(position_stack), jnp.array(vecs)), axis = 0)

            # indicate where basis tensor needs to be != 0
            indices += [ [i, j, jnp.where( (jnp.abs(position_stack - v) < 1e-9).all(1) )[0][0], len(hoppings) + (jnp.where(distances[k] == distances_unique)[0][0])] for k,v in enumerate(distance_vecs) ]
            hoppings     += coupling
            overlap      += ov

    basis =jnp.zeros( (hilbert_dim, hilbert_dim, position_stack.shape[0], len(hoppings)) )
    indices =jnp.array(indices).T
    basis = basis.at[ indices[0,:], indices[1,:], indices[2,:], indices[3,:]  ].set( 1. )            
    return jnp.array(hoppings), jnp.array(overlap), jnp.array(basis), jnp.array(position_stack)

# TODO: quick-and-dirty, because jax.linalg.eigh takes into account the FULL matrix for some reason 
@jax.jit
def hamiltonian_overlap( hoppings, overlap, basis, positions, k_vector  ):
    """ This function calculates the hamiltonian and overlap matrices for a set of k points
    Input:
    hoppings, overlap, basis, positions : Array, Array, Array, Array as returned by matrix_params
    k_vector : N-3 dim array of k-points where the hamiltonian and overlap should be computed
    Output:
    Hamiltonian : Array
    Overlap     : Array
    """
    def inner( arr ):
        mat = jnp.dot( jnp.dot(basis, arr), jnp.exp(1j * positions @ k_vector) )
        return mat + jnp.expand_dims(jnp.ones((mat.shape[0], mat.shape[0])) - jnp.eye(mat.shape[0]), 2) * jnp.transpose(mat.conj(), axes = (1,0,2))    
    return inner(hoppings), inner(overlap)
