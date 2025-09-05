import numpy as np

import matplotlib.pyplot as plt

from granad import *
from granad._graphene_special import _cut_flake_graphene
from granad.materials import *
from granad import _watchdog


def display_lattice_cut(positions, selected_positions, polygon=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw polygon if provided
    if polygon is not None:
        patch = plt.Polygon(
            polygon[:-1], edgecolor="orange", facecolor="none", linewidth=2, linestyle='--'
        )
        ax.add_patch(patch)

    # Plot all lattice positions
    ax.scatter(
        positions[:, 0], positions[:, 1],
        s=20, color='lightgray', label='Lattice Points'
    )

    # Highlight selected positions
    ax.scatter(
        selected_positions[:, 0], selected_positions[:, 1],
        s=40, color='dodgerblue', edgecolor='black', label='Selected Points', zorder=3
    )

    # Styling
    # ax.set_xlim(-1, 1)
    # ax.set_ylim(-1, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize='small', frameon=False)
    ax.set_title("Lattice Cut Visualization", fontsize=12)

    plt.tight_layout()
    plt.savefig("graphene_cutting.pdf")    


def _finalize(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        for species in self.species.keys():
            self._species_to_groups[species] = _watchdog._Watchdog.next_value()
        return method(self, *args, **kwargs)
    return wrapper

@_finalize
def cut_flake_custom( material, polygon, plot=False, minimum_neighbor_number: int = 2):
    """
    Cuts a two-dimensional flake from the material defined within the bounds of a specified polygon.
    It further prunes the positions to ensure that each atom has at least the specified minimum number of neighbors.
    Optionally, the function can plot the initial and final positions of the atoms within the polygon.

    Parameters:
        material (Material): The material instance from which to cut the flake.
        polygon (Polygon): A polygon objects with a vertices property holding an array of coordinates defining the vertices of the polygon within which to cut the flake.
        plot (bool, optional): If True, plots the lattice and the positions of atoms before and after pruning.
                               Default is False.
        minimum_neighbor_number (int, optional): The minimum number of neighbors each atom must have to remain in the final positions.
                                                 Default is 2.

    Returns:
        list: A list of orbitals positioned within the specified polygon and satisfying the neighbor condition.

    Note:
        The function assumes the underlying lattice to be in the xy-plane.
    """
    def _prune_neighbors(
            positions, minimum_neighbor_number, remaining_old=jnp.inf
    ):
        """
        Recursively prunes positions to ensure each position has a sufficient number of neighboring positions
        based on a minimum distance calculated from the unique set of distances between positions.

        Parameters:
            positions (array-like): Array of positions to prune.
            minimum_neighbor_number (int): Minimum required number of neighbors for a position to be retained.
            remaining_old (int): The count of positions remaining from the previous iteration; used to detect convergence.

        Returns:
            array-like: Array of positions that meet the neighbor count criterion.
        """
        if minimum_neighbor_number <= 0:
            return positions
        distances = jnp.round(
            jnp.linalg.norm(positions[:, material.periodic] - positions[:, None, material.periodic], axis=-1), 4
        )
        minimum = jnp.unique(distances)[1]
        mask = (distances <= minimum).sum(axis=0) > minimum_neighbor_number
        remaining = mask.sum()
        if remaining_old == remaining:
            return positions[mask]
        else:
            return _prune_neighbors(
                positions[mask], minimum_neighbor_number, remaining
            )

    if material.name == 'graphene' and polygon.polygon_id in ["hexagon", "triangle"]:
        n, m, vertices, final_atom_positions, initial_atom_positions, sublattice = _cut_flake_graphene(polygon.polygon_id, polygon.edge_type, polygon.side_length, material.lattice_constant)
                
        raw_list, layer_index = [], 0
        for i, position in enumerate(final_atom_positions):
            orb = Orbital(
                position = position,
                layer_index = layer_index,
                tag="sublattice_1" if sublattice[i] == "A" else "sublattice_2",
                group_id = material._species_to_groups["pz"],                        
                spin=material.species["pz"][0],
                atom_name=material.species["pz"][1]
                    )
            layer_index += 1
            raw_list.append(orb)

        orbital_list = OrbitalList(raw_list)
        material._set_couplings(orbital_list.set_hamiltonian_groups, "hamiltonian")
        material._set_couplings(orbital_list.set_coulomb_groups, "coulomb")
        orb_list = orbital_list

    else:
        # to cover the plane, we solve the linear equation P = L C, where P are the polygon vertices, L is the lattice basis and C are the coefficients
        vertices = polygon.vertices
        L = material._lattice_basis[material.periodic,:2] * material.lattice_constant
        coeffs = jnp.linalg.inv(L.T) @ vertices.T * 1.1

        # we just take the largest extent of the shape
        u1, u2 = jnp.ceil( coeffs ).max( axis = 1)
        l1, l2 = jnp.floor( coeffs ).min( axis = 1)
        grid = material._get_grid( [ (int(l1), int(u1)), (int(l2), int(u2)) ]  )

        # get atom positions in the unit cell in fractional coordinates
        orbital_positions =  material._get_positions_in_uc()
        unit_cell_fractional_atom_positions = jnp.unique(
            jnp.round(orbital_positions, 6), axis=0
                )

        initial_atom_positions = material._get_positions_in_lattice(
            unit_cell_fractional_atom_positions, grid
        ) 

        polygon_path = Path(vertices)
        flags = polygon_path.contains_points(initial_atom_positions[:, :2])        
        pruned_atom_positions = initial_atom_positions[flags]

        # get atom positions where every atom has at least minimum_neighbor_number neighbors
        final_atom_positions = _prune_neighbors(
            pruned_atom_positions, minimum_neighbor_number
        )
        orb_list = material._get_orbital_list(final_atom_positions, grid)

    if plot == True:
        display_lattice_cut(
            initial_atom_positions, final_atom_positions, vertices
        )
    return orb_list


triangle = Triangle(15)
flake = cut_flake_custom(get_graphene(), triangle, plot = True)
