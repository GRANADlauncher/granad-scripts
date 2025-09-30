import matplotlib.pyplot as plt

from granad import *

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


def show_2d(
        flake,
        circle_scale = 1e3 * 0.5,
        color_map = {}
):
    """
    Same as your original show_2d, but will draw onto a provided Axes (ax) if given.
    This makes it composable for multi-panel layouts.
    """
    
    # Prepare data structures for plotting
    show_tags = {orb.tag for orb in flake}
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for orb in flake:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(flake.index(orb))

    fig, ax = plt.subplots()

    # Color by tags if no display is given
    for tag, positions in tags_to_pos.items():
        positions = jnp.array(positions)
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            label=tag,
            s = circle_scale,
            color=color_map[tag],
            edgecolor='white',
            alpha=0.7,
        )


    ax.grid(False)
    ax.axis('equal')
    ax.axis("off")


    # Return fig/ax for composition
    return fig, ax

def show_3d(orbs, circle_scale: float = 1e2):

    # Group orbital positions by tag
    show_tags = {orb.tag for orb in orbs}
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for idx, orb in enumerate(orbs):
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(idx)

    # Set up color map
    unique_tags = sorted(tags_to_pos.keys())
    base_cmap = plt.get_cmap('tab10')

    color_map = {
        tag: base_cmap(i / max(len(unique_tags), 1))
        for i, tag in enumerate(unique_tags)
    }

    # Prepare the 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for tag, positions in tags_to_pos.items():
        positions = np.array(positions)
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            s=circle_scale,
            color=color_map[tag],
            edgecolors='k',
            alpha=0.75,
            label=str(tag)
        )

    # Plot appearance settings
    ax.grid(False)
    ax.axis("off")
    fig.tight_layout()

    return fig, ax, color_map
        

color_map = {'B': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 'N': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0), 'orb': (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0), 'sublattice_1': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0), 'sublattice_2': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0)}

# hbn
flake = get_hbn().cut_flake(Rectangle(10, 10))
fig, ax = show_2d(flake, color_map = color_map)
fig.savefig("hbn.png", transparent = True)
flake.rotate(flake.positions[flake.center_index], 0.2)
fig, ax = show_2d(flake, color_map = color_map)
fig.savefig("hbn_rotated.png", transparent = True)

# graphene
flake2 = get_graphene().cut_flake(Triangle(10))
fig, ax = show_2d(flake2, color_map = color_map)
fig.savefig("graphene.png", transparent = True)
del flake2[flake2.center_index]
pos = -2*flake.positions[flake.center_index]
shift = pos + jnp.array([0,0,3])
flake2.shift_by_vector(shift)
fig, ax = show_2d(flake2, color_map = color_map)
fig.savefig("graphene_defective.png", transparent = True)

flake3 = flake + flake2

single_orb = OrbitalList([Orbital(position = flake2.positions[flake2.center_index] + jnp.array([0, 0, 6]), tag = "orb")])
fig, ax = show_2d(single_orb, color_map = color_map)
fig.savefig("orbital.png", transparent = True)
        
flake3 += single_orb
fig, ax, color_map = show_3d(flake3)
fig.savefig("stack.png", transparent = True)
# fig.show()
plt.close()




# triangle = Triangle(15)
# flake = MaterialCatalog.get("graphene").cut_flake( triangle )
# show_3d(flake, name = "single.png")


# flake = MaterialCatalog.get("graphene").cut_flake( triangle )
# flake.rotate(flake.positions[flake.center_index], 0.2)
# flake.shift_by_vector( [0,0,1]  )
# print(flake.positions)
# second_flake = MaterialCatalog.get("graphene").cut_flake( triangle )
# stack = flake + second_flake
# show_3d(stack, name = "double.png")
