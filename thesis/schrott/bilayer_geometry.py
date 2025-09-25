import matplotlib.pyplot as plt

from granad import *

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

def show_3d(orbs, show_tags=None, cmap=None, circle_scale: float = 1e3, title=None, name = None):
    # Determine which tags to display
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    # Group orbital positions by tag
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for idx, orb in enumerate(orbs):
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(idx)

    # Set up color map
    unique_tags = sorted(tags_to_pos.keys())
    if cmap is None:
        base_cmap = plt.get_cmap('tab10')
    else:
        base_cmap = plt.get_cmap(cmap)

    color_map = {
        tag: base_cmap(i / max(len(unique_tags) - 1, 1))
        for i, tag in enumerate(unique_tags)
    }

    # Prepare the 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    for tag, positions in tags_to_pos.items():
        positions = np.array(positions)
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            s=circle_scale * 0.1,
            color=color_map[tag],
            edgecolors='k',
            alpha=0.75,
            label=str(tag)
        )

    # Plot appearance settings
    ax.grid(False)
    ax.axis("off")

    plt.tight_layout()

    if name is None:
        name = "graphene_bilayer.png"
    plt.savefig(name, transparent = True)
        

triangle = Triangle(15)
flake = MaterialCatalog.get("graphene").cut_flake( triangle )
show_3d(flake, name = "single.png")


flake = MaterialCatalog.get("graphene").cut_flake( triangle )
flake.rotate(flake.positions[flake.center_index], 0.2)
flake.shift_by_vector( [0,0,1]  )
print(flake.positions)
second_flake = MaterialCatalog.get("graphene").cut_flake( triangle )
stack = flake + second_flake
show_3d(stack, name = "double.png")
