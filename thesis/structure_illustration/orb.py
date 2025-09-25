# Refine the lobe shape to look more "dumbbell-like" (fatter lobes, narrow node)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Create spherical grid
phi = np.linspace(0, 2*np.pi, 300)
theta = np.linspace(0, np.pi, 300)
phi, theta = np.meshgrid(phi, theta)

# Stylized radius: use a gentle exponent to fatten lobes
R = 1.0
r = R * (np.abs(np.cos(theta))**0.6)

# Convert to Cartesian
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)  # sign handled naturally

# Plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    x, y, z,
    rstride=2, cstride=2,
    linewidth=0.25,
    antialiased=True,
    shade=True,
    alpha=0.9
)

# Clean up the axes
ax.set_box_aspect([1, 1, 1])
ax.set_axis_off()

lims = 1.1 * R
ax.set_xlim(-lims, lims)
ax.set_ylim(-lims, lims)
ax.set_zlim(-lims, lims)

ax.view_init(elev=22, azim=35)
plt.tight_layout()
plt.show()

# Save updated image
out_path2 = "pz_orbital_v2.png"
fig.savefig(out_path2, dpi=200, bbox_inches="tight", transparent=True)

from granad import *

flake = get_graphene().cut_flake(Triangle(10), plot = True)
