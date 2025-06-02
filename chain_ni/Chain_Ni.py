# MAIN TAKEAWAYS:
# 1. the PRB coupling `potential_s` induces a strong energy shift between the eigenstate most localized on Ni1 and the eigenstate most localized on Ni2
# 2. large dissipation rates kill the Rabi oscillations
# 3. with large fields and low dissipation rates, Rabi oscillations set in

from granad import *
import jax.numpy as jnp
import matplotlib.pyplot as plt

def potential_s(r):
    # cf. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.79.014109
    # sigma = -2.319
    # pi = 1.306
    # r0 = 1.88 A
    # q = 3.2
    # cutoff at 3.20 A

    q = 3.2
    c = -2.319*0.3
    r0 = 1.88
    
    r = jnp.linalg.norm(r)
    
    return c * jnp.exp(-q * (r / r0 - 1) )


def potential_p(r):
    # cf. https://journals.aps.org/prb/abstract/10.1103/PhysRevB.79.014109
    # sigma = -2.319
    # pi = 1.306
    # r0 = 1.88 A
    # q = 3.2
    # cutoff at 3.20 A

    q = 3.2
    c = 1.306*0.2
    r0 = 1.88
    
    r = jnp.linalg.norm(r)
    
    return c * jnp.exp(-q * (r / r0 - 1) )

def get_ni(orbitals, electrons):    
    adatom = OrbitalList([Orbital(tag = t) for t in orbitals])    
    adatom.set_electrons(electrons)
    return adatom

def get_system(ni, ni_pos, size = 20):
    flake = MaterialCatalog.get("graphene").cut_flake(Rectangle(size, 5))

    left = flake.positions[:, 0].max()
    right =  flake.positions[:, 0].min()    
    left = jnp.array([left, 0, 0]) + jnp.array(ni_pos[0])
    right =  jnp.array([right, 0, 0])  + jnp.array(ni_pos[1])
    
    ni1.set_position(left)    
    ni2.set_position(right)

    return flake + ni1 + ni2


if __name__ == '__main__':
    # ni as 2 orb model, we can have more
    ni_orbs =  ["s"]
    ni1 = get_ni(ni_orbs, 1)
    ni2 = get_ni(ni_orbs, 1)

    # carbon flake and ni atoms
    flake = get_system([ni1, ni2], [[1 * jnp.cos(jnp.pi / 6),jnp.sin(jnp.pi / 6),0], [-1.5 * jnp.cos(jnp.pi / 6), 1.5*jnp.sin(jnp.pi / 6), 0]], size = 10)
    #flake = get_system([ni1, ni2], [[1 * jnp.cos(jnp.pi / 6), 1 * jnp.sin(jnp.pi / 6),0], [-1.5*jnp.cos(jnp.pi / 6), 1.5*jnp.sin(jnp.pi / 6),0]], size = 7.4)
    #flake.show_2d()
    
    
    # coupling
    flake.set_hamiltonian_groups("s", flake[0], potential_p)

    flake.set_hamiltonian_element(ni1, ni1, 0 + 0j)
    flake.set_hamiltonian_element(ni2, ni2, 0 + 0j)

    flake.show_2d(name = "7-acene.pdf")
    
    # eigenstate index most localized on Ni1
    level1 = int(jnp.argmax(jnp.abs(flake.eigenvectors[-2])))
    
    # eigenstate index most localized on Ni2
    level2 = int(jnp.argmax(jnp.abs(flake.eigenvectors[-1])))
    print(flake.energies[level1], flake.energies[level2])
    flake.set_excitation(level2, level1, 2)
    
    ## Uncomment to visualize the energy 
    #flake.show_energies()

    ## uncomment to visualize the dipole operator
    # plt.matshow(jnp.abs(flake.dipole_operator_e).sum(axis = 0)) # sum modulus of all cartesian components
    # plt.colorbar()
    # plt.show()

    print("Transition dipole moment (modulus) between states most localized on Ni1, Ni2 ", jnp.abs(flake.dipole_operator_e).sum(axis = 0)[level1, level2])
    print("Average transition dipole moment (modulus) ", jnp.mean(jnp.abs(flake.dipole_operator_e)))    

    # # uncomment to visualize localized states
    # flake.show_2d(display = flake.eigenvectors[:, level1])
    # flake.show_2d(display = flake.eigenvectors[:, level2])

    # cf. https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.109.267209, referenced by referee prb
    freq = jnp.abs(flake.energies[level2] - flake.energies[level1])
    print("Driving frequency = Transition frequency between localized states ", freq)
    
    pulse = Pulse(
        amplitudes=[0.1, 0, 0], frequency=freq, fwhm = 85, peak = 50,
    )
    
    result = flake.master_equation(
        grid = 10,
        relaxation_rate = 1/1000,
        illumination = pulse,
        density_matrix = ["occ_e","occ_x"], 
        end_time = 250,
        coulomb_strength = 1,
    )    
    occ_e = result.output[0]
    occ_x = result.output[1]
   # n_steady=0
    #n_steady = jnp.argmin((occ_x[n_steady:, -2] - occ_x[n_steady:, -1]).real)
    n_steady=-1
    # real space occupation plot on Ni1, Ni2
    plt.plot(result.time_axis, occ_x[:, -2], '-.')
    plt.plot(result.time_axis, occ_x[:, -1])
  
    # energy space plot of level 1 and level 2
    #plt.plot(result.time_axis[n_steady:], occ_e[n_steady:, level1], '-.')
    #plt.plot(result.time_axis[n_steady:], occ_e[n_steady:, level2])
    
    plt.show()

#====================Final Plot=================================#
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def get_nearby_connections(points: np.ndarray, k: int = 3, tolerance: float = 0.3) -> np.ndarray:
    """
    Returns an array of connections [point1, point2] where the distance is within the given tolerance.
    """
    if points.shape[1] > 2:
        points = points[:, :2]  # Use only X, Y

    tree = cKDTree(points)
    distances, indices = tree.query(points, k=k+1)  # +1 to include self

    connections_set = set()

    for i in range(len(points)):
        for j in range(1, k+1):  # Skip self (j=0)
            neighbor_index = indices[i, j]
            dist = distances[i, j]

            if dist <= tolerance:
                # Ensure consistent ordering to prevent duplicates
                pair = tuple(sorted((i, neighbor_index)))
                connections_set.add(pair)

    # Convert index pairs to coordinate line segments
    connections = np.array([[points[i], points[j]] for i, j in connections_set])
    return connections

colors = ['red', 'green']
fig,ax=plt.subplots(figsize=(7, 5),dpi=150)  
for idx in [-1,-2]:
    ax.plot(result.time_axis[:n_steady], occ_x[:n_steady, idx], color=colors[idx], label=f'Ni_{jnp.int32(jnp.abs(idx))}')
ax.set_ylabel('site occupation')
ax.set_xlabel('time $[\hbar /eV]$')
ax.grid(False)
ax.set_ylim([0,2])
ax.legend(loc='lower right')

axins = inset_axes(ax, width="50%", height="15%", loc='center right', borderpad=0.5)
axins.scatter(flake.positions[:-2,0],flake.positions[:-2,1],s=5,color="black")
axins.scatter(flake.positions[-1,0],flake.positions[-1,1],s=15,color=colors[-1])
axins.scatter(flake.positions[-2,0],flake.positions[-2,1],s=15,color=colors[-2])
axins.annotate("$Ni_1$",(-12,-1),color=colors[-1])
axins.annotate("$Ni_2$",(10.5,-1), color=colors[-2])
axins.set_xlabel('x' , labelpad=1)
axins.set_ylabel('y', labelpad=1)
axins.set_xticks([])
axins.set_yticks([])
axins.set_xlim(jnp.min(flake.positions[:,0])-1, jnp.max(flake.positions[:,0])+2)
axins.set_ylim(jnp.min(flake.positions[:,1])-1, jnp.max(flake.positions[:,1])+1)
connections=get_nearby_connections(flake.positions, tolerance= 1.5)
for line in connections:
    axins.plot(line[:,0],line[:,1],color="grey",zorder=-10, linewidth=0.8)

plt.show()