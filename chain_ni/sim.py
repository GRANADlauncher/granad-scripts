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
    c = -2.319
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
    c = 1.306
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
    flake = get_system([ni1, ni2], [[1,0,0], [-1.5, 0, 0]], size = 10)
    flake.show_2d()
    
    # coupling
    flake.set_hamiltonian_groups("s", flake[0], potential_s)

    flake.set_hamiltonian_element(ni1, ni1, 0j)
    flake.set_hamiltonian_element(ni2, ni2, 1 + 0j)

    flake.show_2d(name = "7-acene.pdf")
    
    # eigenstate index most localized on Ni1
    level1 = int(jnp.argmax(jnp.abs(flake.eigenvectors[-2])))
    
    # eigenstate index most localized on Ni2
    level2 = int(jnp.argmax(jnp.abs(flake.eigenvectors[-1])))

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
    freq = flake.energies[level1] - flake.energies[level2]
    print("Driving frequency = Transition frequency between localized states ", freq)
    
    pulse = Wave(
        amplitudes=[1, 0, 0], frequency=freq,
    )
    
    result = flake.master_equation(
        grid = 10,
        relaxation_rate = 1e-4,
        illumination = pulse,
        density_matrix = ["occ_x"], 
        end_time = 80,
        coulomb_strength = 0,
    )    
    occ_x = result.output[0]
    
    n_steady = 0
    
    # real space occupation plot on Ni1, Ni2
    plt.plot(result.time_axis[n_steady:], occ_x[n_steady:, -2], '-.')
    plt.plot(result.time_axis[n_steady:], occ_x[n_steady:, -1])

    # energy space plot of level 1 and level 2
    # plt.plot(result.time_axis[n_steady:], occ_x[n_steady:, level1], '-.')
    # plt.plot(result.time_axis[n_steady:], occ_x[n_steady:, level2])
    
    plt.show()
