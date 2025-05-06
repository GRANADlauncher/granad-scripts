from granad import *
import jax.numpy as jnp

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
    right =  flake.positions[:, 1].min()    
    left = jnp.array([left, 0, 0]) + jnp.array(ni_pos[0])
    right =  jnp.array([right, 0, 0])  + jnp.array(ni_pos[1])
    
    ni1.set_position(left)    
    ni2.set_position(right)

    return flake + ni1 + ni2

if __name__ == '__main__':
    # ni as 2 orb model, we can have more
    ni_orbs =  ["s", "p"]
    ni1 = get_ni(ni_orbs, 1)
    ni2 = get_ni(ni_orbs, 1)

    # carbon flake and ni atoms
    flake = get_system([ni1, ni2], [[1,0,0], [-1, 0, 0]])

    # coupling
    flake.set_hamiltonian_groups("s", flake[0], potential_s)
    flake.set_hamiltonian_groups("p", flake[0], potential_p)    

    flake.show_2d()
