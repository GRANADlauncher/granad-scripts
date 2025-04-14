import matplotlib.pyplot as plt

from granad import *

def get_bilayer_graphene(shape, interlayer_scaling = 1.0, d0 = 0.335 * 10, doping = 0, eps_upper = 1.0, eps_lower = 1.0):
    """AB stack of two graphene layers of shapes around the common medoid in the xy-plane.
    
    Parameters:
        shape : granad shape array, assumed identical for both flakes
        d0 : interlayer spacing

    Returns:
        flake
    """    

    def tb_coupling(r):
        d = jnp.linalg.norm(r)
        vpp = vpp0 * jnp.exp(-(d-a0)/delta0)
        vps = vps0 * jnp.exp(-(d-d0)/delta0)
        
        # this is the z-dependent term, so we can scale it to modulate the interlayer interaction
        rd2 = (r[2]/d)**2 * interlayer_scaling
        
        return vpp * (1 - jnp.nan_to_num(rd2, nan = 1)) + jnp.nan_to_num(vps * rd2, nan = 0) + 0j

    def coulomb_coupling(eps_b):
        def inner(r):
            return u0/(eps_b * (jnp.linalg.norm(r) + 1)) + 0j
        return inner

    vpp0, vps0, a0 = -2.7, 0.48, 1.42
    delta0 = 0.184 * a0 * jnp.sqrt(3)
    u0, eps_b = 17.38, 1.0

    # uncoupled layers
    upper, lower = MaterialCatalog.get("graphene").cut_flake(shape), MaterialCatalog.get("graphene").cut_flake(shape)

    # rotate upper layer around medoid
    a = 2.46
    shift = jnp.array([a / 3, a / (3 * jnp.sqrt(3)), d0])
    upper.shift_by_vector(shift)

    # combine layers
    flake = upper + lower

    # doping
    flake.set_electrons(flake.electrons + doping)

    # set approximate tb couplings
    flake.set_hamiltonian_groups(upper, upper, tb_coupling)
    flake.set_hamiltonian_groups(upper, lower, tb_coupling)
    flake.set_hamiltonian_groups(lower, lower, tb_coupling)

    flake.set_coulomb_groups(upper, upper, coulomb_coupling(eps_upper))
    flake.set_coulomb_groups(upper, lower, coulomb_coupling(eps_lower/2 + eps_upper/2))
    flake.set_coulomb_groups(lower, lower, coulomb_coupling(eps_lower))
            
    return flake

def hbn():    
    # get material
    graphene = MaterialCatalog.get( "graphene" )
    
    # cut a 15 Angström wide triangle from the lattice (can also be an arbitrary polygon)
    flake = graphene.cut_flake( Rectangle(15, 15) )
    
    # frequencies
    omegas_rpa = jnp.linspace( 0, 6, 40 )
    
    polarizability = flake.get_polarizability_rpa(
        omegas_rpa,
        relaxation_rate = 1/10,
        polarization = 0, 
        hungry = 2 )

    absorption_rpa = jnp.abs( polarizability.imag * 4 * jnp.pi * omegas_rpa )

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(omegas_rpa, absorption_rpa / jnp.max(absorption_rpa), linewidth=2)
    plt.xlabel(r'$\hbar\omega$', fontsize=20)
    plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
    plt.grid(True)
    plt.savefig("hbn_absorption.pdf")

def bilayer():
    # cut a 15 Angström wide triangle from the lattice (can also be an arbitrary polygon)
    flake = get_bilayer_graphene(Triangle(10))
    
    # frequencies
    omegas_rpa = jnp.linspace( 0, 6, 40 )
    
    polarizability = flake.get_polarizability_rpa(
        omegas_rpa,
        relaxation_rate = 1/10,
        polarization = 0, 
        hungry = 2 )

    absorption_rpa = jnp.abs( polarizability.imag * 4 * jnp.pi * omegas_rpa )

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(omegas_rpa, absorption_rpa / jnp.max(absorption_rpa), linewidth=2)
    plt.xlabel(r'$\hbar\omega$', fontsize=20)
    plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
    plt.grid(True)
    plt.savefig("bilayer_absorption.pdf")

def mos2():
    mos2 = get_mos2()
    flake = mos2.cut_flake(Rectangle(20, 20))
    flake.show_3d(name = "mos2_geometry.pdf")
    flake.show_energies(e_max = 5, e_min = -5, name = "mos2_energies.pdf")

if __name__ == '__main__':
    hbn()
    bilayer()
    mos2()
