import matplotlib.pyplot as plt

import jax.numpy as jnp

from granad import *

# commensurate twist angles $\cos(\theta_i) = \frac{3i^2 +3i + 1/2}{3i^2 + 3i + 1}$
# $L_i = a \sqrt{3i^2 + 3i + 1}$
# $L_i \leq 2R$

def get_size(shape):
    """spatial extent of a structure defined as maximum distance from center to edge"""
    flake = MaterialCatalog.get("graphene").cut_flake(shape)
    pos = flake.positions
    center = pos[flake.center_index]
    return jnp.max(jnp.linalg.norm(pos - center, axis = 1))

def twist_angles(shape):
    """array of twist angles fullfilling $L(\\theta_i) \\leq 2R$"""
    size = get_size(shape)
    indices = jnp.arange(100)    
    angles = jnp.arccos(
        (3*indices**2 + 3*indices + 1/2) / (3*indices**2 + 3*indices + 1)
    ) 
    lengths = 2.46 * jnp.sqrt(3*indices**2 + 3*indices + 1)
    return jnp.r_[0, angles[lengths <= 2*size]]

def get_bilayer_graphene(shape, phi, interlayer_scaling = 1.0, d0 = 0.335 * 10, doping = 0, eps_upper = 1.0, eps_lower = 1.0):
    """rotated AA stack of two graphene layers of shapes around the common medoid in the xy-plane.
    
    Parameters:
        shape : granad shape array, assumed identical for both flakes
        phi : rotation angle
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
    upper.rotate(upper.positions[upper.center_index], phi)
    upper.shift_by_vector([0, 0, d0])

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


def rpa_sim(shape, phi, doping, omega, suffix = '', eps_upper = 1.0, eps_lower = 1.0):
    def pol(flake):            
        return flake.get_polarizability_rpa(
            omega,
            relaxation_rate = 1/10,
            polarization = 0, 
            hungry = 1)

    flake_free = MaterialCatalog.get("graphene").cut_flake(shape)

    # doping
    # flake_free.set_electrons(flake_free.electrons + 2)    
    # pols = [pol(flake_free)]

    names = []
    
    for p in phi:
        pols = []
        flake = get_bilayer_graphene(shape, p, eps_upper = eps_upper, eps_lower = eps_lower)
        el = flake.electrons
        flake.show_2d(name = f'{p}.pdf')
        for d in doping:        
            print(f"rpa for {p * 180/jnp.pi}, {d}, {len(flake)}")

            # doping
            flake.set_electrons(el + d)

            pols.append(pol(flake))
        
        ret =  {"pol" : pols, "omega" : omega, "doping" : doping}

        # save to disk
        name  = f"rpa_{suffix}_{len(flake)}_{p}.npz"
        jnp.savez(name, **ret)
        names.append(name)
    
    return names

def plot_rpa_sim(names):
    
    def plot_matrix(name, omega, doping, pol):
        labels = [f"{d}" for d in doping]

        # Prepare the data array Z with shape (len(omega), len(phis))
        Z = jnp.abs(jnp.array([pol_i.imag * omega for pol_i in pol]).T)**(0.25)  # Transpose to match dimensions

        # Create the plot
        plt.figure()
        cax = plt.matshow(Z, aspect='auto', origin='lower')

        # Set x-axis ticks to correspond to phi values
        plt.xticks(ticks=jnp.arange(len(doping)), labels=[f"{d}" for d in doping], rotation=90)

        # Set y-axis ticks to correspond to omega values (sparsely to avoid clutter)
        num_yticks = 10  # Adjust this number based on how many ticks you want
        yticks_positions = jnp.linspace(0, len(omega) - 1, num=num_yticks, dtype=int)
        plt.yticks(ticks=yticks_positions, labels=[f"{omega[i]:.2f}" for i in yticks_positions])

        # Add colorbar and labels
        plt.colorbar(cax, label='Im(pol) * omega')
        plt.xlabel('# dopants')
        plt.ylabel(r'$\omega$')

        # Save and close the figure
        plt.savefig(name.replace("npz", "pdf"))
        plt.close()
    
    for name in names:
        res = jnp.load(name)
        plot_matrix(name, res["omega"], res["doping"], res["pol"])
        
def plot_rpa_sim(names):
    
    def plot_matrix(name, omega, doping, pol):
        labels = [f"{d}" for d in doping]

        # Prepare the data array Z with shape (len(omega), len(phis))
        Z = jnp.abs(jnp.array([pol_i.imag * omega for pol_i in pol]).T)**(0.25)  # Transpose to match dimensions

        # Create the plot
        plt.figure()
        cax = plt.matshow(Z, aspect='auto', origin='lower')

        # Set x-axis ticks to correspond to phi values
        plt.xticks(ticks=jnp.arange(len(doping)), labels=[f"{d}" for d in doping], rotation=90)

        # Set y-axis ticks to correspond to omega values (sparsely to avoid clutter)
        num_yticks = 10  # Adjust this number based on how many ticks you want
        yticks_positions = jnp.linspace(0, len(omega) - 1, num=num_yticks, dtype=int)
        plt.yticks(ticks=yticks_positions, labels=[f"{omega[i]:.2f}" for i in yticks_positions])

        # Add colorbar and labels
        plt.colorbar(cax, label='Im(pol) * omega')
        plt.xlabel('# dopants')
        plt.ylabel(r'$\omega$')

        # Save and close the figure
        plt.savefig(name.replace("npz", "pdf"))
        plt.close()
    
    for name in names:
        res = jnp.load(name)
        plot_matrix(name, res["omega"], res["doping"], res["pol"])

if __name__ == '__main__':
        
    shape = Hexagon(10, armchair = True)
    angles = jnp.array([0])
    omega = jnp.linspace(0, 10, 100)
    print("angles ", 180 / jnp.pi * angles)
    doping = jnp.arange(10)
    names = rpa_sim(shape, angles, doping, omega, "sym")
    plot_rpa_sim(names)
