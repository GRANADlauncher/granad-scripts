import matplotlib.pyplot as plt

import jax.numpy as jnp

from granad import *

def estimate_n(phi):
    """approximate number of particles in the unit cell for moiré angle phi"""
    return 4/phi**2

def get_bilayer_graphene(shape, phi, interlayer_scaling = 1.0, d0 = 0.335 * 10, doping = 0):
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

    def coulomb_coupling(r):
        return u0/(eps_b * (jnp.linalg.norm(r) + 1)) + 0j

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

    flake.set_coulomb_groups(upper, upper, coulomb_coupling)
    flake.set_coulomb_groups(upper, lower, coulomb_coupling)
    flake.set_coulomb_groups(lower, lower, coulomb_coupling)
            
    return flake

def ip_response(shape, phi, omegas):

    def get_correlator(operators):
        return jnp.array([[flake.get_ip_green_function(o1, o2, omegas, relaxation_rate = 0.1) for o1 in operators] for o2 in operators])

    sus = []
    for p in phi:
        flake = get_bilayer_graphene(shape, p)
        p = flake.dipole_operator_e
        sus.append(get_correlator(p[:2]))

    return {"sus" : sus, "phi" : phi}

def scf_sweep(shape, phi):
    """performs a sweep over phi, plots the resulting cdw order parameter"""

    order_params = []
    
    for p in phi:
        flake = get_bilayer_graphene(shape, p)
        
        # display real space picture of naive solution
        flake.show_2d(display = flake.initial_density_matrix_x.diagonal(), name = f"{p:.2f}_naive.pdf")
        
        r, _ =  scf_loop(flake, 0.01 * flake.coulomb, 1e-3, 1e-9, 100)
        print("trace ", jnp.trace(r), "\n electrons: ", len(flake), cdw_strength(r))
        order_params.append(cdw_strength(r))

        # display real space picture of scf solution
        flake.show_2d(display = r.diagonal(), name = f"{p:.2f}.pdf")
        

    plt.plot(phi, order_params)
    plt.savefig("scf_sweep.pdf")
    plt.close()
    
    return {"param" : order_params, "phi" : phi}

def scf_loop(flake, coulomb, mixing, limit, max_steps):
    """performs closed-shell scf calculation

    Returns:
        rho, ham_eff
    """
    
    def update(arg):
        """scf update"""
        
        rho_old, step, error = arg

        # H = H_+ + H_-
        ham_eff =  ham_0 + 2 * jnp.diag(coulomb @ rho_old.diagonal())

        # diagonalize
        vals, vecs = jnp.linalg.eigh(ham_eff)

        # new density
        rho = (1-mixing) * 2*vecs[:, :N] @ vecs[:, :N].T + mixing * rho_old
        
        # update breaks
        error = jnp.linalg.norm(rho - rho_old) 

        step = jax.lax.cond(error <= limit, lambda x: step, lambda x: step + 1, step)

        return rho, step, error
    
    def step(idx, res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= limit, lambda x: res, update, res)

    ham_0 = flake.hamiltonian
    
    # GRANAD gives a closed-shell hamiltonian => for hubbard model, we split it into 2 NxN matrices, one for each spin component
    N  = ham_0.shape[0] // 2

    # initial guess for the density matrix
    rho_old = jnp.zeros_like(ham_0)

    # scf loop
    rho, steps, error = jax.lax.fori_loop(0, max_steps, step, (rho_old, 0, jnp.inf))
    
    print(f"{steps} / {max_steps}, {error}")

    return (rho,
            ham_0 + 2 * jnp.diag(coulomb @ rho.diagonal()))

def cdw_strength(rho):
    return jnp.std(rho.diagonal())

def saveas(name, kwargs):
    jnp.savez(name, **kwargs)

def get_dip(name, omega_max, omega_min, omega_0):
    """returns induced dipole moment, normalized to its value at omega_0
    and omega axis.
    """
    
    result = TDResult.load(name)
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas, _ = result.ft_illumination( omega_max, omega_min )
    closest_index = jnp.argmin(jnp.abs(omegas - omega_0))
    p_0 = 1.0#jnp.linalg.norm(p_omega[closest_index])
    p_normalized = jnp.linalg.norm(p_omega, axis = -1) / p_0
    return omegas, p_normalized

def plot_omega_dipole(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.semilogy(omegas / omega_0, p)
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()

def ip_plot_omega_dipole(name, omega_max, omega_min, omega_0):
    omegas, p = get_dip(name, omega_max, omega_min, omega_0)
    plt.semilogy(omegas / omega_0, p)
    plt.savefig(f"omega_dipole_moment_{name}.pdf")
    plt.close()

def plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm):
    time = jnp.linspace(0, end_time, 1000)
    pulse = Pulse(amplitudes, omega, peak, fwhm)
    e_field = jax.vmap(pulse) (time)
    plt.plot(time, e_field.real)
    # plt.plot(time, e_field.imag, '--')
    result = TDResult.load(name)        
    plt.plot(result.time_axis, result.output[0], '--')
    plt.savefig(f"t_dipole_moment_{name}.pdf")
    plt.close()
    
def ref():
    # params: Q = 2, N = 330 armchair, light : frequency = 0.68 eV, fwhm = 166fs, pol perp to triangle side, duration: 700, peak at 200
    flake = MaterialCatalog.get("graphene").cut_flake(Triangle(45, armchair = True))
    flake.set_electrons(flake.electrons + 2)
    flake.show_energies(name = "energies")

    name, end_time, amplitudes, omega, peak, fwhm = "cox_50_1e-4_new", 700, [0.03, 0, 0], 0.68, 0.659 * 200, 0.659 * 166
    
    result = flake.master_equation(
        dt = 1e-4,
        end_time = end_time,
        relaxation_rate = 1/10,
        expectation_values = [ flake.dipole_operator ],
        illumination = Pulse(amplitudes, omega, peak, fwhm),
        max_mem_gb = 50,
        grid = 100
    )
    result.save(name)        
    plot_omega_dipole(name, 6*omega, 0, omega)
    plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm)

def td_sim(shape, phi):
     # params: Q = 2, N = 330 armchair, light : frequency = 0.68 eV, fwhm = 166fs, pol perp to triangle side, duration: 700, peak at 200
    name, end_time, amplitudes, omega, peak, fwhm = "phi", 700, [0.03, 0, 0], 0.68, 0.659 * 200, 0.659 * 166

    for p in phi:
        flake = get_bilayer_graphene(shape, p)

        result = flake.master_equation(
            dt = 1e-4,
            end_time = end_time,
            relaxation_rate = 1/10,
            expectation_values = [ flake.dipole_operator ],
            illumination = Pulse(amplitudes, omega, peak, fwhm),
            max_mem_gb = 50,
            grid = 100
        )
        result.save(name + f"{p:.2f}")

    for p in phi:
        plot_omega_dipole(name + f"{p:.2f}", 6*omega, 0, omega)
        plot_t_dipole(name + f"{p:.2f}", end_time, amplitudes, omega, peak, fwhm)

def rpa_sim(shape, phi, omega):
    def pol(flake):            
        return flake.get_polarizability_rpa(
            omega,
            relaxation_rate = 1/100,
            polarization = 0, 
            hungry = 1)

    flake_free = MaterialCatalog.get("graphene").cut_flake(shape)

    # doping
    flake_free.set_electrons(flake_free.electrons + 2)
    
    pols = [pol(flake_free)]

    for p in phi:
        flake = get_bilayer_graphene(shape, p)
        print(f"rpa for {p}, {len(flake)}")
        
        # doping
        flake.set_electrons(flake.electrons + 4)
        
        pols.append(pol(flake))
    
    return {"pol" : pols, "omega" : omega, "phi" : phi }

def plot_rpa_sim(omega, pol, phi):
    labels = ["free"] + [f"{p:.2f}" for p in phi]
    
    for i, label in enumerate(labels):
        plt.plot(omega, pol[i].imag * omega, label = label)
        
    plt.legend()
    plt.savefig("rpa_sim.pdf")
    plt.close()
    
def plot_ip_sim(shape, phi):
    # Assuming these values are provided
    omegas = jnp.linspace(0, 20, 100)
    sus = ip_response(shape, phi, omegas)["sus"]  # sus is assumed to be a 2x2xN matrix

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plot each subplot in the 2x2 grid
    for k, s in enumerate(sus):
        for i in range(2):
            for j in range(2):
                ax = axs[i, j]
                ax.plot(omegas, jnp.abs(s[i, j]) * omegas, label=f"{phi[k]:.2f}")  # Flattening indices to match phi
                ax.set_title(f"i, j")
                ax.legend()

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig('ip.pdf')
    plt.close()


def plot_energy_sim(shape, phi, scf = True):
    # Create a grid for the subplots
    n = len(phi)
    ncols = 2
    nrows = (n + ncols - 1) // ncols  # Calculate the number of rows based on the number of phi

    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows))  # Adjust the figure size as needed
    axs = axs.flatten()  # Flatten to 1D array for easy indexing

    # Iterate over phi and create subplots
    for i, p in enumerate(phi):
        flake = get_bilayer_graphene(shape, p)  # Assuming this function generates flake for each phi

        # if scf:            
        #     r, ham = scf_loop(flake, 1/6 * flake.coulomb, 0.01, 1e-5, 400)            
        #     vals, vecs = jnp.linalg.eigh(ham)
        #     axs[i].plot(jnp.arange(len(flake.energies)), vals, 'o')

        flake.show_2d(name = f"geometry_{p:.2f}.pdf", show_index = True)
                        
        axs[i].scatter(jnp.arange(len(flake.energies)), flake.energies, c = flake.initial_density_matrix_e.diagonal() * len(flake) )
        axs[i].set_title(f"phi = {p:.2f}")
        axs[i].set_xlabel('Index')  # Optional: label for x-axis
        axs[i].set_ylabel('Energy')  # Optional: label for y-axis

    # Remove unused subplots, if any
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.savefig('energies.pdf')
    plt.close()

RUN_REF = False
RUN_IP = False
RUN_TD = False
RUN_SCF_SWEEP = False
RUN_RPA = True
RUN_ENERGY = False

if __name__ == '__main__':
    
    if RUN_REF:
        ref()
    
    shape, phi = Hexagon(20, armchair = True), jnp.linspace(0, jnp.pi/2, 20)

    if RUN_RPA:
        res = rpa_sim(shape, phi, jnp.linspace(0, 1, 100))
        plot_rpa_sim(**res)

    if RUN_SCF_SWEEP:
        scf_sweep(shape, phi)
    
    if RUN_ENERGY:
        plot_energy_sim(shape, phi)
    
    if RUN_IP:
        plot_ip_sim(shape, phi)

    if RUN_TD:
        td_sim(shape, phi)
