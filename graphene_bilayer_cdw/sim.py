import jax.numpy as jnp

from granad import *

def estimate_n(phi):
    """approximate number of particles in the unit cell for moiré angle phi"""
    return 4/phi**2

def get_bilayer_graphene(shape, phi, d0 = 0.335 * 10, doping = 0):
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
        rd2 = (r[2]/d)**2
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
        p = flake.dipole_operator_e
        sus.append(get_correlator(p[:2]))

    return {"sus" : sus, "phi" : phi}

def scf_loop(flake, coulomb, mixing, limit, max_steps):
    """performs closed-shell scf calculation

    Returns:
        rho, ham_eff
    """
    
    def update(arg):
        """scf update"""
        
        rho_old, step, error = arg

        # H = H_+ + H_-
        ham_eff =  ham_0 + 2 * jnp.diag(coulomb @ rho.diagonal())

        # diagonalize
        vals, vecs = jnp.linalg.eigh(ham_eff)

        # new density
        rho = 2*vecs[:, :N] @ vecs[:, :N].T + mixing * rho_old
        
        # update breaks
        error = jnp.linalg.norm(rho - rho_old) 

        step = jax.lax.cond(error <= limit, lambda x: step, lambda x: step + 1, step)

        return rho, step, error
    
    def step(idx, res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= limit, lambda x: res, update, res)

    ham_0 = flake.hamiltonian
    
    # GRANAD gives a closed-shell hamiltonian => for hubbard model, we split it into 2 NxN matrices, one for each spin component
    N, _ = ham_0.shape

    # initial guess for the density matrix
    rho_old = jnp.zeros_like(ham_0)

    # scf loop
    rho, steps, error = jax.lax.fori_loop(0, max_steps, step, (rho_old, 0, jnp.inf))
    
    print(f"{steps} / {max_steps}")

    return (rho,
            ham_0 + 2 * jnp.diag(coulomb @ rho.diagonal()))

def cdw_strength(rho):
    return jnp.std(rho.diagonal())

def saveas(name, d):
    jnp.savez(name, **d)

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
    plot_omega_dipole(name, 6*omega_0, 0, omega_0)
    plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm)

def td_sim(shape, phi):
     # params: Q = 2, N = 330 armchair, light : frequency = 0.68 eV, fwhm = 166fs, pol perp to triangle side, duration: 700, peak at 200
    name, end_time, amplitudes, omega, peak, fwhm = "phi", 700, [0.03, 0, 0], 0.68, 0.659 * 200, 0.659 * 166

    for p in phi:
        flake = get_bilayer_graphene(shape, phi)

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
        plot_omega_dipole(name + f"{p:.2f}", 6*omega_0, 0, omega_0)
        plot_t_dipole(name, end_time, amplitudes, omega, peak, fwhm)
    
def plot_ip_sim(shape, phi):
    omegas = jnp.linspace(0, 20, 100)
    sus = ip_response(shape, phi, omegas)["sus"]
    for i, s in enumerate(sus):
        plt.plot(omegas, s * omegas, label = f"{p:.2f}")
    plt.savefig('ip.pdf')
    plt.close()
    
def plot_energy_sim(shape, phi):
    for p in phi:
        flake = get_bilayer_graphene(shape, phi)
        plt.plot(jnp.arange(len(flake)), flake.energies, 'o', label = f"{p:.2f}")
    plt.savefig('energies.pdf')
    plt.close()
    
RUN_REF = True
RUN_IP = True
RUN_ENERGY = True
RUN_TD = True

if __name__ == '__main__':
    
    if RUN_REF:
        ref()
    
    shape, phi = Triangle(45, armchair = True), jnp.linspace(0, jnp.pi/2, 10)
    
    if RUN_ENERGY:
        plot_energy_sim(shape, phi)
    
    if RUN_IP:
        plot_ip_sim(shape, phi)

    if RUN_TD::
        td_sim(shape, phi)
    
