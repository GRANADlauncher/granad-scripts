import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import diffrax

from granad import MaterialCatalog, Rhomboid, Pulse, get_fourier_transform, TDResult, get_graphene

def load_reference_data():
    data = jnp.array(np.genfromtxt('acsnano.csv', delimiter=';'))
    h = 6.6261e-34
    c = 2.9979e8
    frequencies = h * c / (data[:, 0] * 1e-9) / (1.6e-19)
    valid_indices = jnp.argwhere(data[:, 0] < 800)
    return frequencies[valid_indices][:, 0], data[valid_indices, 1] / jnp.max(data[valid_indices, 1])

def sim(length, hopping = -2.66, rpa = True, td = True, reload_td = True):    
    graphene = get_graphene(hoppings = [0, hopping])
    flake = graphene.cut_flake( Rhomboid(18, length, armchair = True), plot = False )
    plt.close()    
    print("Flake with atoms:", len(flake))

    omegas_rpa, ref_data = load_reference_data()
    if rpa:
        start_time = time.time()
        polarizability = flake.get_polarizability_rpa(
            omegas_rpa,
            relaxation_rate = 1/10,
            polarization = 0, 
            hungry = 1 )
        absorption_rpa = jnp.abs( polarizability.imag * 4 * jnp.pi * omegas_rpa )
        print(f"RPA Simulation time: {time.time() - start_time}")
        
    if td:
        start_time = time.time()
        pulse = Pulse(
            amplitudes=[1e-5, 0, 0], frequency=2.3, peak=2, fwhm=0.5
        )
        result = flake.master_equation(
            expectation_values = [ flake.dipole_operator ],
            end_time=100,
            dt = 1e-3,
            grid = 10,
            max_mem_gb = 40,
            stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
            relaxation_rate=1/10,
            illumination=pulse,
        )
        print(f"TD Simulation time: {time.time() - start_time}")
        result.save("res2.2")
        
    if reload_td:
        result = TDResult.load("res2.2")
        omega_max = omegas_rpa.max()
        omega_min = omegas_rpa.min()
        p_omega = result.ft_output( omega_max, omega_min )[0]
        omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
        absorption_td = jnp.abs( -omegas_td * jnp.imag( p_omega[:,0] / pulse_omega[:,0] ) )

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    if rpa:
        plt.plot(omegas_rpa, absorption_rpa / jnp.max(absorption_rpa), '-', linewidth=2, label = 'RPA')
    if reload_td:
        plt.plot(omegas_td, absorption_td / jnp.max(absorption_td), linewidth=2, ls = '--', label = 'TD' )
    plt.plot(omegas_rpa, ref_data, 'o', label='Reference')
    plt.xlabel(r'$\hbar\omega$', fontsize=20)
    plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
    plt.title('Absorption Spectrum as a Function of Photon Energy', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'acsnano_res_td_{hopping}.pdf')

if __name__ == '__main__':
    for length in [180]:
        for hopping in [-2.2]:
            sim(length, hopping = hopping, rpa = False, td = False, reload_td = True)
