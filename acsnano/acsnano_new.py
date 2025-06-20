import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from granad import MaterialCatalog, Rhomboid, Pulse, get_fourier_transform
import gc

def load_reference_data():
    data = jnp.array(np.genfromtxt('acsnano.csv', delimiter=';'))
    h = 6.6261e-34
    c = 2.9979e8
    frequencies = h * c / (data[:, 0] * 1e-9) / (1.6e-19)
    valid_indices = jnp.argwhere(data[:, 0] < 800)
    return frequencies[valid_indices][:, 0], data[valid_indices, 1] / jnp.max(data[valid_indices, 1])

def td_sim( flake, omega_min, omega_max ):
    pulse = Pulse(
            amplitudes=[1e-5, 0, 0], frequency=2.3, peak=2, fwhm=0.5
        )    

    time_points = jnp.linspace(0, 100, 100)
    delta_time = time_points[1] - time_points[0]
    initial_density_matrix = flake.initial_density_matrix    
    t, p = [], []
    
    for start_time in time_points:
        print(start_time, start_time + delta_time)
        time_axis, density_matrices =  flake.get_density_matrix_time_domain(
            start_time = start_time,
            end_time = start_time + delta_time,
            initial_density_matrix = initial_density_matrix,
            steps_time=1e3,
            relaxation_rate=1/10,
            illumination=pulse,
            skip=5,
        )
        p.append( np.array(flake.get_expectation_value(density_matrix = density_matrices, operator = flake.dipole_operator)) )
        t.append(time_axis)
        initial_density_matrix = density_matrices[-1].copy()
        
        del density_matrices
        gc.collect()

    time_axis = jnp.concatenate(t)
    omegas, p_omega = get_fourier_transform(time_axis, jnp.concatenate(p))
    _, pulse_omega = get_fourier_transform(time_axis, jax.vmap(pulse)(time_axis))
    mask = (omegas >= omega_min) & (omegas <= omega_max)
    omegas = omegas[mask]
    p_omega = p_omega[mask]
    pulse_omega = pulse_omega[mask]
    return omegas, jnp.abs( -omegas * jnp.imag( p_omega[:,0] / pulse_omega[:,0] ) )


def sim( length, hopping = -2.66, rpa = True, td = False):
    
    graphene = MaterialCatalog.get( "graphene" )
    graphene.add_interaction('hopping', ('pz', 'pz'), [0, hopping])
    flake = graphene.cut_flake( Rhomboid(18, length, armchair = True), plot = False )
    plt.close()    
    print( len(flake) )

    omegas_rpa, ref_data = load_reference_data()
    if rpa:
        start_time = time.time()
        polarizability = flake.get_polarizability_rpa(
            omegas_rpa, #.reshape(1, omegas_rpa.size),
            relaxation_rate = 1/10,
            polarization = 0, 
            hungry = 1 )
        absorption_rpa = jnp.abs( polarizability.imag * 4 * jnp.pi * omegas_rpa )
        print(f"Simulation time: {time.time() - start_time}")
        
    if td:
        import diffrax
        # start_time = time.time()
        # omegas_td, absorption_td = td_sim( flake, jnp.min(omegas_rpa), jnp.max(omegas_rpa) )
        # print(f"Simulation time: {time.time() - start_time}")    
        pulse = Pulse(
            amplitudes=[1e-5, 0, 0], frequency=2.3, peak=2, fwhm=0.5
        )
        result = flake.master_equation(
            expectation_values = [ flake.dipole_operator ],
            end_time=100,
            dt = 1e-3,
            grid = 10,
            stepsize_controller=diffrax.PIDController(atol=1e-10, rtol=1e-10),
            relaxation_rate=1/10,
            illumination=pulse,
        )
        omega_max = omegas_rpa.max()
        omega_min = omegas_rpa.min()
        p_omega = result.ft_output( omega_max, omega_min )[0]
        omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
        absorption_td = jnp.abs( -omegas_td * jnp.imag( p_omega[:,0] / pulse_omega[:,0] ) )

        # time_axis, density_matrices =  flake.get_density_matrix_time_domain(
        #         end_time=100,
        #         steps_time=1e5,
        #         relaxation_rate=1/10,
        #         illumination=pulse,
        #         skip=100,    
        # )
        # omegas, p_omega = flake.get_expectation_value_frequency_domain(
        #     operator=flake.dipole_operator,  
        #     time=time_axis,
        #     density_matrices = density_matrices,
        #     omega_min=jnp.min(omegas_rpa) - 0.5,
        #     omega_max=jnp.max(omegas_rpa) + 1.5,
        # )            
        # absorption_td = jnp.abs( -omegas_td * jnp.imag( dipole_omega[:,0] / pulse_omega[:,0] ) )
        
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    if rpa:
        plt.plot(omegas_rpa, absorption_rpa / jnp.max(absorption_rpa), '-', linewidth=2, label = 'RPA')
    if td:
        plt.plot(omegas_td, absorption_td / jnp.max(absorption_td), linewidth=2, ls = '--', label = 'TD' )
    plt.plot(omegas_rpa, ref_data, 'o', label='Reference')
    plt.xlabel(r'$\hbar\omega$', fontsize=20)
    plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
    plt.title('Absorption Spectrum as a Function of Photon Energy', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'acsnano_res_td_{hopping}.pdf')

if __name__ == '__main__':
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))    
    for length in [180]:
        for hopping in [-2.66]:
            sim(length, hopping = hopping, rpa = False, td = True)
