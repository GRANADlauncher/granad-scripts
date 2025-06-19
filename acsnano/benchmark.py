# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: base
#     language: python
#     name: base
# ---

# # Linear response
#

# We will calculate the optical absorption in the RPA and compare it to TD simulations with a weak external field.

### RPA

# First, we set up the RPA simulation. We will consider a small triangle such that the required simulation time stays in the seconds range.

# +
import jax
import jax.numpy as jnp
import time
import os
import diffrax
from collections import namedtuple
from granad import MaterialCatalog, Rectangle, _numerics, Pulse, TDResult, Triangle
import matplotlib.pyplot as plt

def build( size ):
    # get material
    graphene = MaterialCatalog.get( "graphene" )
    
    # cut a 15 Angström wide triangle from the lattice (can also be an arbitrary polygon)
    flake = graphene.cut_flake( Triangle(15) )
    
    flake = graphene.cut_flake( Rectangle( size, size ), plot = True )
    plt.savefig('geometry.pdf')
    print(len(flake))
    
    flake._build()

    # Define the namedtuple type
    CustomArgs = namedtuple('CustomArgs', ['hamiltonian', 'coulomb', 'dipole', 'initial_density_matrix', 'stationary_density_matrix', 'electrons', 'relaxation_rate'] )
    rhs_args = CustomArgs( hamiltonian = flake.hamiltonian, coulomb = flake.coulomb, dipole = flake.dipole_operator[:2].diagonal(axis1=-1, axis2=-2), initial_density_matrix = flake.initial_density_matrix, stationary_density_matrix = flake.stationary_density_matrix, electrons = flake.electrons, relaxation_rate = 1/10)
    jnp.savez(savefile, **rhs_args._asdict())
    print("Finished")

    
def sim( size ):
    solver = diffrax.Ralston()
    stepsize_controller = diffrax.PIDController(rtol=1e-10,atol=1e-10)
    # solver = diffrax.Euler()
    # stepsize_controller = diffrax.ConstantStepSize()
    # solver = diffrax.Dopri5()
    # stepsize_controller = diffrax.PIDController(rtol=1e-10,atol=1e-10)    
    start_time = 0.0
    end_time = 40.0
    dt = 1e-6
    grid = 1000    
    max_mem_gb = 4

    with open(savefile, 'rb') as f:
        data = jnp.load(f)
        CustomArgs = namedtuple('CustomArgs', data.keys() )
        rhs_args = CustomArgs(**data)
    
    initial_density_matrix = rhs_args.initial_density_matrix
    print(f"size : {len(initial_density_matrix)}")
    
    pulse = Pulse(
        amplitudes=[1e-5, 0], frequency=2.3, peak=5, fwhm=2
    )


    ham_dip = lambda t, r, args : args.hamiltonian + jnp.diag(args.coulomb @ (r-args.stationary_density_matrix).diagonal() * args.electrons + jnp.einsum('Ki,K->i', args.dipole, pulse(t).real))       
    hamiltonian = [ ham_dip ]

    dec = lambda t,r,args: -(r - args.stationary_density_matrix) * args.relaxation_rate
    dissipator = [dec]

    induced_dip = lambda r,args : jnp.einsum( 'ti,Ki->tK', args.electrons*(args.stationary_density_matrix.diagonal() - r.diagonal(axis1=-1,axis2=-2)), args.dipole )
    postprocesses = [induced_dip]

    # start = time.time(); h_times_d = args.hamiltonian @ r; hermitian_term = -1j * (h_times_d  - h_times_d.conj().T); print( time.time() - start )
    # start = time.time(); hermitian_term = -1j * (hamiltonian @ r  - (hamiltonian @ r).conj().T); print( time.time() - start )
    # def my_rhs( time, density_matrix, args ): h_total = ham_dip(time, density_matrix, args); h_times_d = h_total @ density_matrix; return -1j * (h_times_d  - h_times_d.conj().T) + dec(time, density_matrix, args)
    # my_rhs_jit = jax.jit(my_rhs)    
    # start = time.time(); rhs(0, args.initial_density_matrix, args); print(time.time() - start)
    # # import pdb; pdb.set_trace()
    # # import time
    # # start = time.time(); rhs(0.0, rhs_args.initial_density_matrix, rhs_args); print( time.time() - start )
    # start = time.time(); args.hamiltonian @ args.hamiltonian; print( time.time() - start )
    # # timeit.timeit('rhs(0.0, rhs_args.initial_density_matrix, rhs_args)', number = 1)
    
    # batched time axis to save memory 
    mat_size = initial_density_matrix.size * initial_density_matrix.itemsize / 1e9
    time_axis = _numerics.get_time_axis( mat_size = mat_size,
                                         grid = grid,
                                         start_time = start_time,
                                         end_time = end_time,
                                         max_mem_gb = max_mem_gb,
                                         dt = dt )

    ## integrate
    start=time.time()
    final, output = _numerics.td_run(
        initial_density_matrix,
        _numerics.get_integrator(list(hamiltonian),
                                 list(dissipator),
                                 list(postprocesses),
                                 solver,
                                 stepsize_controller,
                                 dt),
        time_axis,
        rhs_args)
    print(time.time()-start)
    
    result = TDResult(
        td_illumination = jax.vmap(pulse)(jnp.concatenate(time_axis)) ,
        output = output,
        final_density_matrix = final,
        time_axis = jnp.concatenate( time_axis )
    )

    result.save(  resfile.replace('.npz', '') )

def plot_results():
    result = TDResult.load(resfile.replace('.npz', ''))
    # import pdb; pdb.set_trace()
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(result.time_axis, result.output[0], linewidth=2, ls = '--', label = 'TD' ) 
    plt.xlabel(r'$\hbar\omega$', fontsize=20)
    plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
    plt.title('Absorption Spectrum as a Function of Photon Energy', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result_dip_{len(result.final_density_matrix)}.pdf')

    
    omega_max = 20
    omega_min = 0
    p_omega = result.ft_output( omega_max, omega_min )[0]
    omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min )
    absorption_td = jnp.abs( p_omega[:,0] ) 

    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    plt.plot(omegas_td, absorption_td / jnp.max(absorption_td), linewidth=2, ls = '--', label = 'TD' ) 
    plt.xlabel(r'$\hbar\omega$', fontsize=20)
    plt.ylabel(r'$\sigma(\omega)$', fontsize=25)
    plt.title('Absorption Spectrum as a Function of Photon Energy', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'result_{len(result.final_density_matrix)}.pdf')


if __name__ == '__main__':
    size = 117
    savefile = f"/scratch/local/ddams/save_{size}.npz"
    resfile = f"/scratch/local/ddams/res_{size}_bosh.npz"

    if os.path.isfile( savefile ) == False:
        build( size )

    if os.path.isfile( resfile ) == False:
        sim(size)
        
    plot_results(  )
