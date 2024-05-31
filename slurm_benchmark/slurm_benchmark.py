import time
import pprint

import jax
import diffrax

from granad import *

def td_optimized(flake, params):
    """runs a td sim for the induced dipole moment discarding redundant matrix elements in the dipole operator. assumes no intra-atomic transitions."""
    
    from collections import namedtuple

    # arguments passed explicitly to rhs of the master equation, avoiding capturing big arrays in closures
    CustomArgs = namedtuple('CustomArgs', ['hamiltonian', 'coulomb', 'dipole', 'initial_density_matrix', 'stationary_density_matrix', 'electrons', 'relaxation_rate'] )    
    rhs_args = CustomArgs( hamiltonian = flake.hamiltonian,
                           coulomb = flake.coulomb,
                           dipole = flake.dipole_operator[:2].diagonal(axis1=-1, axis2=-2),
                           initial_density_matrix = flake.initial_density_matrix,
                           stationary_density_matrix = flake.stationary_density_matrix,
                           electrons = flake.electrons,
                           relaxation_rate = params["relaxation_rate"])
    f = params["illumination_optimized"]

    # hamiltonian function
    ham = lambda t, r, args : args.hamiltonian + jnp.diag(args.coulomb @ (r-args.stationary_density_matrix).diagonal() * args.electrons + jnp.einsum('Ki,K->i', args.dipole, f(t).real))       

    # decoherence function
    dec = lambda t,r,args: -(r - args.stationary_density_matrix) * args.relaxation_rate

    # induced dipole moment
    post = jax.jit(lambda r,args : jnp.einsum( 'tii,Ki->tK', args.electrons*(args.stationary_density_matrix - r), args.dipole ))

    # integration
    result = OrbitalList._integrate_master_equation( [ham],
                                                     [dec],
                                                     [post],
                                                     rhs_args,
                                                     params["illumination_optimized"],
                                                     params["solver"],
                                                     params["stepsize_controller"],
                                                     rhs_args.initial_density_matrix,
                                                     params["start_time"],
                                                     params["end_time"],
                                                     params["grid"],
                                                     params["max_mem_gb"],
                                                     params["dt"]
                                                    )
    return result

def sim(params):

    # simulation summary
    runtimes = {"cutting" : None, "diag" : None, "td_default" : None, "td_optimized" : None}
    summary = {"atoms" : None, "runtimes" : runtimes}

    ### Cutting the flake ###
    start = time.time()
    flake = MaterialCatalog.get("graphene").cut_flake(params["shape"])
    summary["runtimes"]["cutting"] = time.time() - start
    summary["atoms"] = len(flake)

    ### Setting up Arrays (Hamiltonian, Coulomb, density matrix, ...) ###
    if params["run_diag"] == True:
        start = time.time()
        flake.hamiltonian
        summary["runtimes"]["diag"] = time.time() - start


    ### Time propagation ###
    if params["run_td_default"] == True:
        start = time.time()
        result_default = flake.master_equation(
            start_time = params["start_time"],
            end_time = params["end_time"],
            relaxation_rate=params["relaxation_rate"],
            illumination=params["illumination_default"],
            expectation_values = [flake.dipole_operator] 
        )
        summary["runtimes"]["td_default"] = time.time() - start

    ### Optimized Time propagation ###
    if params["run_td_optimized"] == True:
        start = time.time()
        result_optimized = td_optimized(flake, params)
        summary["runtimes"]["td_optimized"] = time.time() - start

    ### sanity check ###
    if params["plot"] == True:
        import matplotlib.pyplot as plt
        plt.plot( result_default.time_axis, result_default.output[0] )
        plt.plot( result_optimized.time_axis, result_optimized.output[0], '--' )
        plt.show()

    return summary

def get_params():
    """returns a dictionary containing simulation parameters"""

    params = {
        "grid": 1000,
        "max_mem_gb": 50,
        "dt": 1e-5,
        "start_time": 0,
        "end_time": 40,
        "relaxation_rate": 1/10,
        "solver": diffrax.Dopri5(),
        "stepsize_controller": diffrax.PIDController(rtol=1e-10, atol=1e-10),
        "illumination_default": Pulse(amplitudes=[1e-5, 0, 0], frequency=2.3, peak=5, fwhm=2),
        "illumination_optimized": Pulse(amplitudes=[1e-5, 0], frequency=2.3, peak=5, fwhm=2),
        "shape": Triangle(10),
        "name": "triangle_10",
        "run_diag": True,
        "run_td_default": True,
        "run_td_optimized": True,
        "plot" : False,
    }

    return params

if __name__ == '__main__':       
    print(sim(get_params()))
