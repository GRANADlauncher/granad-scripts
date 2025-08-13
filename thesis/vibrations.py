from granad import *

import jax.numpy as jnp
import matplotlib.pyplot as plt

# vibration: flake is fixed at y_max, y_min, such that there is a standing wave oscillating in y-direction, propagating along x
def get_phonon(positions, u, k):
    vt = 1
    wn = vt*k
    def inner(t):
        strain = u * jnp.sin(k * positions) * jnp.cos(wn * t) * jnp.exp(-10 * t)
        return strain
    return inner

def get_ham(ham_0, phonon, beta = 3, a_0 = 1.42):
    def inner(time, rho, args):
        strain = phonon(time)
        d = jnp.linalg.norm(strain - strain[:, None, :], axis=-1)
        ham_linear= -ham_0 * (beta * d / a_0)
        return ham_0 + ham_linear    
    return inner


flake = get_graphene().cut_flake(Rhomboid(20,20))
flake.set_electrons(len(flake) + 2)

# initialize vibrational mode
u0 = 0.05 * 1.42
amplitudes = jnp.array([0, u0, 0])
max_l = jnp.abs(flake.positions[:, 1] - flake.positions[:, 1, None]).max()

n = 5
k = n * jnp.pi / max_l

phonon = get_phonon(flake.positions, amplitudes, k)

# strain = phonon(0)
# print(jnp.linalg.norm(strain - strain[:, None, :], axis=-1))
# 1/0

hamiltonian_model = flake.get_hamiltonian()
hamiltonian_model["bare_hamiltonian"] = get_ham(flake.hamiltonian, phonon)

result = flake.master_equation( hamiltonian = hamiltonian_model,
                                expectation_values = [flake.dipole_operator],
                                relaxation_rate = 1/10,
                                end_time = 80)


# Main figure
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(result.time_axis, result.output[0][:, 0], linewidth=2, color='tab:blue')
ax.set_xlabel("Time (fs)")
ax.set_ylabel("Dipole Moment (a.u.)")
ax.set_title("Dipole Moment")
ax.grid(True, linestyle=':', alpha=0.6)
plt.show()

# spectral range where the absorption cross section should be computed:
omega_max = 10
omega_min = 0
p_omega = result.ft_output( omega_max, omega_min )[0] # polarization in the frequency domain
omegas_td, pulse_omega = result.ft_illumination( omega_max, omega_min ) # illuminating field in the frequency domain
absorption_td = jnp.abs( -omegas_td * jnp.imag( p_omega[:,0] / pulse_omega[:,0] ) ) # absorption cross section evaluated based on the time-domain simulation

plt.plot(omegas_td, jnp.abs(p_omega))
plt.show()
plt.close()

# lrt
omegas = jnp.linspace(0, 20, 100)
h = get_ham(flake.hamiltonian, phonon)
perturbation = h(0, 0, 0) - flake.hamiltonian
p_e = flake.transform_to_energy_basis(perturbation)
gf = flake.get_ip_green_function(p_e, flake.dipole_operator_e[0], omegas)
plt.plot(omegas, gf.imag)
plt.show()
