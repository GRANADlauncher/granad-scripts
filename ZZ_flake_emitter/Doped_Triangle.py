import jax.numpy as jnp
import matplotlib.pyplot as plt
from granad._numerics import iterate, get_fourier_transform
import jax.numpy as jnp
from granad import *
from granad._plotting import *
def get_graphene_hubbard(U = 0, size = 20):
    graphene_spinful_hubbard = (
        Material("graphene_spinful_hubbard")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz+", atom='C')
        .add_orbital_species("pz-", atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz+")
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz-")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz+")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz-")
        .add_interaction(
            "hamiltonian",
            participants=("pz+", "pz+"),
            parameters=[0,-2.66, -0.1, -0.33 ],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz-", "pz-"),
            parameters=[0,-2.66, -0.1, -0.33 ],
        )
        .add_interaction(
            "coulomb",
            participants=("pz+", "pz-"),
            parameters=[U],
        )
    )

    shape = Triangle(size, armchair = False)
    flake = graphene_spinful_hubbard.cut_flake(shape) 
    flake.set_open_shell()
    flake.set_electrons(len(flake)//2)
    return flake

@iterate
def calculate_ImG_from_rho(rhos_diag_t, time_axis, dipole_position, induced_field_func,max_omega=4,min_omega=0):
    IF_t=jax.vmap(lambda r: induced_field_func(dipole_position,r))(rhos_diag_t)[:,0,:]
    omega,IF_w=get_fourier_transform(time_axis,IF_t[:,:],max_omega,min_omega)
    return omega,IF_w.imag

@iterate
def get_ImG_spectra(flake,coulomb=0,doping=0,dpm=jnp.array([1,0,0]),DP_X_idx=0,DP_Z_AA=5, dipole_position = None):
    print(flake.electrons)
    if dipole_position is None:
        dipole_position=flake.positions[DP_X_idx]+jnp.array([0,0,DP_Z_AA])

    time_axis=jnp.arange(0,400,1e-2)
    from granad.potentials import DipolePulse
    DP_func=DipolePulse(dipole_moment=dpm, source_location=dipole_position,
                        omega=4, sigma=1, t0=0.0, kick=True, dt = 1e-2)
    hamiltonian_DP_field=flake.get_hamiltonian()
    hamiltonian_DP_field["dipole_kick"]=DP_func

    result = flake.master_equation(end_time = 400, dt = 1e-2,
                                hamiltonian = hamiltonian_DP_field,
                                density_matrix=['diag_x'],
                                relaxation_rate = 1/40,
                                solver = "RK45",
                                max_mem_gb=50,
                                grid=1,
                                use_rwa=True,
                                coulomb_strength=coulomb # high accuracy explicit solver to avoid oscillating tails
                               )
    omega,ImG=calculate_ImG_from_rho(rhos_diag_t=result.output[0]-jnp.diagonal(flake.initial_density_matrix_x),
                           time_axis=result.time_axis,
                           dipole_position=dipole_position,
                           induced_field_func=flake.get_induced_field)
   
    return omega, ImG

def find_atoms_along_path(flake_pos,path):
    return jnp.unique(jax.vmap(lambda point : jnp.argmin(jnp.linalg.norm(flake_pos-point,axis=-1)) )(jnp.array(path)))
def get_emitter_positions(flake, points=100):
    x_min=jnp.argmin(flake.positions[:,0])
    x_max=jnp.argmax(flake.positions[:,0])
    mid_point=(flake.positions[x_min]+flake.positions[x_max])/2
    center_point=flake.positions[flake.center_index]
    linspace_x = jnp.linspace(mid_point[0], center_point[0], points)
    linspace_y = jnp.linspace(mid_point[1], center_point[1], points)
    line = jnp.stack([linspace_x, linspace_y, 5 * jnp.ones_like(linspace_x)]).T
    
    A=line[0]
    B=line[-1]
    k=.2
    extended_point=A-k*(B-A)
    #print(extended_point)
    line=jnp.vstack((extended_point, line))
    line2=line+jnp.array([1.42*jnp.sqrt(3)/2,0,0])
    final_line=jnp.vstack((line,line2))
    emitter_positions=find_atoms_along_path(flake.positions,final_line)
    return jnp.unique(emitter_positions,axis=0), line

def get_dopped_flake(size,U,doping):
    from granad import _numerics, Params
    flake = get_graphene_hubbard(U = U, size = size)
    flake.rotate(x=jnp.array([0,0,0]),phi=-jnp.pi/6)
    flake.set_electrons(len(flake)//2 + doping)
    flake.set_mean_field(iterations=1000, mix=0.35)    
    return flake


#==============Execution Code================#
U,size,dope_max=1,40,11#1,32,11
all_flakes =[ get_dopped_flake(size=size,U=U,doping=d) for d in range(dope_max+1)]
flake=all_flakes[0]
emitter_positions_idx,line = get_emitter_positions(flake, points=20)
emitter_positions=[dp+jnp.array([0,0,5]) for dp in flake.positions[emitter_positions_idx]]
emitter_positions=sorted(emitter_positions, key=lambda pos: pos[1])

#========= plot structure with emitter path ============#
plt.scatter(flake.positions[:,0], flake.positions[:,1])
plt.scatter(flake.positions[emitter_positions_idx,0], flake.positions[emitter_positions_idx,1], color="red")
plt.show()

# ======== run simulation =========================#

emitter_positions=[dp+jnp.array([0,0,5]) for dp in flake.positions[emitter_positions_idx]]
emitter_positions=sorted(emitter_positions, key=lambda pos: pos[1])
cs=[0,1]

o_path, img_path=get_ImG_spectra(flake=all_flakes,
                      coulomb=cs,
                      doping=None,
                      DP_X_idx=None,
                      DP_Z_AA=None,
                      dpm=[jnp.array([1,0,0]), jnp.array([0,1,0]), jnp.array([0,0,1])], #
                      dipole_position=emitter_positions)
#img shape (doping, coulomb, polarization, emitter_position, frequency, component)
data = jnp.array(img_path)
omega = jnp.array(o_path)[0,0,0,0,:]
print(data.shape)
print(omega.shape)
jnp.savez(f"Doped_Triangle{size}.npz", img=data, omega = omega)

# ========== Plotting ===========================#

import os
from itertools import product

# Ensure output directory exists
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

index = {0: 'x', 1: 'y', 2: 'z'}

# Loop through all combinations of parameters (0, 1, 2)
n_dope,n_coulomb, n_polarization, _, _,_ = data.shape
# Loop over all combinations
for dope, coulomb, polarization in product(range(n_dope), range(n_coulomb), range(n_polarization)):
    plt.figure()
    plt.imshow(
        data[dope,coulomb, polarization, :,:, polarization].T,
        origin="lower",
        extent=[0, len(emitter_positions), omega[0], omega[-1]],
        aspect=6
    )
    title = f"Tri{size}_d{dope}_U{coulomb}, P_{index[polarization]}, E_{index[polarization]}"
    plt.title(title)
    plt.xlabel("edge to bulk")
    plt.ylabel("energy [eV]")
    plt.colorbar()

    # Save figure
    filename = f"Tri{size}_d{dope}_U{coulomb}_P{index[polarization]}_E{index[polarization]}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()













