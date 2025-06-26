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
def get_ImG_spectra(chain,coulomb=0,doping=0,dpm=jnp.array([1,0,0]),DP_X_idx=0,DP_Z_AA=5, dipole_position = None):
   
    if coulomb!=0:
        chain.set_mean_field()
        chain.set_electrons(chain.electrons+doping)

    if dipole_position is None:
        dipole_position=chain.positions[DP_X_idx]+jnp.array([0,0,DP_Z_AA])
    
    time_axis=jnp.arange(0,400,1e-2)

    from granad.potentials import DipolePulse
    DP_func=DipolePulse(dipole_moment=dpm, source_location=dipole_position,
                        omega=4, sigma=1, t0=0.0, kick=True, dt = 1e-2)
    hamiltonian_DP_field=chain.get_hamiltonian()
    hamiltonian_DP_field["dipole_kick"]=DP_func

    import diffrax
    result = chain.master_equation(end_time = 400, dt = 1e-2,
                                hamiltonian = hamiltonian_DP_field,
                                density_matrix=['diag_x'],
                                relaxation_rate = 1/40,
                                solver = "RK45",
                                max_mem_gb=50,
                                grid=1,
                                use_rwa=True,
                                coulomb_strength=coulomb # high accuracy explicit solver to avoid oscillating tails
                               )
    omega,ImG=calculate_ImG_from_rho(rhos_diag_t=result.output[0]-jnp.diagonal(chain.initial_density_matrix_x),
                           time_axis=result.time_axis,
                           dipole_position=dipole_position,
                           induced_field_func=chain.get_induced_field)
   

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
    k=.3 # for different shape and size of the triangle, k may need adjustment.
    extended_point=A-k*(B-A)
    line=jnp.vstack((extended_point, line))
    line2=line+jnp.array([1.42*jnp.sqrt(3)/2,0,0])
    final_line=jnp.vstack((line,line2))
    emitter_positions=find_atoms_along_path(flake.positions,final_line)
    return jnp.unique(emitter_positions,axis=0), line

def localization(flake):
    """Compute eigenstates edge localization"""
    # edges => neighboring unit cells are incomplete => all points that are not inside a "big hexagon" made up of nearest neighbors
    positions, states, energies = flake.positions, flake.eigenvectors, flake.energies 

    distances = jnp.round(jnp.linalg.norm(positions - positions[:, None], axis = -1), 4)
    nnn = jnp.unique(distances)[2]
    mask = (distances == nnn).sum(axis=0) < 12
    # localization => how much eingenstate 
    l = (jnp.abs(states[mask, :])**2).sum(axis = 0) # vectors are normed 

    return l

def Ve_all(flake, emitter_position, upto=20):
    """"" Computes and plots interaction hamiltonian in energy basis"""
    from granad.potentials import DipolePulse
    from matplotlib.ticker import MaxNLocator
    import matplotlib.pyplot as plt
    import jax.numpy as jnp

    index = {0: 'x', 1: 'y', 2: 'z'}
    dpm = {
        0: jnp.array([1, 0, 0]),
        1: jnp.array([0, 1, 0]),
        2: jnp.array([0, 0, 1])
    }

    time_axis = jnp.arange(0, 400, 1e-2)
    interaction_matrices = {}

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    for i in range(3):
        DP_func = DipolePulse(
            dipole_moment=dpm[i],
            source_location=emitter_position,
            omega=4,
            sigma=1,
            t0=0.0,
            kick=True,
            dt=1e-2
        )
        V = DP_func(0, flake.initial_density_matrix, flake.get_args())
        Ve = flake.eigenvectors.conj().T @ V @ flake.eigenvectors
        interaction_matrices[index[i]] = Ve

        ax = axs[i]
        im = ax.imshow(
            Ve.real[flake.homo - upto:flake.homo + upto + 1,
                    flake.homo - upto:flake.homo + upto + 1],
            extent=[-upto, +upto, -upto, +upto]
        )
        ax.set_title(f"$V^{{{index[i]}}}_{{\\mathrm{{int}}}}$") #{index[i]}
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    plt.savefig("Vint.png", dpi=300)

    return interaction_matrices


# =====================================================Execution Code=========================================#
#
# =========Plot the triangular flake and the path ==============#

if __name__ == '__main__':

    size=40
    flake = get_graphene_hubbard(U = 2, size = size)
    emitter_positions_idx,line = get_emitter_positions(flake, points=10000)
    emitter_positions=[dp+jnp.array([0,0,5]) for dp in flake.positions[emitter_positions_idx]]
    emitter_positions=sorted(emitter_positions, key=lambda pos: pos[1])
    plt.scatter(flake.positions[:,0], flake.positions[:,1])
    for i,pos in enumerate(jnp.array(emitter_positions)):
        plt.scatter(flake.positions[emitter_positions_idx,0], flake.positions[emitter_positions_idx,1],label=i, color="red")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    #========== Show energy levels with localization =================#
    
    flake.set_mean_field()
    #flake.show_energies(display=localization(flake))
    
    #========== Plotting interaction matrix ==========================#
    
    V=Ve_all(flake,emitter_positions[0], upto=10)
    
    
    #========= Run simulation For different Couloumb interaction, emitter position and dipole moment =========#
    
    
    cs=[0.0,0.5,1.0]
    o_path, img_path =get_ImG_spectra(flake,
                          coulomb=cs,
                          doping=0,
                          DP_X_idx=None,
                          DP_Z_AA=None,
                          dipole_position=emitter_positions,
                          dpm=[jnp.array([1,0,0]),jnp.array([0,1,0]), jnp.array([0,0,1]) ])
    
    #====== Save Result ==========#
    #structure of img_path variable
    #(coulomb,emitter_positions, dipole_polarization, frequency_axis, component)
    data = jnp.array(img_path)
    print(data.shape)
    omega = jnp.array(o_path)[0,0,0,:]
    jnp.savez(f"Triangle{size}.npz", img=jnp.array(img_path), omega = omega)
    
    #======== Plot the ImG spectra =====================#
    
    import os
    from itertools import product
    
    # Ensure output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    index = {0: 'x', 1: 'y', 2: 'z'}
    
    # Loop through all combinations of parameters (0, 1, 2)
    n_coulomb, _, n_polarization, _, n_component = data.shape

    # Loop over all combinations
    for coulomb, polarization, component in product(range(n_coulomb), range(n_polarization), range(n_component)):
        plt.figure()
        plt.imshow(
            data[coulomb, :, polarization, :, component].T,
            origin="lower",
            extent=[0, len(emitter_positions), omega[0], omega[-1]],
            aspect=6
        )
        title = f"Tri{size}_U{coulomb*2}, P_{index[polarization]}, E_{index[component]}"
        plt.title(title)
        plt.xlabel("edge to bulk")
        plt.ylabel("energy [eV]")
        plt.colorbar()
    
        # Save figure
        filename = f"Tri{size}_U{coulomb}_P{index[polarization]}_E{index[component]}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close()











