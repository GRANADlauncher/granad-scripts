from itertools import combinations

import matplotlib.pyplot as plt
import jax.numpy as jnp

from granad import *
from granad._numerics import rpa_susceptibility_function

# haldane model has topological phase for Im[t2] > \frac{M}{3 \sqrt{3}} => for 0.3 Im[t_2]_crit ~ 0.06
# sim input : (shape, t1, t2, delta (mass term), name)
def get_haldane_graphene(t1, t2, delta):
    """Constructs a graphene model with
    onsite hopping difference between sublattice A and B, nn hopping, nnn hopping = delta, t1, t2
    """
    return (
        Material("haldane_graphene")
        .lattice_constant(2.46)
        .lattice_basis([
            [1, 0, 0],
            [-0.5, jnp.sqrt(3)/2, 0]
        ])
        .add_orbital_species("pz1", l=1, atom='C')
        .add_orbital_species("pz2", l=1, atom='C')
        .add_orbital(position=(0, 0), tag="sublattice_1", species="pz1")
        .add_orbital(position=(-1/3, -2/3), tag="sublattice_2", species="pz2")
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz2"),
            parameters= [t1],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz1", "pz1"),            
            # a bit overcomplicated
            parameters=[                
                [0, 0, 0, delta], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, t2], 
                [2.46, 0, 0, jnp.conj(t2)],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)]
            ],
        )
        .add_interaction(
            "hamiltonian",
            participants=("pz2", "pz2"),
            parameters=[                
                [0, 0, 0, 0], # onsite                
                # clockwise hoppings
                [-2.46, 0, 0, jnp.conj(t2)], 
                [2.46, 0, 0, t2],
                [2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, t2],
                [2.46*0.5, -2.46*jnp.sqrt(3)/2, 0, jnp.conj(t2)],
                [-2.46*0.5, 2.46*jnp.sqrt(3)/2, 0, t2]
            ],
        )
        .add_interaction(                
            "coulomb",
            participants=("pz1", "pz2"),
            parameters=[8.64],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz1", "pz1"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
        .add_interaction(
            "coulomb",
            participants=("pz2", "pz2"),
            parameters=[16.522, 5.333],
            expression=ohno_potential(0)
        )
    )

### RESPONSE FUNCTIONS ###
def rpa_sus(evs, omegas, occupations, energies, coulomb, relaxation_rate = 1e-1):
    
    def inner(omega):
        mat = delta_occ / (omega + delta_e + 1j*relaxation_rate)
        sus = jnp.einsum('ab, ak, al, bk, bl -> kl', mat, evs, evs.conj(), evs.conj(), evs)
        return sus @ jnp.linalg.inv(one - coulomb @ sus)

    one = jnp.identity(evs.shape[0])
    evs = evs.T
    delta_occ = (occupations[:, None] - occupations) * flake.electrons
    delta_e = energies[:, None] - energies
    
    return jax.lax.map(jax.jit(inner), omegas)

def rpa_response(flake, results_file, cs):
    """computes j-j response from p-p in RPA"""
       
    omegas =  jnp.linspace(0, 15, 150)
    res = []
    
    for c in cs:
        
        sus = rpa_sus(flake.eigenvectors, omegas, flake.initial_density_matrix_e.diagonal(), flake.energies, c*flake.coulomb)
        
        p = flake.positions.T
        
        ref = jnp.einsum('Ii,wij,Jj->IJw', p, sus, p)
        
        res.append(omegas[None, None, :]**2 * ref)
        
    jnp.savez("rpa_" + results_file, cond = res, omegas = omegas, cs = cs)

def ip_response(results_file):
    """runs simulation for IP j-j and p-p total and topological response.     
    saves j-j, pp in "cond_" + results_file, "pol_" + results_file
    """
    
    args_list = [
        (Triangle(30, armchair = True), -2.66, -1j*t2, delta, f"haldane_graphene_{t2}" )
        for (t2, delta) in [(0.0, 0.0), (-0.5, 0.3), (-0.1, 0.3), (0.1, 0.3), (0.5, 0.3)]
        ]

    print("plotting edge states")
    for args in args_list:
        plot_edge_states(args)
        plot_energies(args)
    print("lrt")

    cond, pol = {}, {}
    omegas = jnp.linspace(0, 20, 200)    
    for args in args_list:        
        flake = get_haldane_graphene(*args[1:4]).cut_flake(args[0])
        
        v, p = flake.velocity_operator_e, flake.dipole_operator_e
              
        # compute only topological sector
        trivial = jnp.abs(flake.energies) > 1e-1
        mask = jnp.logical_and(trivial[:, None], trivial)

        cond[args[-1]] = jnp.array([[flake.get_ip_green_function(v[i], v[j], omegas, relaxation_rate = 0.1) for i in range(2)] for j in range(2)])
        pol[args[-1]] = jnp.array([[flake.get_ip_green_function(p[i], p[j], omegas, relaxation_rate = 0.1) for i in range(2)] for j in range(2)])
        
        cond["topological." + args[-1]] = jnp.array([[flake.get_ip_green_function(v[i], v[j], omegas, relaxation_rate = 0.1, mask = mask) for i in range(2)] for j in range(2)])
        pol["topological." + args[-1]] = jnp.array([[flake.get_ip_green_function(p[i], p[j], omegas, relaxation_rate = 0.1, mask = mask) for i in range(2)] for j in range(2)])
        
    cond["omegas"], pol["omegas"] = omegas, omegas
    jnp.savez("cond_" + results_file, **cond)
    jnp.savez("pol_" + results_file, **pol)

# TODO: lookup greens function
def chiral_ldos(results_file, illu, r):
    return

### GROUND STATE ###
def rho(es, vecs, thresh):
    """constructs the open-shell density matrix"""
    d = jnp.where(es <= thresh, 1, 0)
    return jnp.einsum('ij,j,kj->ik', vecs, d, vecs.conj())

def scf_loop(flake, U, mixing, limit, max_steps):
    """performs open-shell scf calculation

    Returns:
        rho_up, rho_dow, ham_eff_up, ham_eff_down
    """
    
    def update(arg):
        """scf update"""
        
        rho_old_up, rho_old_down, step, error = arg

        # H = H_+ + H_-
        ham_eff_up =  ham_0 + U * jnp.diag(jnp.diag(rho_old_down))        
        ham_eff_down =  ham_0 + U * jnp.diag(jnp.diag(rho_old_up))

        # diagonalize
        vals_up, vecs_up = jnp.linalg.eigh(ham_eff_up)
        vals_down, vecs_down = jnp.linalg.eigh(ham_eff_down)    

        # build new density matrices
        thresh = jnp.concatenate([vals_up, vals_down]).sort()[N]
        rho_up = rho(vals_up, vecs_up, thresh) + mixing * rho_old_up
        rho_down = rho(vals_down, vecs_down, thresh) + mixing * rho_old_down

        # update breaks
        error = ( jnp.linalg.norm(rho_up - rho_old_up) +  jnp.linalg.norm(rho_down - rho_old_down) ) / 2

        step = jax.lax.cond(error <= limit, lambda x: step, lambda x: step + 1, step)

        return rho_up, rho_down, step, error
    
    def step(idx, res):
        """single SCF update step"""
        return jax.lax.cond(res[-1] <= limit, lambda x: res, update, res)

    ham_0 = flake.hamiltonian
    
    # GRANAD gives a closed-shell hamiltonian => for hubbard model, we split it into 2 NxN matrices, one for each spin component
    N, _ = ham_0.shape

    # initial guess for the density matrices
    rho_old_up = jnp.zeros_like(ham_0)
    rho_old_down = jnp.zeros_like(ham_0)

    # scf loop
    rho_up, rho_down, steps, error = jax.lax.fori_loop(0, max_steps, step, (rho_old_up, rho_old_down, 0, jnp.inf))
    
    print(f"{steps} / {max_steps}")

    return (rho_up,
            rho_down,
            ham_0 + U * jnp.diag(jnp.diag(rho_down)),
            ham_0 + U * jnp.diag(jnp.diag(rho_up)))
            

def gs_stability(flake, Us):
    """runs Hubbard model mean field simulation for a range of coupling constants"""
    
    res = []
    for U in Us:
        res.append(scf_loop(flake, U, 0.0, 1e-10, 100))

    jnp.savez("scf.npz",
              res = res,
              Us = Us)

def localization(positions, states, energies, uniform = False):
    """Compute eigenstates edge localization"""
    # edges => neighboring unit cells are incomplete => all points that are not inside a "big hexagon" made up of nearest neighbors 
    distances = jnp.round(jnp.linalg.norm(positions - positions[:, None], axis = -1), 4)
    nnn = jnp.unique(distances)[2]
    mask = (distances == nnn).sum(axis=0) < 6

    # localization => how much eingenstate 
    l = (jnp.abs(states[mask, :]).sum(axis = 0) / jnp.abs(states).sum(axis = 0))**2

    if uniform:
        return l, mask.nonzero()[0].size / mask.size

    return l    
    
def plot_localization(positions, states, energies, name = "localization.pdf"):
    loc = localization(positions, states, energies)
    
    fig, ax = plt.subplots(1, 1)
    plt.colorbar(
        ax.scatter(
            jnp.arange(energies.size),
            energies,
            c=loc,
        ),
        label=r"localization = $\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$",
    )
    ax.set_xlabel("eigenstate number")
    ax.set_ylabel("energy (eV)")    
    plt.savefig(name)
    plt.close()
    
### postprocessing ###
def plot_edge_states(args):
    """plot topological E~0 excitations, if present"""    
    shape, t1, t2, delta, name = args
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    try:
        idx = jnp.argwhere(jnp.abs(flake.energies) < 1e-1)[0].item()
        flake.show_2d(display = flake.eigenvectors[:, idx], scale = True, name = name + ".pdf")
    except IndexError:
        print(args, " has no edge states")
    
def plot_energies(args):
    """plots energy landscape of haldane graphene flake specified by args"""    
    shape, t1, t2, delta, name = args
    flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    flake.show_energies(name = name + "_energies.pdf")

def plot_static(args):
    """wrapper around plot_energies and plot_edge_states"""
    for args in args_list:
        plot_edge_states(args)
        plot_energies(args)

def to_helicity(mat):
    """converts mat to helicity basis"""
    trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
    trafo_inv = jnp.linalg.inv(trafo)
    return jnp.einsum('ij,jmk,ml->ilk', trafo, mat, trafo_inv)

def plot_response_functions(results_file):
    """plots j-j response directly and as obtained from p-p response"""    
    with jnp.load("cond_" + results_file) as data:
        cond = dict(data)
        cond_omegas = cond.pop("omegas")        
    with jnp.load("pol_" + results_file) as data:
        pol = dict(data)
        pol_omegas = pol.pop("omegas")
        
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    keys = cond.keys()
    for k in keys:
        if 'topo' in k:
            continue
        for i in range(2):
            for j in range(2):
                axs[i, j].plot(cond_omegas, cond[k][i, j].imag, label='cond_' + k)
                axs[i, j].plot(pol_omegas, pol_omegas**2 * pol[k][i, j].imag, '--', label='pol_' + k)
                axs[i, j].set_title(f'i,j = {i,j}')
                
    axs[0, 0].legend(loc="upper left")
    plt.savefig("cond_pol_comparison_imag.pdf")
    plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    keys = cond.keys()
    for k in keys:
        if 'topo' in k:
            continue
        for i in range(2):
            for j in range(2):
                axs[i, j].plot(cond_omegas, cond[k][i, j].real, label='cond_' + k)
                axs[i, j].plot(pol_omegas, pol_omegas**2 * pol[k][i, j].real, '--', label='pol_' + k)
                axs[i, j].set_title(f'i,j = {i,j}')
                
    axs[0, 0].legend(loc="upper left")
    plt.savefig("cond_pol_comparison_real.pdf")
    plt.close()

    
def load_data(results_file, keys):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data.pop("omegas")        
    return omegas, data, data.keys() if keys is None else keys
    
def plot_excess_chirality(results_file, keys = None):
    """plots excess chirality of the total response"""
    
    omegas, data, keys = load_data(results_file, keys)
    
    for i, key in enumerate(keys):
        mat = to_helicity(data[key])
        mat_real, mat_imag = mat.real, mat.imag

        if 'topological' in key:
            continue
        
        excess = jnp.abs(mat_imag[0,0]) / jnp.abs(mat_imag[1, 1])
        excess /= excess.max()
        plt.plot(omegas, excess, label = key.split("_")[-1])
        
    plt.legend(loc = "upper left")
    plt.savefig("excess_chirality.pdf")
    plt.close()


def plot_chirality_difference(results_file, keys = None):
    """plots excess chirality of the total response"""
    omegas, data, keys = load_data(results_file, keys)
    
    # Loop through each key to plot the data
    for i, key in enumerate(keys):
        mat = to_helicity(data[key])
        mat_real, mat_imag = mat.real, mat.imag

        if 'topological' in key:
            continue

        idx = 10
        diff = (mat_imag[1, 1] - mat_imag[0, 0]) / (mat_imag[1, 1] + mat_imag[0, 0])
        plt.plot(omegas[idx:], diff[idx:], label=key.split("_")[-1])

    # Adding titles and labels to make it clear
    plt.xlabel(r'$\omega$ (eV)$')
    plt.ylabel(r'$\delta_{+-}$')
    plt.legend(loc="upper left")

    # Adjusting layout and saving
    plt.tight_layout()
    plt.savefig("chirality_difference.pdf")
    plt.close()    
    
def plot_chirality_components(results_file, keys = None):
    """plots excess chirality of the total response"""
    omegas, data, keys = load_data(results_file, keys)
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # Adjust figure size if needed

    # Loop through each key to plot the data
    for i, key in enumerate(keys):
        mat = to_helicity(data[key])
        mat_real, mat_imag = mat.real, mat.imag

        if 'topological' in key:
            continue

        axs[0].plot(omegas, mat_imag[0, 0], label=key.split("_")[-1])
        axs[1].plot(omegas, mat_imag[1, 1])

    # Adding titles and labels to make it clear
    axs[0].set_title(r'$\sigma_{++}$')
    axs[0].set_ylabel(r'$\sigma$ (a.u.)')
    axs[0].legend(loc="upper left")

    axs[1].set_title(r'$\sigma_{--}$')
    axs[1].set_ylabel(r'$\sigma$ (a.u.)')
    axs[1].set_xlabel(r'$\omega$ (eV)')

    # Adjusting layout and saving
    plt.tight_layout()
    plt.savefig("chirality_components.pdf")
    plt.close()
    
def plot_topological_total(results_file, keys = None):
    """plots topological and total response on same x-axis"""
    
    omegas, data, keys = load_data(results_file, keys)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    for i, key in enumerate(keys):
        mat = to_helicity(data[key])
        mat_real, mat_imag = mat.real, mat.imag

        if '-' in key or not ('0.5' in key or '0.0' in key):
            continue
        
        delta = mat_imag[1, 1] - mat_imag[0, 0]
        
        if 'topological' in key:
            ax2.plot(omegas, delta, ls = '--', label = key.split("_")[-1])
        else:
            ax1.plot(omegas, delta, label = key.split("_")[-1])
            
    ax1.legend(loc = "upper left")
    plt.savefig("chirality_topological_total.pdf")
    plt.close()

def plot_rpa_response(results_file):
    with jnp.load(results_file) as data:
        data = dict(data)
        omegas = data["omegas"]
        cond = data["cond"][:, :2, :2, :]
        cs = data["cs"]
        
    def to_helicity(mat):
        """converts mat to helicity basis"""
        trafo = 1 / jnp.sqrt(2) * jnp.array([ [1, 1j], [1, -1j] ])
        trafo_inv = jnp.linalg.inv(trafo)
        return jnp.einsum('ij,jmk,ml->ilk', trafo_inv, mat, trafo)
        
    for i, coulomb_strength in enumerate(cs):
        c = to_helicity(cond[i])
        delta = (c.imag[1, 1] - c.imag[0, 0]) / (c.imag[1, 1] + c.imag[0, 0])
        plt.plot(omegas, delta, label = fr'$\lambda$ = {coulomb_strength}')

    plt.xlabel(r'$\omega$ (eV)$')
    plt.ylabel(r'$\delta_{+-}$')
    plt.legend(loc = "upper left")
    plt.savefig("rpa.pdf")
    plt.close()

def plot_stability(flake):
    """loads scf results, plots energy landscape, current directionality"""
    
    with jnp.load("scf.npz") as data:
        data = dict(data)
        res = data["res"]
        Us = data["Us"]

    # energy landscapes
    for i, U in enumerate(Us):
        rho_up, rho_down, h_up, h_down = res[i]        
        v_up, vecs_up = jnp.linalg.eigh(h_up)
        v_down, vecs_down = jnp.linalg.eigh(h_down)        
        c_up = jnp.diagonal(vecs_up.conj().T @ rho_up @ vecs_up)
        c_down = jnp.diagonal(vecs_down.conj().T @ rho_down @ vecs_down)
        
        plot_localization(flake.positions, vecs_up, v_up, f"{U}_up.pdf")
        plot_localization(flake.positions, vecs_down, v_down, f"{U}_down.pdf")

    # current
    l = []
    for i, U in enumerate(Us):
        rho_up, rho_down, h_up, h_down = res[i]        
        v_up, vecs_up = jnp.linalg.eigh(h_up)
        
        l_up = localization(flake.positions, vecs_up, v_up).max()
        l_down = localization(flake.positions, vecs_down, v_down).max()

        l.append( (l_up + l_down) / 2)
            
    plt.plot(Us, l, '.')
    plt.xlabel('U (eV)')
    plt.ylabel(r"$\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$")
    plt.savefig("scf_localization.pdf")
    plt.close()      

def show_2d(orbs, ax, show_tags=None, show_index=False, display = None, scale = False, cmap = None, circle_scale : float = 2*1e2, title = None):
    # decider whether to take abs val and normalize 
    def scale_vals( vals ):
        return jnp.abs(vals) / jnp.abs(vals).max() if scale else vals
    
    # Determine which tags to display
    if show_tags is None:
        show_tags = {orb.tag for orb in orbs}
    else:
        show_tags = set(show_tags)

    # Prepare data structures for plotting
    tags_to_pos, tags_to_idxs = defaultdict(list), defaultdict(list)
    for orb in orbs:
        if orb.tag in show_tags:
            tags_to_pos[orb.tag].append(orb.position)
            tags_to_idxs[orb.tag].append(orbs.index(orb))

    cmap = plt.cm.bwr if cmap is None else cmap
    colors = scale_vals(display)
    scatter = ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], c=colors, edgecolor='black', cmap=cmap, s = circle_scale*jnp.abs(display) )
    # ax.scatter([orb.position[0] for orb in orbs], [orb.position[1] for orb in orbs], color='black', s=1, marker='o')
    # cbar = fig.colorbar(scatter, ax=ax)

    # Optionally annotate points with their indexes
    if show_index:
        for orb in [orb for orb in orbs if orb.tag in show_tags]:
            pos = orb.position
            idx = orbs.index(orb)
            ax.annotate(str(idx), (pos[0], pos[1]), textcoords="offset points", xytext=(0,10), ha='center')

    # Finalize plot settings
    # plt.title('Orbital positions in the xy-plane' if title is None else title)
    ax.grid(True)
    ax.axis('equal')
    return scatter

def plot_edge_states_energy_landscape():
    setups = [
        (shape, -2.66, -1j, 0.3, f"haldane_graphene" )
        for shape in [Triangle(18, armchair = False), Rectangle(10, 10), Hexagon(20, armchair = True)]
        ]

    fig, axs = plt.subplots(3,2)
    axs_flat = list(axs.flat)
    
    for i, s in enumerate(setups):        
        flake = get_haldane_graphene(*s[1:4]).cut_flake(s[0])
        
        loc = localization(flake.positions, flake.eigenvectors, flake.energies)

        sc1 = show_2d(flake, axs_flat[2*i], display = jnp.abs(flake.eigenvectors[:, loc.argmax()]) )
        axs_flat[2*i].set_xlabel('X')
        axs_flat[2*i].set_ylabel('Y')

        cb1 = fig.colorbar(sc1, ax=axs_flat[2*i])
        cb1.set_label(r'$|\psi|^2$')


        sc2 = axs_flat[2*i + 1].scatter(
            jnp.arange(flake.energies.size),
            flake.energies,
            c=loc)
        axs_flat[2*i+1].set_xlabel('# eigenstate')
        axs_flat[2*i+1].set_ylabel('E (eV)')

        cb2 = fig.colorbar(sc2, ax=axs_flat[2*i + 1])
        cb2.set_label(r"$\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$")  


    plt.tight_layout()
    plt.savefig("edge_states_energy_landscape.pdf")
    plt.close()

def plot_localization_varying_hopping():
    setups = [
        (shape, -2.66, -1j, 0.3, f"haldane_graphene" )
        for shape in [Triangle(18, armchair = False), Rectangle(10, 10), Hexagon(20, armchair = True)]
        ]

    fig, axs = plt.subplots(3,1)
    axs_flat = list(axs.flat)

    def loc(shape, n, nn, delta):
        flake = get_haldane_graphene(n, nn, delta).cut_flake(shape)
        l = localization(flake.positions, flake.eigenvectors, flake.energies)
        return jnp.max(l)

    nns = jnp.linspace(0, 0.5, 10)
    for i, s in enumerate(setups):        
        locs = [loc(s[0], s[1], 1j*nn, s[3]) for nn in nns]
        axs[i].plot(nns, locs)

        flake = get_haldane_graphene(*s[1:4]).cut_flake(s[0])
        _, v = localization(flake.positions, flake.eigenvectors, flake.energies, uniform = True)
        
        axs[i].axhline(y=v, ls='--', label = 'uniform')
        axs[i].axvline(x=0.03, c = 'r', label = 'bulk transition')

        axs[i].set_xlabel(r'$t_2$')
        axs[i].set_ylabel(r'$\dfrac{|\psi_{\text{edge}}|^2}{|\psi|^2}$')

    axs[i].legend()
    plt.tight_layout()
    plt.savefig("localization_varying_hopping.pdf")
    plt.close()
        
if __name__ == '__main__':
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # plot edge states vs localization-annotated energy landscape of a few structures    
    # plot_edge_states_energy_landscape()
    
    # plot edge state localization-annotated energy landscape for varying t2
    plot_localization_varying_hopping()
    
    # plot localization depending on Hubbard-U
    # t1, t2, delta, shape = -2.66, -1j, 0.3, Triangle(30)
    # flake = get_haldane_graphene(t1, t2, delta).cut_flake(shape)
    # gs_stability(flake, [0, 0.1, 0.2, 1., 1.5, 2., 2.5, 3.])
    # plot_stability(flake)

    # compute IP response
    f = "lrt.npz"
    ip_response(f)
    plot_chirality_difference("cond_" + f)
    plot_response_functions(f)
    
    # plot_excess_chirality("cond_" + f)
    # plot_chirality_components("cond_" + f)
    # plot_topological_total("cond_" + f)

    # check response stability with RPA
    # flake = get_haldane_graphene(-2.66, -0.5j, 0.3).cut_flake(Triangle(30))
    # rpa_response(flake, "triangle", [0, 0.01, 0.1, 0.5, 0.7, 1.0])
    # plot_rpa_response("rpa_triangle.npz")

    # TODO: compute chiral LDOS of magnetic dipole antenna
