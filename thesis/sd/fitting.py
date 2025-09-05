import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

from tbfit import hamiltonian_overlap, UnitCell, show2D, matrix_params

def reciprocal( vecs ):
    """Returns reciprocal lattice as 3x3 array rec for vectors contained in 3x3 array vecs. The i-th vector must be given by vecs[i,:]. The i-th 
    reciprocal lattice vector is then given by rec[i,:]
    """
    return 2*jnp.pi/(vecs[0,:] @ jnp.cross(vecs[1,:], vecs[2,:])) * jnp.array( [
        jnp.cross(vecs[1,:], vecs[2,:]),
        jnp.cross(vecs[2,:], vecs[0,:]),
        jnp.cross(vecs[0,:], vecs[1,:]) ]
                     )

def parse_band_structure_qe( bands_file, E_f, band_selection ):
    """Parses a quantum espresso band structure file. Shifts energies by E_f and returns only bands matching array band_selection.
    """
    with open( bands_file ) as f:
        ks, bands = [], []
        for i, line in enumerate(f):
            if i == 0:
                continue
            elif i % 2 == 1:
                ks.append( [float(x) for x in filter(lambda x: x, line.replace('\n','').split(' ') )] )
            elif i % 2 == 0:
                bands.append( [float(x) for x in filter(lambda x: x, line.replace('\n','').split(' ') )] )
    return jnp.array(bands)[:, band_selection] +  E_f, 2*jnp.pi*jnp.array(ks)

def loss( hop, over, h_idxs, o_idxs, basis, positions, ks, data ):
    """ Returns the squared error  between tb model defined by hopping rates hop, overlap elements over, basis, positions and data over k points given in ks. Very unelegantly, parameters are forced to be equal via the two index arrays h_idxs, o_idxs
    """
    H,S = hamiltonian_overlap(hop[h_idxs], over[o_idxs], basis, positions, ks.T)
    band = jnp.linalg.eigh(jnp.linalg.inv(S.T) @ H.T)[0][:,0]
    return ( (band - data)**2).sum()
    
if __name__ == '__main__':
    # NOTE that the k-vectors in bands.dat are defined in cartesian coordinates, so we can use them directly
    data, ks  = parse_band_structure_qe( 'data/graphene.bands.dat', 2.3462, jnp.array([1,2,3]))
    
    # TODO could be put into parsing function
    lb = jnp.array([ [1,0,0], [-1/2,jnp.sqrt(3)/2,0] ])
    rec = reciprocal(
    jnp.array( [ [1.002201202,-0.000000000,0.000000000],
                [-0.501100601, 0.867931701, 0.000000000],
                 [0.000000000, 0.000000000, 8.130081301]
                ] )
    )
    A, B = rec[:2,:]

    # example on how one might define a set of k-vectors in QE
    # 0.0000000000 0.000000000 0.000000000 30 ! G
    # -0.333333333 0.666666667 0.000000000 30 ! K
    # 0.000000000 0.500000000 0.000000000 30 ! M
    # 0.0000000000 0.000000000 0.000000000 30 ! G
    
    # the analogous way to compute the corresponding tbfit k vectors would be
    # ks = jnp.concatenate(
    #     ( jnp.expand_dims(jnp.linspace(0,-1/3,100),1)*A + jnp.expand_dims(jnp.linspace(0,2/3,100),1)*B, # G -> K
    #       jnp.expand_dims(jnp.linspace(-1/3,0,100),1)*A + jnp.expand_dims(jnp.linspace(2/3,0.5,100),1)*B, # K -> M
    #       jnp.expand_dims(jnp.linspace(0.5,0,30),1)*B # M -> G          
    #     ))

    # informed first guess based on 10.1103/PhysRevB.66.035412
    uc = UnitCell( 'graphene',
                lb,
                   {
                       'A' : (jnp.array([0,0,0]),
                               {'A' : [-0.36,-0.12], 'B' : [-2.78,-0.068] },
                               {'A' : [1,0.001], 'B' : [0.106,0.003] }
                              ),
                       'B' : ( 1/3*lb[0,:] + 2/3*lb[1,:],
                               {'B' : [-0.36,-0.12] },
                               { 'B' : [1, 0.001] }
                              ),
                   }
                  )


    hoppings_dummy, overlap_dummy, basis, positions = matrix_params( uc )
    hoppings, h_idxs = jnp.unique(hoppings_dummy, return_inverse = True)
    overlap, o_idxs = jnp.unique(overlap_dummy, return_inverse = True)

    # reorder data such that we get correct pi band
    data = jnp.concatenate( (data[:12,0], data[12:14,1], data[14:74,2], data[74:78,1], data[78:,0] ) )

    # inspect data
    # plt.plot( jnp.arange(ks.shape[0]), data, '.' )    
    # plt.show()
    # pdb.set_trace()
    
    ## gradient descent
    rate = 0.00005
    print("Begin fitting")
    for i in range(0):
        if i == 0:
            print( f'Loss after {i}th iteration: {loss(hoppings, overlap, h_idxs, o_idxs, basis, positions, ks, data)}' )
        h_grad, o_grad = jax.grad(loss, argnums = (0,1))(hoppings, overlap, h_idxs, o_idxs, basis, positions, ks, data)
        hoppings -= rate * h_grad
        overlap -=  rate * o_grad
        if i % 100 == 0:
            print( f'Loss after {i}th iteration: {loss(hoppings, overlap, h_idxs, o_idxs, basis, positions, ks, data)}' )
    print(f'Final values {hoppings} {overlap}')
    

    # Compute tight-binding band structure
    H, S = hamiltonian_overlap(hoppings[h_idxs], overlap[o_idxs], basis, positions, ks.T)
    band = jnp.linalg.eigh(jnp.linalg.inv(S.T) @ H.T)[0][:, 0]

    # Define x-axis values and k-point labels
    x_vals = jnp.arange(ks.shape[0])
    kpoint_indices = [0, 30, 60, 91]
    kpoint_labels = [r'$\Gamma$', 'K', 'M', 'K']

    # Plot with fivethirtyeight style
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x_vals, band, label='Tight-Binding Fit', linewidth=2)
    ax.plot(x_vals, data, '.', label='DFT Data', markersize=6)

    # Set custom ticks for high-symmetry points
    ax.set_xticks(kpoint_indices)
    ax.set_xticklabels(kpoint_labels, fontsize=11)

    # Axis labels and title
    ax.set_xlabel("k-path", fontsize=11)
    ax.set_ylabel("Energy (eV)", fontsize=11)
    ax.set_title("Tight-Binding vs DFT Band Structure", fontsize=13)

    # Legend and grid
    ax.legend(frameon=False, fontsize='small', loc='best')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Vertical lines at high-symmetry points (optional)
    for kpt in kpoint_indices:
        ax.axvline(x=kpt, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)

    # Tidy layout and save
    plt.tight_layout()
    plt.savefig("graphene_tb_fit.pdf", dpi=300, bbox_inches='tight')
    plt.show()

    # saving and loading data
    jnp.savez("params.npz", hoppings = hoppings[h_idxs], overlap = overlap[o_idxs], basis = basis, positions = positions, ks = ks, dft_data = data )    
    data = jnp.load('params.npz')
    hoppings, overlap, basis, positions, ks, dft_data = data["hoppings"], data["overlap"], data["basis"], data["positions"], data["ks"], data["dft_data"]
