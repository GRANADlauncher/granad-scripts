## tests the functions defined in sim.py ##

from sim import *

# shapes:
# stationary density => NxN
# rhos => TxNxN
# propagator => 3 x N x N
def test_field_at():
    class Dummy:
        def __init__(self, r, e):
            self.stationary_density_matrix = r
            self.electrons = e
            
    source_positions = jnp.arange(3*10).reshape(10, 3)
    target_position = jnp.array([0.1, 0.1, 0.1])
    f = field_at(source_positions, target_position)
    stationary_density_matrix = jnp.arange(10*10).reshape(10,10)    
    rhos = jnp.arange(4*10*10).reshape(4,10,10)
    electrons = 10    
    assert f(rhos, Dummy(stationary_density_matrix, electrons)).shape == (4, 3)    
