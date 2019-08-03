import numpy as np




def correct_density(n_sg, gd, setups, spos_ac):
    # Every rank needs the density
    n1_sg = gd.collect(n_sg, broadcast=True)
    n1_sg[n1_sg < 1e-7] = 1e-8
    
    if not hasattr(setups[0], "calculate_pseudized_atomic_density"):
        return n1_sg
        
    if len(n1_sg) != 1:
        raise NotImplementedError
            
    dens = n1_sg[0].copy()
            
    for a, setup in enumerate(setups):
        spos_ac_indices = list(filter(lambda x: x[1] == setup, enumerate(setups)))
        spos_ac_indices = [x[0] for x in spos_ac_indices]
        spos_ac = spos_ac[spos_ac_indices]
        t = setup.calculate_pseudized_atomic_density(spos_ac)
        t.add(dens)
        
    return np.array([dens])


def get_ni_grid(rank, size, n_sg, pts_per_rank):
    assert rank >= 0 and rank < size
    # Make an interface that allows for testing
    
    # Algo:
    # Define a global grid. We want a grid such that each rank has not too many
    num_pts = pts_per_rank *size
    fulln_i = np.linspace(0, np.max(n_sg), num_pts)
    
    # Split it up evenly
    my_start = rank * pts_per_rank
    my_n_i = fulln_i[my_start:my_start+pts_per_rank]

    # Each ranks needs to know the closest pts outside its own range
    if rank == 0:
        my_lower = 0
        my_upper = fulln_i[min(num_pts - 1, pts_per_rank)]
    elif rank == size - 1:
        my_lower = fulln_i[my_start - 1]
        my_upper = fulln_i[-1]
    else:
        my_lower = fulln_i[my_start - 1]
        my_upper = fulln_i[my_start + pts_per_rank]
    
    return my_n_i, my_lower, my_upper 
