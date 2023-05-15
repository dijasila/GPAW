import pytest
import numpy as np
from gpaw.test.conftest import response_band_cutoff
from gpaw import GPAW
import gpaw.mpi as mpi


@pytest.mark.response
def test_response_band_cutoff(in_tmp_dir, gpw_files):

    for gs, nbands in response_band_cutoff.items():
        calc = GPAW(gpw_files[gs],
                    communicator=mpi.serial_comm)
        nconv = calc.parameters.convergence['bands']
        print(gs)
        assert nbands < nconv
        possible_cutoffs = get_nbands_cutoff_list(calc.wfs,
                                                  nconv)
        assert nbands in possible_cutoffs

        
def get_nbands_cutoff_list(wfs, nconv, atol=1e-4):
    """ Possible cutoffs for response calc
    Returns the set all allowed band cutoffs in a response calculation.
    Assures that there are no  degeneracies at the edge of the cutoff
    """
    nibzk = wfs.kd.nibzkpts
    allset = set(range(nconv + 1))
    # Loop over spins and k-points
    for s in range(wfs.nspins):
        for ik in range(nibzk):
            kpt = wfs.kpt_qs[ik][s]
            eps_n = kpt.eps_n
            # check degenerate eigenvalues
            cutlist = np.isclose(eps_n[:nconv - 1], eps_n[1:nconv])
            cutlist = np.argwhere(~cutlist)
            # cutoff is allowed index + 1
            cutlist += 1
            thisset = set(cutlist.flatten())
            # find minimum cutoff that works for all k
            allset = thisset & allset
            
    return allset
    
