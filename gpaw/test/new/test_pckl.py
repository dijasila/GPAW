import pickle

from gpaw.core import UGDesc, PWDesc
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.setup import Setups
import numpy as np
from gpaw.core.atom_arrays import AtomDistribution


def test_pckl(in_tmp_dir):
    g1 = UGDesc(cell=[1, 2, 3], size=(5, 10, 15)).zeros(2)
    p1 = PWDesc(cell=g1.desc.cell, ecut=5.0).zeros(2)
    g2, p2 = pickle.loads(pickle.dumps((g1, p1)))
    print(g2, p2)
    wfs1 = PWFDWaveFunctions(
        g1,
        spin=0,
        q=0,
        k=0,
        setups=Setups([1], {}, {}, 'LDA'),
        fracpos_ac=np.zeros((1, 3)),
        atomdist=AtomDistribution([0]))
    wfs2 = pickle.loads(pickle.dumps(wfs1))


if __name__ == '__main__':
    test_pckl(42)
