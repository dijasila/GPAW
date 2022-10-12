import numpy as np
from ase.build import graphene_nanoribbon
from ase.io import write
from gpaw import GPAW
from gpaw.lcao.local_orbitals import LocalOrbitals
from gpaw.lcao.tightbinding import TightBinding
from gpaw.lcao.tools import get_bfi
from matplotlib import pyplot as plt
from scipy.linalg import eigvalsh

plt.style.use('bmh')
plt.rcParams['font.size'] = 13
plt.rcParams['lines.linewidth'] = 2


def compare_bandstructure(lcao, los, fname, title):
    """Compare the bandstructure of the of the low-energy model with 
    the full LCAO calculation as horizontal lines"""
    # Define a bandpath
    global calc
    bp = calc.atoms.cell.bandpath('GX', npoints=60)
    xcoords, special_xcoords, labels = bp.get_linear_kpoint_axis()
    # Get the bandstructure
    lcao_eps = lcao.band_structure(bp.kpts, blochstates=False)
    los_eps = los.band_structure(bp.kpts, blochstates=False)
    # Plot
    plt.figure()
    lines = plt.plot(xcoords, lcao_eps - calc.get_fermi_level(), 'tab:blue')
    lines[0].set_label('LCAO')
    lines = plt.plot(xcoords, los_eps - calc.get_fermi_level(),
                     '-.', color='tab:orange')
    lines[0].set_label('LOs')
    plt.legend()
    plt.hlines(
        0., xmin=special_xcoords[0], xmax=special_xcoords[1], color='k', linestyle='--')
    plt.ylim(-10., 10.)
    plt.title(title)
    plt.xticks(special_xcoords, labels)
    plt.ylabel('Energy (eV)')
    plt.savefig(fname, bbox_inches='tight')


# Graphene nanoribbon
gnr = graphene_nanoribbon(2, 1, type='zigzag', saturated=True,
                          C_H=1.1, C_C=1.4, vacuum=5.0)

# LCAO calculation
try:
    calc = GPAW('C4H2.gpw')
    if calc.wfs.S_qMM is None:
        calc.wfs.set_positions(calc.spos_ac)
    calc.atoms = gnr
except:
    calc = GPAW(mode='lcao', xc='PBE', basis='szp(dzp)', txt=None, kpts={'size': (1, 1, 11), 'gamma': True},
            symmetry={'point_group': False, 'time_reversal': True})
    calc.atoms = gnr
    calc.get_potential_energy()
    calc.write('C4H2.gpw', mode='all')

# LCAO Tight Binding model
tb = TightBinding(calc.atoms, calc)

# Construct a LocalOrbital Object
los = LocalOrbitals(calc)

# Subdiagonalize carbon atoms and group by symmetry and energy.
los.subdiagonalize('C', groupby='symmetry')
print("Groups of LOs\n", los.groups)

# Take minimal model
los.take_model(minimal=True)

# Compare the bandstructure of the effective model and compare it with LCAO
compare_bandstructure(tb, los, "C4H2_minimal.png", "minimal=True")

# Extend with groups of LOs that overlap with the minimal model
los.take_model(minimal=False)

# Compare the bandstructure of the effective model and compare it with LCAO
compare_bandstructure(tb, los, "C4H2_extended.png", "minimal=False")
