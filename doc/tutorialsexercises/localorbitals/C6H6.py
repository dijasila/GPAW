# creates: C6H6_minimal.png, C6H6_extended.png
import numpy as np
from ase.build import molecule
from ase.io import write
from gpaw import GPAW
from gpaw.lcao.local_orbitals import LocalOrbitals
from gpaw.lcao.pwf2 import LCAOwrap
from matplotlib import pyplot as plt
from scipy.linalg import eigvalsh

plt.style.use('bmh')
plt.rcParams['font.size'] = 13
plt.rcParams['lines.linewidth'] = 2


def get_eigvals(model, erng=[-10, 10]):
    """Helper function to get the eigenvalues.

    Parameters
    ----------
    model : Object
        Must have a `get_hamiltonian` and `get_overlap` methods.
    erng : (float,float), optional
        Energy range of min and max eigenvalues.
    """
    global calc
    # Compute eigenvalues from H and S.
    H = model.get_hamiltonian()
    S = model.get_overlap()
    if np.allclose(S, np.eye(S.shape[0]), atol=1e-4):
        eigs = np.linalg.eigvalsh(H)
    else:
        eigs = eigvalsh(H, S)
    lims = calc.get_fermi_level() + erng[0], calc.get_fermi_level() + erng[1]
    mask = (eigs > lims[0]) & (eigs < lims[1])
    return eigs[mask]


def compare_eigvals(lcao_eigs, los_eigs, fname, title):
    """Compare the eigenvalues of the low-energy model with 
    the full LCAO calculation as horizontal lines.
    """
    global calc
    # Get the eigenvalues
    lcao_eigs = get_eigvals(lcao)
    los_eigs = get_eigvals(los)
    plt.figure()
    plt.hlines(lcao_eigs - calc.get_fermi_level(), -1, -0.01, color='tab:blue')
    plt.hlines(los_eigs - calc.get_fermi_level(), 0.01,
               1, linestyles='-.', color='tab:orange')
    plt.hlines(0., -1., 1., linestyle='--', color='black')
    plt.grid(axis='x')
    _ = plt.xticks(labels=['LCAO', 'LOs'], ticks=[-0.5, 0.5])
    plt.title(title)
    plt.ylabel('Energy (eV)')
    plt.savefig(fname, bbox_inches='tight')


# Atoms
benzene = molecule('C6H6', vacuum=10)

# LCAO calculation
calc = GPAW(mode='lcao', xc='PBE', basis='szp(dzp)', txt=None)
calc.atoms = benzene
calc.get_potential_energy()

lcao = LCAOwrap(calc)

# Construct a LocalOrbital Object
los = LocalOrbitals(calc)

# Subdiagonalize carbon atoms and group by energy.
los.subdiagonalize('C', groupby='energy')
print("Groups of LOs\n", los.groups)

# Dump s-type LOs in cube format
w_wG = los.get_orbitals(los.groups[-19.2])
for w, w_G in enumerate(w_wG):
    write(f's-lo-{w}.cube', calc.atoms, data=w_G)

# Take minimal model
los.take_model(minimal=True)
print("Number of LOs in low-energy model:", len(los.model))
print("Size of matrices", los.get_hamiltonian().shape)

# Assert model indeed conicides with pz-type LOs
assert los.indices == los.groups[-6.8]

# Dump model to cube files
w_wG = los.get_orbitals(los.indices)
for w, w_G in enumerate(w_wG):
    write(f'pz-lo-{w}.cube', calc.atoms, data=w_G)

# Compare eigenvalues with LCAO
compare_eigvals(lcao, los, "C6H6_minimal.png", "minimal=True")

# Extend with groups of LOs that overlap with the minimal model
los.take_model(minimal=False)
print("Number of LOs in low-energy model:", len(los.model))
print("Size of matrices", los.get_hamiltonian().shape)

# Assert model is extended with other groups.
assert los.indices == (los.groups[-6.8] + los.groups[20.1] + los.groups[21.5])

# Compare eigenvalues with LCAO
compare_eigvals(lcao, los, "C6H6_extended.png", "minimal=False")
