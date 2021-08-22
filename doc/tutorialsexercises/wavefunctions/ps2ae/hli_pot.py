import matplotlib.pyplot as plt
from ase.units import Bohr
from gpaw.utilities.ps2ae import PS2AE
from gpaw import GPAW

calc = GPAW('hli.gpw', txt=None)

# Avarage PS potentials:
vh, vli = calc.get_atomic_electrostatic_potentials()
zh, zli = calc.atoms.positions[:, 2]
rh, rli = [max(setup.rcut_j) * Bohr for setup in calc.setups]
plt.plot([zh - rh, zh + rh], [vh, vh], label=r'$\tilde v$(H)')
plt.plot([zli - rli, zli + rli], [vli, vli], label=r'$\tilde v$(Li)')

# Transformer:
t = PS2AE(calc, grid_spacing=0.05)

# Interpolated PS and AE potentials:
ps = t.get_electrostatic_potential(ae=False)
ae = t.get_electrostatic_potential()
i = ps.shape[0] // 2
z = t.gd.coords(2) * Bohr

plt.plot(z, ps[i, i], '--', label=r'$\tilde v$')
plt.plot(z, ae[i, i], '-', label=r'$v$')

# Raw PS potential:
ps0 = calc.get_electrostatic_potential()
gd = calc.hamiltonian.finegd
i = ps0.shape[0] // 2
Z = gd.coords(2) * Bohr
plt.plot(Z, ps0[i, i], 'o')

plt.plot(z, 0 * z, 'k')
plt.xlabel('z [Ang]')
plt.ylabel('potential [eV]')
plt.ylim(bottom=-100, top=10)
plt.legend()
plt.savefig('hli-pot.png')
