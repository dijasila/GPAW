# creates: hli-pot.png
import matplotlib.pyplot as plt
from ase.units import Bohr
from gpaw.utilities.ps2ae import PS2AE
from gpaw import GPAW

calc = GPAW('hli.gpw', txt=None)

# Transformer:
t = PS2AE(calc, h=0.05)

# Interpolated PS and AE potentials:
ps = t.get_electrostatic_potential(ae=False)
ae = t.get_electrostatic_potential()
i = ps.shape[0] // 2
x = t.gd.coords(2) * Bohr

plt.plot(x, ps[i, i], '--', label=r'$\tilde v$')
plt.plot(x, ae[i, i], '-', label=r'$v$')

# Raw PS wfs:
ps0 = calc.get_electrostatic_potential()
gd = calc.hamiltonian.finegd
I = ps0.shape[0] // 2
X = gd.coords(2) * Bohr
plt.plot(X, ps0[I, I], 'o')

plt.plot(x, 0 * x, 'k')
plt.xlabel('z [Ang]')
plt.ylabel('potential [eV]')
plt.ylim(ymin=-100, ymax=10)
plt.legend()
plt.savefig('hli-pot.png')
