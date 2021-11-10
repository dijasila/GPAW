import matplotlib.pyplot as plt
from ase import Atoms
from ase.units import Bohr
from gpaw.utilities.ps2ae import PS2AE
from gpaw import GPAW

plt.rcParams['font.size'] = 12

hli = Atoms('HLi', positions=[[0, 0, 0], [0, 0, 1.6]])
hli.center(vacuum=2.5)
hli.calc = GPAW(txt='hli.txt', mode='fd')
hli.get_potential_energy()

# Transformer:
t = PS2AE(hli.calc, grid_spacing=0.05)

for n in range(2):
    ps = t.get_wave_function(n, ae=False)
    ae = t.get_wave_function(n)
    norm_squared = t.gd.integrate(ae**2) * Bohr**3
    print('Norm:', norm_squared)
    assert abs(norm_squared - 1) < 1e-2
    i = ps.shape[0] // 2
    x = t.gd.coords(2) * Bohr

    # Interpolated PS and AE wfs:
    plt.plot(x, ps[i, i], '--', color=f"C{n}",
             label=r'$\tilde\psi_{}$'.format(n))
    plt.plot(x, ae[i, i], '-', color=f"C{n}",
             label=r'$\psi_{}$'.format(n))

    # Raw PS wfs:
    ps0 = hli.calc.get_pseudo_wave_function(n, pad=True)
    gd = hli.calc.wfs.gd
    i = ps0.shape[0] // 2
    X = gd.coords(2) * Bohr
    plt.plot(X, ps0[i, i], 'o', color=f"C{n}")

plt.hlines(0, xmin=x[0], xmax=x[-1])
plt.xlabel(r'z [$\rm \AA$]')
plt.ylabel(r'wave functions [$\rm \AA^{-3/2}$]')
plt.legend()
plt.tight_layout()
plt.savefig('hli-wfs.png')
hli.calc.write('hli.gpw')
plt.close()
