# %%
# teacher
import ase.visualize as viz
viz.view = lambda atoms, repeat=None: None

# %%
"""
# Non-collinear magnetism - VI$_2$

Having looked at the ferromagnetic compound CrI$_3$, we now move on to a bit more complicated material. We will stay in the framework of 2D materials, but now move on to anti-ferromagnetic exchange coupling. We will still have the Hamiltonian

$$H = -\frac{1}{2}\sum_{ij}J_{ij}\mathbf{S}_i\cdot \mathbf{S}_j+A\sum_i(S_i^z)^2$$

in mind, but now with $J<0$. Go to the 2D database at https://cmrdb.fysik.dtu.dk/?project=c2db and search for VI$_2$ in the CdI$_2$ prototype. Click on the *ferromagnetic* structure and download the .xyz file. Since we will need to do LDA calculations later on, we start by relaxing the structure with the LDA functional. We will be interested in the anti-ferromagnetic state later, but perform the relaxation in the ferromagnetic state, which has a smaller unit cell. Fill in the missing pieces and run the cell below. V has the electronic configuration [Ar]3d$^3$4s$^2$, which can be used to guess the initial magnetic moments. The calculation takes about 17 minutes.

"""

# %%
from ase.io import read
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from gpaw import GPAW, PW

S = 3 / 2  # student: S = ???
m = S * 2
layer = read('VI2.xyz')
layer.set_initial_magnetic_moments([m, 0, 0])
view(layer)

calc = GPAW(mode=PW(400), xc='LDA', kpts=(4, 4, 1))
layer.calc = calc

uf = UnitCellFilter(layer, mask=[1, 1, 0, 0, 0, 1])
opt = BFGS(uf)
opt.run(fmax=0.1)

calc.set(symmetry='off')
calc.get_potential_energy()

calc.write('VI2_relaxed.gpw')

# %%
"""
### Magnetic anisotropy
Note that we switch off symmetry in the end of the script and do a last calculations with all $k$-points in the Brillouin zone. This is because spinors transform in a non-trivial way and the spin-orbit coupling can be obtained from the irreducible $k$-points without transforming the wavefunctions at symmetry related $k$-points. Evaluate the magnetic anisotropy in the cell below in a manner similar to the case of CrI$_3$. Is the easy-axis in plane or out of plane? Do you expect to find a finite critical temperature based on this?
"""

# %%
from gpaw.spinorbit import get_anisotropy
# teacher:
from math import pi

calc = GPAW('VI2_relaxed.gpw', txt=None)
e_x = get_anisotropy(calc, theta=pi / 2, phi=0)
e_y = get_anisotropy(calc, theta=pi / 2, phi=pi / 2)
e_z = get_anisotropy(calc, theta=0, phi=0)
de_zx = e_z - e_x
de_zy = e_z - e_y
print('dE_zx = %1.3f meV' %  (de_zx * 1000))
print('dE_zy = %1.3f meV' %  (de_zy * 1000))
A = de_zx

# %%
"""
### Anti-ferromagnetic state

We can do an anti-ferromagnetic calculation by repeating the structure we just relaxed and setting the initial magnetic moments accordingly. Fill in the missing values in the cell below and run it. The calculation takes about 7 minutes.

"""

# %%
layer_afm = layer.repeat((2, 1, 1))
layer_afm.set_initial_magnetic_moments([m, 0, 0, -m, 0, 0])
view(layer_afm)

calc = GPAW(mode=PW(400),
            xc='LDA',
            kpts=(2, 4, 1))  # student: kpts=???))
layer_afm.calc = calc
layer_afm.get_potential_energy()
calc.write('V2I4_afm.gpw')

...
# teacher:
layer_fm = layer.repeat((2, 1, 1))
layer_fm.set_initial_magnetic_moments([m, 0, 0, m, 0, 0])
calc = GPAW(mode=PW(400), xc='LDA', kpts=(2, 4, 1))
layer_fm.calc = calc
layer_fm.get_potential_energy()
calc.write('V2I4_fm.gpw')

# %%
"""
### Calculating J
Is the total energy of anti-ferromagnetic state smaller than the ferromagnetic one? It should be. But since we are running with rather low parameters for $k$-point sampling and plane wave cutoff, we better perform a ferromagnetic calculation with exactly the same parameters to make sure. Run the cell above with in a ferromagnetic spin state and compare the resulting energies.

The anti-ferromagnetic state we constructed appears to have a lower energy, but can we really be sure that this is the magnetic ground state? The exchange coupling must be negative, which indicates that spins prefer to be anti-aligned. Draw the magnetic configuration of the lattice on a piece of paper and convince yourself that all spins cannot be anti-aligned on the hexagonal lattice. The anti-ferromagnetic structure we obtained must thus be frustrated and possibly not the true ground state.

Let us put that aside for the moment and try to calculate $J$. Use the Heisenberg model with classical spins, nearest neighbor interaction only, and $A=0$ to derive that the energy per site of the two configurations can be written as

$$E_{\mathrm{FM}} = E_0 - \frac{1}{2}6S^2J$$

and

$$E_{\mathrm{AFM}} = E_0 + \frac{1}{2}2S^2J$$

per site, where $E_0$ is some reference energy. Use these expressions to eliminate $E_0$ and express $J$ in terms of the energy difference. Use the energies obtained with DFT to calculate $J$. You should get -1.4 meV. Do it with python in the cell below.
"""

# %%
E_fm = layer_fm.get_potential_energy() / 2  # student: E_fm = ???
E_afm = layer_afm.get_potential_energy() / 2  # student: E_afm = ???
dE = E_afm - E_fm  # student:
J = dE / 4 / S**2  # student: J = ???
print('J = %1.2f meV' % (J * 1000))

# %%
"""
### Non-collinear configuration
As it turn out the optimal spin structure of a hexagonal lattice with anti-ferromagntice coupling is taking all spins at 120$^\circ$ angles with respect to each other. Draw this structure and convince yourself that it can be done.

1. What is the minimal number of magnetic atoms required in the magnetic unit cell
2. Verrify that the Heisenberg model with classical spins gives a lower energy with this configuration that the anti-aligned structure calculated above. The energy per site of this state should be

$$E_{\mathrm{NC}}=E_0+\frac{3}{2}S^2J.$$

We will now check if LDA can verify this prediction. To do that we need to perform a calculation with non-collinear spin. This is done in the cell below. Assert that the the total energy per site is lower than what we obtained with the collinear anti-ferromagnetic configuration above. Also check the local magnetic moments printet at the end of the calculation.
"""

# %%
import numpy as np
from ase.io import read
from ase.visualize import view
from gpaw import GPAW, PW, MixerDif

m = 3
cell_cv = layer.get_cell()
layer_nc = layer.repeat((3, 1, 1))
new_cell_cv = [[3 * cell_cv[0, 0] / 2, 3**0.5 * cell_cv[0, 0] / 2, 0.0],
               [3 * cell_cv[0, 0] / 2, -3**0.5 * cell_cv[0, 0] / 2, 0.0],
               [0.0, 0.0, cell_cv[2, 2]]]
layer_nc.set_cell(new_cell_cv)
view(layer_nc)

magmoms = np.zeros((len(layer_nc), 3), float)
magmoms[0] = [m, 0, 0]
magmoms[3] = [m * np.cos(2 * np.pi / 3), m * np.sin(2 * np.pi / 3), 0]
magmoms[6] = [m * np.cos(2 * np.pi / 3), -m * np.sin(2 * np.pi / 3), 0]

calc = GPAW(mode=PW(400),
            xc='LDA',
            mixer=MixerDif(),
            symmetry='off',
            experimental={'magmoms': magmoms, 'soc': False},
            parallel={'domain': 1, 'band': 1},
            kpts=(2, 2, 1),
            )
layer_nc.calc = calc
layer_nc.get_potential_energy()
calc.write('nc_nosoc.gpw')

# teacher:
magmoms = np.zeros((len(layer_nc), 3), float)
magmoms[0] = [m, 0, 0]
magmoms[3] = [m, 0, 0]
magmoms[6] = [m, 0, 0]
calc = GPAW(mode=PW(400),
            xc='LDA',
            mixer=MixerDif(),
            symmetry='off',
            experimental={'magmoms': magmoms, 'soc': False},
            parallel={'domain': 1, 'band': 1},
            kpts=(2, 2, 1),
            )
layer_nc.calc = calc
layer_nc.get_potential_energy()
calc.write('fm_nosoc.gpw')

# %%
"""
### Anisotropy and exhange coupling from the non-collinear configuration

In the cell above we could have set 'soc'=True' to include spin-orbit
coupling in the self-consistent non-collinear solution. However, it is more
convenient for us to exclude it such that we can explicitly obtain the
anisotropy based on this calculation.

If the Heisenberg Hamiltonian with nearest neighbor interactions is a good
model we should be able to obtain both $J$ and $A$ from the non-collinear
calculation as well. Write some python code in the cell below that return $J$
and $A$ based on the non-collinear calculation. The calculation of $J$
requires two spin configurations and we could use the ferromagnetic
calculation in the simple unit cell obtained at the top of the notebook as
one of them. But, since the energy differences are rather small it is much
better if you can obtain a ferromagnetic state with the same unit cell and
parameters as we used for the non-collinear calculation. You may thus run the
cell above once more, but with ferromagnetic alignment.
"""

# %%
from gpaw.spinorbit import get_anisotropy
from math import pi

# teacher:
calc = GPAW('nc_nosoc.gpw', txt=None)
E_nc = calc.get_potential_energy() / 3
e_x = get_anisotropy(calc, theta=pi / 2, phi=0) / 3
e_y = get_anisotropy(calc, theta=pi / 2, phi=pi / 2) / 3
e_z = get_anisotropy(calc, theta=0, phi=0) / 3
de_zx = e_z - e_x
de_zy = e_z - e_y
print('NC: A_zx = %1.3f meV' %  (de_zx * 1000))
print('NC: A_zy = %1.3f meV' %  (de_zy * 1000))
print()

calc = GPAW('fm_nosoc.gpw', txt=None)
E_fm = calc.get_potential_energy() / 3
e_x = get_anisotropy(calc, theta=pi / 2, phi=0) / 3
e_y = get_anisotropy(calc, theta=pi / 2, phi=pi / 2) / 3
e_z = get_anisotropy(calc, theta=0, phi=0) / 3
de_zx = e_z - e_x
de_zy = e_z - e_y
print('FM: A_zx = %1.3f meV' %  (de_zx * 1000))
print('FM: A_zy = %1.3f meV' %  (de_zy * 1000))
print()

dE = E_nc - E_fm
J = 2 * dE / 9 / S**2
print('J = %1.2f meV' % (J * 1000))
print()
calc = GPAW('VI2_relaxed.gpw', txt=None)
E_fm = calc.get_potential_energy()
dE = E_nc - E_fm
J = 2 * dE / 9 / S**2
print('J = %1.2f meV' % (J * 1000))


# %%
"""
### Critical temperature?
Now answer the following questions:

1. What is the easy axis? (Hint: the anisotropy is calculated by rotating
   the initial spin configuration first by $\theta$ and then by $\varphi$).
   Does it agree with waht you found above for the simple ferromagnetic state?

2. Is there any rotational freedom left in the non-collinear ground state if
   we assume in plane isotropy?

You might be able to convince yourself that some degree of in-plane
anisotropy is required as well to obtain a finite critical temperature for
magnetic order.

Clearly the non-collinear spin state of VI$_2$ is more difficult to describe
than the ferromagnetic state in CrI$_3$ and we do not yet have a simple
theoretical expression fot the critical temperature as a function of
anisotropy and exchange coupling constants. However, with the rapid
development of excperimental techniques to synthesize and characterize 2D
materials it does seem plausible that such a non-collinear 2D material may be
observed in the near future.

Again, bear in mind that all the calculations in the present notebook ought
to be properly converged with respect to $k$-points, plane wave cutoff etc.
"""
