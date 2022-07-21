# %%
# teacher
import ase.visualize as viz
viz.view = lambda atoms, repeat=None: None

# %%
"""
# Curie temperature of CrI$_3$

In 2017, ferromagnetic order was observed in a monolayer of CrI$_3$ below 45 K (*Nature* **546** 270 (2017)). It comprises the first demonstration of magnetic order in a 2D material and has received a lot of attention due to the peculiar properties of magnetism in 2D. The physics of magnetic order in 2D is rather different than in 3D and in order to understand what is going on we will need to introduce a bit of theory. But before we get to that let us get started with the calculations.
"""

# %%
"""
## DFT calculation - finding the atomic structure of CrI$_3$

We start by setting up the atomic structure of a CrI$_3$ monolayer and optimizing the atomic positions and unit cell. There are two formula units in the minimal unit cell and only the Cr atoms bear significant magnetic moments. A spin-polarized calculation is initiated by specifying the initial magnetic moments of the all the atoms in units of $\mu_B$.

1. What do you expect the ionic value for the magnetic moment of the Cr atoms to be? (Hint: Use Hund's rule. The electronic configuration of a Cr atom is [Ar]3d$^5$4s$^1$ and each of the iodine atoms will steal one electron). Use your answer to fill in the spin state `S` in the cell below.

Try to understand the individual lines in the input cell below and run it. The calculation will open the ase gui to show the initial atomic structure. You may, for example, try to repeat the structure (under view) to get a better feeling for the material. Then look at the text output below the cell and answer the following questions:

2.  How many electrons are used in the calculation? Which valence states of Cr and I are included?
3.  What is the total magnetic moment after the first DFT calculation and on which atoms are the magnetic moments located?
4.  What is the number of irreducible k-points (k-points not related by symmetry) in the calculation?
5.  What is the maximum force on the atoms after the first DFT calculation? Does it become smaller after subsequent calculations?

In order to get the script running fast, we have set a few of the computational parameters to values which are expected to produce a somewhat inaccurate result. Can you identify what parameters one would need to converge or modify in order to produce more accurate results?

Leave the script running and continue with the theory section below.
"""

# %%
# %load summerschool.py
from gpaw import GPAW, PW
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter

a = 7.0
c = 10.0
S = 3 / 2  # student: S = ???
m = S * 2                              # Magnetic moment in Bohr magnetons
cell = [[a, 0, 0],
        [-0.5 * a, 3**0.5 / 2 * a, 0],
        [0, 0, c]]                     # Unit cell in \AA
scaled_pos = [[2 / 3., 1 / 3., 0.5],
              [1 / 3., 2 / 3., 0.5],
              [0.6, 1.0, 0.35],
              [0.4, 0.4, 0.35],
              [0.0, 0.6, 0.35],
              [0.4, 1.0, 0.65],
              [0.6, 0.6, 0.65],
              [0.0, 0.4, 0.65]]         # Positions in unit cell vectors
a = Atoms('Cr2I6', cell=cell, scaled_positions=scaled_pos, pbc=True)
a.set_initial_magnetic_moments([m, m, 0, 0, 0, 0, 0, 0])
view(a)

Nk = 2
calc = GPAW(mode=PW(200),
                xc='PBE',
                convergence={'density': 0.001, 'eigenstates': 0.1},
                kpts=(Nk, Nk, 1))
a.calc = calc

uf = UnitCellFilter(a, mask=[1, 1, 0, 0, 0, 1])
opt = BFGS(uf)
opt.run(fmax=0.2)
calc.write('CrI3_relaxed.gpw')

# %%
"""
## A bit of theory

### The Heisenberg model

If we want to calculate the Curie temperature of CrI$_3$ it is, in principle, clear what we have to do. We should calculate the magnetization as a function of temperature using standard methods from statistical physics, and record the temperature where the magnetization vanishes. Unfortunately, this requires knowledge of all the excited states of the system, which we do not have access to. In particular, the magnetization will be dominated by collective magnetic exciations, and these are not directly accessible from the Kohn-Sham spectrum produced by our DFT calculations.

Instead we will consider the Heisenberg Hamiltonian, which captures the basic physics of typical spin systems. It is given by

$$H = -\frac{1}{2}\sum_{ij}J_{ij}\mathbf{S}_i\cdot \mathbf{S}_j,$$

where $\mathbf{S}_i$ denotes the spin operator at site $i$ in units of $\hbar$ and $J_{ij}$ are magnetic exchange coupling constants. If we want to model a real material with the Heisenberg model we then need to identify a set of magnetic sites and calculate the exchange coupling constants $J_{ij}$.

1.   What are the magnetic sites of CrI$_3$?
2.   How many nearest neighbors does each magnetic site have?
3.   What are the possible values of $S_i^z$ for a magnetic site in CrI$_3$?
4.   What is the unit of $J_{ij}$?

In the following, we will assume that the physics is dominated by neareast neighbor interactions such that $J_{ij}\equiv J$ if atoms $i$ and $j$ are nearest neighbors and zero otherwise. In 3D systems a reasonable estimate of the Curie temperature can be obtained from mean-field theory as

$$T_c^{\mathrm{MF}}=\frac{nS(S+1)J}{3k_B},$$

where $k_B$ is Boltzmann's constant, $n$ is the number of nearest neighbors, and $S$ is the maximum value of $S^z_i$.
"""

# %%
"""
## DFT calculation of $J$

We now want to make a first principles calculation of the nearest neighbor exchange coupling constant $J$. Since the exchange coupling essentially measures the energy difference between aligned and anti-aligned spin configurations we can obtain $J$ by considering the energy difference between a ferromagnetic and an anti-ferromagnetic calculation. Note that both can be obtained as DFT ground states subjected to different spin constraints. For the CrI$_3$ system, $J$ can calculated as

$$J=\frac{E_{\mathrm{AFM}}-E_{\mathrm{FM}}}{3S^2},$$

where $E_{\mathrm{FM}}$ is the energy of the anti-ferromagnetic configuration and $E_{\mathrm{AFM}}$ is the energy of the ferromagnetic configuration. Try to derive this expression from the Heisenberg model with classical spins. In particular, where did the factor of 3 come from?

We have compiled a database of various 2D materials at https://cmrdb.fysik.dtu.dk/?project=c2db, which are relaxed with the PBE functional. We will therefore refrain from doing a full coverged geometry optimization and simply download the optimized structure from the database. Search the database for CrI$_3$ and select the ferromagnetic structure (The present material has the prototype BiI$_3$). You can now look at various properties of the material like band structure and stability if you like. Download the .xyz filem, save it as CrI3.xyz, and run the cell below to obtain a .gpw file containing a converged ferromagnetic calculation. The calculation will take about 30 minutes. To speed it up, you can submit this and the next job to DTU computers on multiple CPU cores, i.e. parallelize over 8 k-points. Follow instructions [here](https://wiki.fysik.dtu.dk/gpaw/summerschools/summerschool18/submitting.html). If the relaxation in the cell above did not finish you may kill it. It is not crucial to complete the rest of the exercise. Continue with the theory below while you wait for the calculations to finish.
"""

# %%
from ase.io import read
from gpaw import GPAW, PW

a = read('CrI3.xyz')
a.set_initial_magnetic_moments([m, m, 0, 0, 0, 0, 0, 0])

Nk = 4
calc = GPAW(mode=PW(400), xc='PBE', kpts=(Nk, Nk, 1), symmetry='off')
a.calc = calc
a.get_potential_energy()
calc.write('CrI3_fm.gpw')


# %%
"""
Now run the cell again but change the initial magnetic configuration such that it becomes anti-ferromagnetic (remember to change the name of the .gpw file so you do not overwrite the CrI3_fm.gpw). In practise we do not need to constrain the anti-ferromagnetic calculation because it comprises a local minimum, but check the magnetic moments in the output to verify that we indeed ended up with an anti-ferromagnetic state!
"""

# %%
# teacher:
a = read('CrI3.xyz')
a.set_initial_magnetic_moments([m, -m, 0, 0, 0, 0, 0, 0])

Nk = 4
calc = GPAW(mode=PW(400), xc='PBE', kpts=(Nk, Nk, 1))
a.calc = calc
a.get_potential_energy()
calc.write('CrI3_afm.gpw')

# %%
"""
$J$ and $T_c^{\mathrm{MF}}$ can then be evaluated by
"""

# %%
calc_fm = GPAW('CrI3_fm.gpw', txt=None)      # Ferromagnetic calculation
calc_afm = GPAW('CrI3_afm.gpw', txt=None)    # Anti-ferromagnetic calculation

N = 3
E_fm = calc_fm.get_potential_energy() / 2    # Energy per site
E_afm = calc_afm.get_potential_energy() / 2  # Energy per site
dE = E_afm - E_fm

J = dE / S**2 / 3  # student: J = ???
print(f'J = {J * 1000:1.3f} meV')

from ase.units import kB                     # Boltzmann's constant in eV/K
T_c = N * J * S * (S + 1) / 3 / kB  # student: T_c = ???
print(f'T_c(MF) = {T_c:1.1f} K')

# %%
"""
## More theory

### The Mermin-Wagner theorem
You should obtain a value of $T_c$, which is on the order of 100 K. This is much larger than the experimental value.  However in 2D materials mean-field theory fails miserably and the result cannot be trusted. In fact, at finite temperatures the Heisenberg model stated above does not exhibit magnetic order in 2D. The reason is that entropy tends to dominate over internal energy in 2D, such that the free energy is always minimized by disordered configurations at finite temperatures. This is the content of the Mermin-Wagner theorem which states that:

*Continuous symmetries cannot be spontaneously broken at finite temperature in systems with short-range interactions in dimensions $d\le2$*.

The Heisenberg model above has a continuous rotational symmetry of the spins and magnetic order is obtained by choosing a certain direction for all the spins. This is known as a spontaneously broken symmetry, because the magnetically ordered state has broken the symmetry of the Hamiltonian. The direction of magnetization is arbitrary and can be rotated at no energy cost, but the spins are aligned with respect to each other and the system is therefore magnetically ordered.

In the Heisenberg model it is straightforward to calculate the magnetic excitations in the spin-wave approximation, which yields the dispersion

$$\varepsilon(q)=aq^2$$

for small values of $q$. The excitations are bosons and the magnetization can thus be calculated from

$$M=M_0 - \int_0^\infty\frac{g(\epsilon)d\varepsilon}{e^{\varepsilon/k_bT} - 1}$$

where $M_0$ is the ground state magnetization and $g(\varepsilon)$ is the density of states.

1.  Show that the integral diverges for $d\le2$ in the lower limit. (Hint you should start by calculating $g(\varepsilon)$. If you do not remember how to do that we give it here: $g(\varepsilon)=b\varepsilon^{(d-2)/2}$, where $d$ is the dimension of the problem).

The divergence of the integral in $d\le2$ signals that the ground state is unstable and comprises an example of the Mermin-Wagner theorem: in $d\le2$, the free energy is always dominated by entropy at finite temperatures and magnetic order cannot be maintained if a material has rotational symmetry of the spins. The instability in 2D is closely related to the vanishing energy of magnetic excitations in the limit of $q\rightarrow0$, which is a consequence of the rotational symmetry.

### Magnetic anisotropy
The consequence of the Mermin Wagner theorem is that magnetic order in 2D is only possible if the rotational symmetry of the spins is *explicitly broken*. That is, there must be an additional term in the Hamiltonian that breaks the symmetry. Such a term can be provided by spin-orbit coupling which couples the spin to the lattice through the electronic orbitals.
We assume that CrI$_3$ is isotropic in the plane of the monolayer and introduce an anisotropy term in the Heisenberg Hamiltonian of the form

$$H_{\mathrm{ani}}=A\sum_i(S_i^z)^2,$$

where we have chosen the $z$-direction to be orthogonal to the plane. Try to
describe the physics of this term in the cases of $A<0$ and $A>0$. Does the
term fully break rotational symmetry of the ground state in both cases?

## Magnetic anisotropy from DFT

In the cell below the magnetic anisotropy is calculated for the ferromagnetic
ground state. The function calculate_band_energy() will return the sum over
occupied SOC corrected states. This energy will depend on the direction of
the spins, which is specified by the polar and azimuthal angles $\theta$ and
$\varphi$ respectively.

What is the sign of $A$ in the Hamiltonian above? Does spin-orbit coupling
break the rotational symmetry of the ground state?

"""

# %%
from gpaw.spinorbit import soc_eigenstates

e_x = soc_eigenstates(calc_fm, theta=90, phi=0).calculate_band_energy() / 2
e_y = soc_eigenstates(calc_fm, theta=90, phi=90).calculate_band_energy() / 2
e_z = soc_eigenstates(calc_fm, theta=0, phi=0).calculate_band_energy() / 2
de_zx = e_z - e_x
de_zy = e_z - e_y
print(f'dE_zx = {de_zx * 1000:1.3f} meV')
print(f'dE_zy = {de_zy * 1000:1.3f} meV')
A = de_zx  # student: A = ???


# %%
"""
We can also plot the total energy of the ground state as a function of angle
with the out of plane direction. This is done by the cell below.

Run the cell and inspect the plot. Does it look like you would expect?
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

thetas = np.linspace(0, 180, 13)

e_n = []
for theta in thetas:
    soc = soc_eigenstates(calc_fm, theta=theta, phi=0)
    e_n.append(soc.calculate_band_energy() / 2)

e_n = np.array(e_n) - e_n[0]

plt.figure()
plt.plot(thetas, e_n * 1000, 'o-')
plt.xlabel(r'$\theta$', size=18)
plt.ylabel('E [meV]', size=18)


# %%
"""
Now we have obtained the the anisotropy constant A. But how do we get the critical temperature if we cannot apply mean-field theory? One
way is to perform Monte-Carlo simulations of the Heisenberg model at different temperature and find the point where the total magnetization
vanishes. The results are well approximated by the expression

$$T_c=T_c^{\mathrm{Ising}}\tanh^{1/4}\Big[\frac{6}{N}\log\Big(1-0.033\frac{A}{J}\Big)\Big],$$

where $N$ is the number of nearest neighbors and and $T_c^{\mathrm{Ising}}=1.52\cdot S^2J/k_B$ is the critical temperature of the
corresponding Ising model. Calculate the critical temperature from this expression using the values of $A$ and $J$ found above. Do with
python in the cell below.

"""

# %%
from numpy import tanh, log

T0 = 1.52
T_c = T0 * S**2 * J / kB * (tanh(2 * log(1 - 0.033 * A / J)))**(0.25)  # student: T_c = ???
print(f'T_c = {T_c:.1f} K')


# %%
"""
The result for $T_c$ should be in reasonable agreement with the experimental value. Of course, one should carefully check the convergence
of all calculations in the present notebook. In fact a converged calculation yields $J = 3.1$ meV and $A=-0.38$ meV, which results in $T_c
= 36$ K.
"""
