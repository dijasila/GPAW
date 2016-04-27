import numpy as np

cs = '224'

pristine = np.loadtxt('GaAs.'+cs+'.pristine.V_av.dat')
defective = np.loadtxt('GaAs.'+cs+'.Ga_vac.V_av.dat')
model = np.loadtxt('GaAs.'+cs+'.model.V_av.dat')
diff1 = defective[:,1] - pristine[:,1]
dV = model[:,1] - diff1
print '# z model-defective+pristine defective-pristine'
for z, V, diff in zip(pristine[:,0], dV, diff1):
    s = str(z) + ' ' + str(V) + ' ' + str(diff)
    print(s)
import numpy as np

from ase import Atoms
from ase.lattice import bulk
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.wavefunctions.pw import PW

cx = 2
cy = 2
cz = 4
a = 5.628

lattice      = np.array([[a, 0.0 ,0.0], 
                         [0.0,a,0.0],
                         [0.0,0.0,a]])

basiccell = Atoms(
                  symbols = 'H',
                  scaled_positions = [[0.0,0.0,0.0]],
                  cell = lattice,
                  pbc = (1,1,1)
                 )

#basiccell = bulk('SiC', 'zincblende', a=4.36)

supercell_size = (cx,cy,cz)
unitcell = basiccell.repeat(supercell_size)

calc = GPAW(kpts={'size':(1,1,1),'gamma':True},
            dtype=complex,
            symmetry='off',
            mode = PW(20 * Hartree))
calc.initialize(unitcell)

G_Gv = calc.wfs.pd.get_reciprocal_vectors(q=0,add_q=False) # \vec{G} in Bohr^-1
G2_G = calc.wfs.pd.G2_qG[0] # |\vec{G}|^2 # In Bohr^-2

nG = len(G2_G)

# Gaussian
# rho_r = q  * (2 \pi sigma^2)^-1.5 * exp(-0.5*r^2/sigma^2)
# If r = sigma * \sqrt(2 ln 2), rho_r = 0.5 * rho_0
# i.e. FWHM = 2 * sigma * \sqrt(2 ln 2)
# sigma = FWHM / ( 2 \sqrt(2 ln 2))

q = -3.0 # charge in units of electron charge
FWHM = 2.0 # in Bohr
sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

epsilon = 12.7 # dielectric medium

rho_G = q * np.exp(-0.5*G2_G*sigma*sigma) # Fourer transform of Gaussian

Omega = unitcell.get_volume() / (Bohr ** 3.0) # Cubic Bohr

Elp = 0.0

for rho, G2 in zip(rho_G, G2_G):
    if np.isclose(G2,0.0):
        print('Hi!')
        continue
    Elp += 2.0 * np.pi / (epsilon * Omega) * rho * rho/ G2

El = Elp - q * q / (2 * epsilon * sigma * np.sqrt(np.pi))
print(El*Hartree)
import numpy as np

from ase import Atoms
from ase.lattice import bulk
from ase.units import Bohr, Hartree

from gpaw import GPAW
from gpaw.wavefunctions.pw import PW

cx = 2
cy = 2
cz = 4
a = 5.628

lattice      = np.array([[a, 0.0 ,0.0], 
                         [0.0,a,0.0],
                         [0.0,0.0,a]])

basiccell = Atoms(
                  symbols = 'H',
                  scaled_positions = [[0.0,0.0,0.0]],
                  cell = lattice,
                  pbc = (1,1,1)
                 )

#basiccell = bulk('SiC', 'zincblende', a=4.36)

supercell_size = (cx,cy,cz)
unitcell = basiccell.repeat(supercell_size)

calc = GPAW(kpts={'size':(1,1,1),'gamma':True},
            dtype=complex,
            symmetry='off',
            mode = PW(20 * Hartree))
calc.initialize(unitcell)

G_Gv = calc.wfs.pd.get_reciprocal_vectors(q=0,add_q=False) # \vec{G} in Bohr^-1
G2_G = calc.wfs.pd.G2_qG[0] # |\vec{G}|^2 # In Bohr^-2

nG = len(G2_G)

r_3xyz = calc.density.gd.refine().get_grid_point_coordinates() # r[:,ix,iy,iz] gives point ix iy iz in Bohr

nrx = np.shape(r_3xyz)[1]
nry = np.shape(r_3xyz)[2]
nrz = np.shape(r_3xyz)[3]

nx = nrx * nry * nrz

# Gaussian
# rho_r = q  * (2 \pi sigma^2)^-1.5 * exp(-0.5*r^2/sigma^2)
# If r = sigma * \sqrt(2 ln 2), rho_r = 0.5 * rho_0
# i.e. FWHM = 2 * sigma * \sqrt(2 ln 2)
# sigma = FWHM / ( 2 \sqrt(2 ln 2))

q = -3.0 # charge in units of electron charge
FWHM = 2.0 # in Bohr
sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

epsilon = 12.7 # dielectric medium

rho_G = q * np.exp(-0.5*G2_G*sigma*sigma) # Fourer transform of Gaussian

# Fourier Transform and cube output

Omega = unitcell.get_volume() / (Bohr ** 3.0) # Cubic Bohr

cellall = unitcell.get_cell() # Voxels for cube file
vox1 = cellall[0,:] / Bohr / nrx
vox2 = cellall[1,:] / Bohr / nry
vox3 = cellall[2,:] / Bohr / nrz




# This way uses that the grid is arranged with z increasing fastest, then y then x (like a cube file)
x_g = 1.0 * r_3xyz[0].flatten()
y_g = 1.0 * r_3xyz[1].flatten()
z_g = 1.0 * r_3xyz[2].flatten()

selectedG = []
for iG, G in enumerate(G_Gv):  # Only need (0,0,Gz) G vectors for z average
    if np.isclose(G[0],0.0) and np.isclose(G[1],0.0):
        selectedG.append(iG)

potential = True
zread = True
if potential:

    if zread:
        z_g = np.loadtxt('input_z.'+str(cx)+str(cy)+str(cz)+'.dat')
        nrz = len(z_g)
#   Then the potential
#   Set the V(G=0) term to zero:
    assert(np.isclose(G2_G[selectedG[0]],0.0))
    selectedG.pop(0)

    outfilepotential = open('potential.zav.model.dat','w')
    for ix in np.arange(nrz):
        phase_G = np.exp(1j * (G_Gv[selectedG,2] * z_g[ix]))
        V = 4.0 * np.pi / (epsilon * Omega) * np.sum(phase_G * rho_G[selectedG]/(G2_G[selectedG])) * Hartree
        outfilepotential.write(str(z_g[ix]) + ' ' + str(np.real(V)) + ' ' + str(np.imag(V)) + '\n')

    # Add the periodic repeat of the zero term again (helps for integration)
    if not zread:
        V = 4.0 * np.pi / (epsilon * Omega) * np.sum(rho_G[selectedG]/(G2_G[selectedG])) * Hartree
        outfilepotential.write(str(vox3[2] * nrz) + ' ' + str(np.real(V)) + ' ' + str(np.imag(V)) + '\n') 
import numpy as np 
from ase.units import Hartree, Bohr
from gpaw import GPAW

calc = GPAW('../cell_224/GaAs.Ga_vac.gpw', txt=None)
#calc = GPAW('../cell_224/GaAs.pristine.gpw', txt=None)
calc.restore_state()
v = (-1.0) * calc.hamiltonian.vHt_g * Hartree # units eV; potential energy = qV

v_z = v.mean(0).mean(0)
z_z = np.linspace(0, calc.atoms.cell[2, 2], len(v_z) + 1, endpoint=True)

for iv in np.arange(len(v_z)):
    z = z_z[iv]
    v = v_z[iv]
    print(str(z/Bohr) + ' ' + str(v))


# Last point
print(str(z_z[-1]/Bohr) + ' ' + str(v_z[0]))
