import numpy as np
# from ase.lattice import bulk
from ase.units import Bohr, Hartree
from ase import Atoms
from gpaw import GPAW
# from gpaw.wavefunctions.pw import PW

pristine = GPAW('GaAs.pristine.gpw')
defect = GPAW('GaAs.Ga_vac.gpw')
atoms = pristine.atoms

pd = pristine.wfs.pd
G_Gv = pd.get_reciprocal_vectors(q=0, add_q=False)  # \vec{G} in Bohr^-1
G2_G = pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2

nG = len(G2_G)

# Gaussian
# rho_r = q  * (2 \pi sigma^2)^-1.5 * exp(-0.5*r^2/sigma^2)
# If r = sigma * \sqrt(2 ln 2), rho_r = 0.5 * rho_0
# i.e. FWHM = 2 * sigma * \sqrt(2 ln 2)
# sigma = FWHM / ( 2 \sqrt(2 ln 2))

q = -3.0  # charge in units of electron charge
FWHM = 2.0  # in Bohr
sigma = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))

epsilon = 12.7  # dielectric medium

rho_G = q * np.exp(-0.5 * G2_G * sigma * sigma)  # Fourier transformed Gaussian

Omega = atoms.get_volume() / (Bohr ** 3.0)  # Cubic Bohr

Elp = 0.0

for rho, G2 in zip(rho_G, G2_G):
    if np.isclose(G2, 0.0):
        print('Skipping G^2=0 contribution to Elp')
        continue
    Elp += 2.0 * np.pi / (epsilon * Omega) * rho * rho / G2

El = Elp - q * q / (2 * epsilon * sigma * np.sqrt(np.pi))
print(El * Hartree)


def calculate_z_avg_model_potential(rho_G, calc):
    pd = calc.wfs.pd
    G_Gv = pd.get_reciprocal_vectors(q=0, add_q=False)  # \vec{G} in Bohr^-1
    G2_G = pd.G2_qG[0]  # |\vec{G}|^2 in Bohr^-2
    r_3xyz = calc.density.gd.refine().get_grid_point_coordinates()

    nrx = np.shape(r_3xyz)[1]
    nry = np.shape(r_3xyz)[2]
    nrz = np.shape(r_3xyz)[3]

    cell = calc.atoms.get_cell()
    vox1 = cell[0, :] / Bohr / nrx
    vox2 = cell[1, :] / Bohr / nry
    vox3 = cell[2, :] / Bohr / nrz

    # This way uses that the grid is arranged with z increasing fastest, then y
    # then x (like a cube file)
    x_g = 1.0 * r_3xyz[0].flatten()
    y_g = 1.0 * r_3xyz[1].flatten()
    z_g = 1.0 * r_3xyz[2].flatten()

    selectedG = []
    for iG, G in enumerate(G_Gv):
        if np.isclose(G[0], 0.0) and np.isclose(G[1], 0.0):
            selectedG.append(iG)

    assert(np.isclose(G2_G[selectedG[0]], 0.0))
    selectedG.pop(0)
    zs = []
    Vs
    for ix in np.arange(nrz):
        phase_G = np.exp(1j * (G_Gv[selectedG, 2] * z_g[ix]))
        V = (np.sum(phase_G * rho_G[selectedG] / (G2_G[selectedG]))
             * Hartree * 4.0 * np.pi / (epsilon * Omega))
        s = str(z_g[ix]) + ' ' + str(np.real(V)) + ' ' + str(np.imag(V)) + '\n'
        zs.append(z_g[idx])
        Vs.append(V)

    V = (np.sum(rho_G[selectedG] / (G2_G[selectedG]))
         * Hartree * 4.0 * np.pi / (epsilon * Omega))
    zs.append(vox3[2])
    Vs.append(V)
    return np.array(zs), np.array(Vs)
    # outfilepotential.write(s)


z, v_model = calculate_z_avg_model_potential(rho_G, pristine)
v_0 = pristine.get_electrostatic_potential().mean(0).mean(0)
v_X = defect.get_electrostatic_potential().mean(0).mean(0)
