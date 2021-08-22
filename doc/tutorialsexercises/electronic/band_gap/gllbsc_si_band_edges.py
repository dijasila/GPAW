# web-page: bs_si.png
"""GLLBSC band gap tutorial

Calculate the band structure and band gap of Si using GLLBSC.
"""
# P1
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

# Ground state calculation
atoms = bulk('Si', 'diamond', 5.431)
calc = GPAW(mode=PW(200),
            xc='GLLBSC',
            kpts={'size': (8, 8, 8), 'gamma': True},
            occupations=FermiDirac(0.01),
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()

# Band structure calculation with fixed density
bs_calc = calc.fixed_density(nbands=10,
                             kpts={'path': 'LGXWKL', 'npoints': 60},
                             symmetry='off',
                             convergence={'bands': 8},
                             txt='bs.out')

# Plot the band structure
bs = bs_calc.band_structure().subtract_reference()
bs.plot(filename='bs_si.png', emin=-6, emax=6)
# P2
# Get the accurate HOMO and LUMO from the band structure calculator
homo, lumo = bs_calc.get_homo_lumo()

# Calculate the discontinuity potential using the ground state calculator and
# the accurate HOMO and LUMO
response = calc.hamiltonian.xc.response
dxc_pot = response.calculate_discontinuity_potential(homo, lumo)

# Calculate the discontinuity using the band structure calculator
bs_response = bs_calc.hamiltonian.xc.response
KS_gap, dxc = bs_response.calculate_discontinuity(dxc_pot)

# Fundamental band gap = Kohn-Sham band gap + derivative discontinuity
QP_gap = KS_gap + dxc

print(f'Kohn-Sham band gap:         {KS_gap:.2f} eV')
print(f'Discontinuity from GLLB-sc: {dxc:.2f} eV')
print(f'Fundamental band gap:       {QP_gap:.2f} eV')
