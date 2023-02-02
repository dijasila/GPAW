"""GLLBSC band gap tutorial

Calculate the band gap of Si using GLLBSC.
"""
# P1
from ase.build import bulk
from gpaw import GPAW, PW, FermiDirac

# Ground state calculation
atoms = bulk('Si', 'diamond', 5.431)
calc = GPAW(mode=PW(200),
            xc='GLLBSC',
            kpts=(7, 7, 7),  # Choose and converge carefully!
            occupations=FermiDirac(0.01),
            txt='gs.out')
atoms.calc = calc
atoms.get_potential_energy()
# P2
# Calculate the discontinuity potential and the discontinuity
homo, lumo = calc.get_homo_lumo()
response = calc.hamiltonian.xc.response
dxc_pot = response.calculate_discontinuity_potential(homo, lumo)
KS_gap, dxc = response.calculate_discontinuity(dxc_pot)

# Fundamental band gap = Kohn-Sham band gap + derivative discontinuity
QP_gap = KS_gap + dxc

print(f'Kohn-Sham band gap:         {KS_gap:.2f} eV')
print(f'Discontinuity from GLLB-sc: {dxc:.2f} eV')
print(f'Fundamental band gap:       {QP_gap:.2f} eV')
