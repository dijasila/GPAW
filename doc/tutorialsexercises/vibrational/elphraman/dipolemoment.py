from gpaw import GPAW
from gpaw.lcao.dipoletransition import get_momentum_transitions


calc = GPAW("scf.gpw")

if not hasattr(calc.wfs, 'C_nM'):
    print("Need to initialise")
    calc.initialize_positions(calc.atoms)
get_momentum_transitions(calc.wfs, savetofile=True)
