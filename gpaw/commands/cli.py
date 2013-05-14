from gpaw.wavefunctions.pw import PW
from gpaw.occupations import FermiDirac, MethfesselPaxton
from gpaw.mixer import Mixer, MixerSum, MixerDif
from gpaw.poisson import PoissonSolver
from gpaw.eigensolvers import RMM_DIIS
from ase.cli.cli import run as aserun


hook = {'name': 'gpaw',
        'namespace': {'PW': PW,
                      'FermiDirac': FermiDirac,
                      'MethfesselPaxton': MethfesselPaxton,
                      'Mixer': Mixer,
                      'MixerSum': MixerSum,
                      'MixerDif': MixerDif,
                      'PoissonSolver': PoissonSolver,
                      'RMM_DIIS': RMM_DIIS},
        'commands': ['egg-box-test', 'bulk-test'],
        'command_module': 'gpaw.commands'}


def run(command=None, **kwargs):
    aserun(command, hook=hook, calculator='gpaw', **kwargs)
