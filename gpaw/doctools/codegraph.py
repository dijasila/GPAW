import numpy as np
import graphviz
from gpaw.core.atom_centered_functions import AtomCenteredFunctions
from gpaw.core.atom_arrays import DistributedArrays, AtomArrays


g = graphviz.Digraph(node_attr={'shape': 'record'})
n = 0
ok = set("""\
Atoms
ASECalculator
DFTCalculation
Atoms
DFTState
IBZWaveFunctions
Density
Potential
SCFLoop
IBZ
MonkhorstPackKPoints
Davidson
InputParameters
Logger
Timer
OccupationNumberCalculator
Preconditioner
XCFunctional
Hamiltonian
Mixer
Symmetries
UniformGridPotentialCalculator
PWFDWaveFunctions
A
C""".splitlines())

skip = {'Atoms'}
subclasses = {}


def graph(obj=None, name=None, dct=None):
    global n
    id = str(n)
    n += 1
    name = name or obj.__class__.__name__
    dct = dct if dct is not None else obj.__dict__
    print(name)
    attrs = []
    for x, o in dct.items():
        if x[0] != '_':
            a = o.__class__.__name__
            if a in ok:
                x = x.replace('wfs_qs', 'wfs_qs[q][s]')
                if a in subclasses:
                    s, oo = subclasses[a]
                    at = {k for k in o.__dict__ if k[0] != '_'}
                    at0 = at.copy()
                    for o1 in oo:
                        at0 = at0 & {k for k in o1.__dict__ if k[0] != '_'}
                    y = graph(name=s, dct={k: v for k, v in o.__dict__.items()
                                           if k in at0})
                    g.edge(id, y, label=x)
                    for o1 in oo + [o]:
                        if o1.__class__.__name__ != s:
                            y1 = graph(name=o1.__class__.__name__,
                                       dct={k: v
                                            for k, v in o1.__dict__.items()
                                            if k not in at0})
                            g.edge(y1, y, arrowhead='onormal')
                else:
                    y = graph(o)
                    g.edge(id, y, label=x)
            elif name not in skip:
                if isinstance(o, bool):
                    h = 'bool'
                elif isinstance(o, int):
                    h = 'int'
                elif isinstance(o, str):
                    h = 'str'
                elif isinstance(o, float):
                    h = 'float'
                elif isinstance(o, dict):
                    h = 'dict'
                elif isinstance(o, list):
                    h = 'list'
                elif isinstance(o, np.ndarray):
                    h = 'ndarray'
                elif isinstance(o, AtomCenteredFunctions):
                    h = 'ACF'
                elif isinstance(o, AtomArrays):
                    h = 'AA'
                elif isinstance(o, DistributedArrays):
                    h = 'DA'
                elif hasattr(o, 'broadcast'):
                    h = 'MPIComm'
                else:
                    h = '?'
                attrs.append(f'{x}: {h}')
    if attrs:
        s = r'\n'.join(attrs)
        g.node(id, f'{{{name} | {s}}}')
    else:
        g.node(id, name)
    return id


class A:
    def __init__(self, b):
        self.a = 1
        self.b = b

    def m(self):
        pass


class B:
    pass


class C(B):
    pass


from gpaw.new.ase_interface import GPAW
import ase


fd = GPAW(mode='fd')
pw = GPAW(mode='pw')
a = ase.Atoms('H', cell=[2, 2, 2], pbc=1)


class Atoms:
    def __init__(self, calc):
        self.calc = calc


a0 = Atoms(fd)
fd.get_potential_energy(a)
pw.get_potential_energy(a)
fd.calculation.state.ibzwfs.wfs_qs = fd.calculation.state.ibzwfs.wfs_qs[0][0]
from gpaw.new.brillouin import BZPoints
from gpaw.new.wave_functions import WaveFunctions

subclasses['MonkhorstPackKPoints'] = ('BZPoints', [BZPoints(np.zeros((1, 3)))])
subclasses['UniformGridPotentialCalculator'] = (
    'PotentialCalculator', [pw.calculation.pot_calc])
subclasses['PWFDWaveFunctions'] = (
    'WaveFunctions', [
        WaveFunctions(0, pw.calculation.setups, np.zeros((1, 3)))])
obj = a0

graph(obj)

subclasses['C'] = ('B', [B()])
graph(A(C()))
g.view()
