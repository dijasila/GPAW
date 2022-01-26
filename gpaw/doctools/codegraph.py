import ase
import graphviz
import numpy as np
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   DistributedArrays)
from gpaw.core.atom_centered_functions import AtomCenteredFunctions
from gpaw.new.ase_interface import GPAW
from gpaw.new.brillouin import BZPoints
from gpaw.new.builder import builder
from gpaw.new.wave_functions import WaveFunctions

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


def make_graph(obj, skip_first=False):
    g = graphviz.Digraph(node_attr={'shape': 'record'})
    graph(g, obj, skip_first=skip_first)
    return g


def graph(g, obj=None, name=None, dct=None, skip_first=False):
    global n
    id = str(n)
    n += 1
    name = name or obj.__class__.__name__
    dct = dct if dct is not None else obj.__dict__
    print(name, list(dct))
    attrs = []
    for x, o in dct.items():
        if x[0] != '_':
            a = o.__class__.__name__
            print('hmmm', x, a)
            if a in ok:
                x = x.replace('wfs_qs', 'wfs_qs[q][s]')
                if a in subclasses:
                    s, oo = subclasses[a]
                    print(x, s, oo)
                    at = {k for k in o.__dict__ if k[0] != '_'}
                    at0 = at.copy()
                    for o1 in oo:
                        at0 = at0 & {k for k in o1.__dict__ if k[0] != '_'}
                    y = graph(g,
                              name=s,
                              dct={k: v for k, v in o.__dict__.items()
                                   if k in at0})
                    if not skip_first:
                        g.edge(id, y, label=x)
                    for o1 in oo + [o]:
                        if o1.__class__.__name__ != s:
                            y1 = graph(g,
                                       name=o1.__class__.__name__,
                                       dct={k: v
                                            for k, v in o1.__dict__.items()
                                            if k not in at0})
                            g.edge(y1, y, arrowhead='onormal')
                else:
                    y = graph(g, o)
                    if not skip_first:
                        g.edge(id, y, label=x)
            elif name not in skip:
                if isinstance(o, bool):
                    h = 'bool'
                elif isinstance(o, (int, np.int64)):
                    h = 'int'
                elif isinstance(o, str):
                    h = 'str'
                elif isinstance(o, float):
                    h = 'float'
                elif isinstance(o, dict):
                    h = 'dict'
                elif isinstance(o, list):
                    h = 'list'
                elif isinstance(o, tuple):
                    h = 'tuple'
                elif isinstance(o, np.ndarray):
                    h = 'ndarray'
                elif isinstance(o, AtomCenteredFunctions):
                    h = 'ACF'
                elif isinstance(o, AtomArrays):
                    h = 'AA'
                elif isinstance(o, DistributedArrays):
                    h = 'DA'
                elif isinstance(o, PlaneWaves):
                    h = 'PlaneWaves'
                elif isinstance(o, UniformGrid):
                    h = 'UniformGrid'
                elif hasattr(o, 'broadcast'):
                    h = 'MPIComm'
                elif o == float:
                    h = 'dtype'
                else:
                    h = '?'
                attrs.append(f'{x}: {h}')
    if skip_first:
        return id
    if attrs:
        s = r'\n'.join(attrs)
        g.node(id, f'{{{name} | {s}}}')
    else:
        g.node(id, name)
    return id


def make_figures():
    fd = GPAW(mode='fd')
    pw = GPAW(mode='pw')
    a = ase.Atoms('H', cell=[2, 2, 2], pbc=1)

    class Atoms:
        def __init__(self, calc):
            self.calc = calc

    a0 = Atoms(fd)
    fd.get_potential_energy(a)
    pw.get_potential_energy(a)
    ibzwfs = fd.calculation.state.ibzwfs
    ibzwfs.wfs_qs = ibzwfs.wfs_qs[0][0]

    subclasses['MonkhorstPackKPoints'] = (
        'BZPoints', [BZPoints(np.zeros((1, 3)))])
    subclasses['UniformGridPotentialCalculator'] = (
        'PotentialCalculator', [pw.calculation.pot_calc])
    subclasses['PWFDWaveFunctions'] = (
        'WaveFunctions', [
            WaveFunctions(0, pw.calculation.setups, np.zeros((1, 3)))])
    obj = a0

    g = make_graph(obj)
    g.render('code', format='svg')

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

    subclasses['C'] = ('B', [B()])
    g = make_graph(A(C()))
    g.render('abc', format='svg')

    ok.add('UniformGridAtomCenteredFunctions')
    subclasses['UniformGridAtomCenteredFunctions'] = (
        'AtomCenteredFunctions (ACF)',
        [pw.calculation.pot_calc.nct_ag])
    obj = A(1)
    obj.a = fd.calculation.pot_calc.nct_aR
    g = make_graph(obj, skip_first=True)
    g.render('acf', format='svg')

    ok.add('UniformGridFunctions')
    ok.add('UniformGrid')
    subclasses['UniformGridFunctions'] = (
        'DistributedArrays (DA)',
        [pw.calculation.state.ibzwfs.wfs_qs[0][0].psit_nX])
    subclasses['UniformGrid'] = (
        'Domain',
        [pw.calculation.state.ibzwfs.wfs_qs[0][0].psit_nX.desc])
    obj = A(1)
    obj.a = fd.calculation.state.ibzwfs.wfs_qs.psit_nX
    g = make_graph(obj, skip_first=True)
    g.render('da', format='svg')

    b = []
    for mode in ['fd', 'pw']:
        b.append(builder(a, {'mode': mode}))
    ok.clear()
    ok.add('FDDFTComponentsBuilder')
    ok.add('XCFunctional')
    ok.add('Atoms')
    ok.add('InputParameters')
    ok.add('IBZ')
    ok.add('Symmetries')
    ok.add('MonkhorstPackKPoints')
    subclasses['FDDFTComponentsBuilder'] = (
        'DFTComponentsBuilder',
        [b[1]])
    obj = A(1)
    obj.a = b[0]
    g = make_graph(obj, skip_first=True)
    g.render('builder', format='svg')

    ok.add('AtomArrays')
    ok.add('AtomArraysLayout')
    ok.add('AtomDistribution')
    obj = A(1)
    obj.a = AtomArraysLayout([1]).empty()
    g = make_graph(obj, skip_first=True)
    g.render('aa', format='svg')


if __name__ == '__main__':
    make_figures()
