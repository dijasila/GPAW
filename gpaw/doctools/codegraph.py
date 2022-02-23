import ase
import numpy as np
from gpaw.core import PlaneWaves, UniformGrid
from gpaw.core.matrix import Matrix
from gpaw.core.atom_arrays import (AtomArrays, AtomArraysLayout,
                                   DistributedArrays)
from gpaw.core.atom_centered_functions import AtomCenteredFunctions
from gpaw.new.ase_interface import GPAW
from gpaw.new.brillouin import BZPoints
from gpaw.new.builder import builder
from gpaw.new.wave_functions import WaveFunctions
from gpaw.fd_operators import FDOperator
from gpaw.new.eigensolver import Eigensolver
from gpaw.new.old import OldStuff


def create_nodes(obj, objects, include):
    node1 = create_node(obj, include)
    nodes = {node.name: node for node in node1.nodes()}
    for obj in objects:
        node = create_node(obj, include=lambda obj: False)
        assert node.name not in nodes
        nodes[node.name] = node

    for name, node in nodes.items():
        node.subclasses = []
        for cls in node.obj.__class__.__subclasses__():
            if cls.__name__ in nodes:
                node.subclasses.append(nodes[cls.__name__])

    new = {}
    for name, node in nodes.items():
        bases = node.obj.__class__.__bases__
        (cls,) = bases
        if cls is not object:
            base = nodes.get(cls.__name__)
            base = base or new.get(cls.__name__)
            if not base:
                base = Node(node.obj, node.attrs, list(node.has),
                            lambda obj: False)
                base.name = cls.__name__
                new[cls.__name__] = base
            if node not in base.subclasses:
                base.subclasses.append(node)
            node.base = base

    for name, node in new.items():
        nodes[name] = node

    for node in nodes.values():
        node.fix()

    return list(nodes.values())


def create_node(obj, include):
    attrs = []
    arrows = []
    print(obj, obj.__class__)
    for key, value in obj.__dict__.items():
        if key[0] != '_':
            if not include(value):
                attrs.append(key)
            else:
                arrows.append(key)
    return Node(obj, attrs, arrows, include)


class Node:
    def __init__(self, obj, attrs, arrows, include):
        self.obj = obj
        self.name = obj.__class__.__name__
        self.attrs = attrs
        self.has = {key: create_node(getattr(obj, key), include)
                    for key in arrows}
        self.base = None
        self.subclasses = []
        self.rgb = None

    def __repr__(self):
        return (f'Node({self.name}, {self.attrs}, {list(self.has)}, ' +
                f'{self.base.name if self.base is not None else None}, ' +
                f'{[o.name for o in self.subclasses]})')

    def nodes(self):
        yield self
        for node in self.has.values():
            yield from node.nodes()

    def keys(self):
        return set(self.attrs + list(self.has))

    def superclass(self):
        return self if self.base is None else self.base.superclass()

    def fix(self):
        keys = self.keys()
        print('FIX', self.name, keys)
        for obj in self.subclasses:
            print(obj)
            keys -= obj.keys()
        print('FIX', self.name, keys)
        self.attrs = [attr for attr in self.attrs if attr in keys]
        self.has = {key: value for key, value in self.has.items()
                    if key in keys}
        for obj in self.subclasses:
            obj.attrs = [attr for attr in obj.attrs if attr not in keys]
            obj.has = {key: value for key, value in obj.has.items()
                       if key not in keys}

    def color(self, rgb):
        self.rgb = rgb
        for obj in self.subclasses:
            obj.color(rgb)

    def plot(self, g):
        kwargs = {'style': 'filled',
                  'fillcolor': self.rgb} if self.rgb else {}
        if self.attrs:
            a = r'\n'.join(self.attrs)
            txt = f'{{{self.name} | {a}}}'
        else:
            txt = self.name
        g.node(self.name, txt, **kwargs)


def plot_graph(figname, nodes, colors={}):
    import graphviz
    g = graphviz.Digraph(node_attr={'shape': 'record'})

    for node in nodes:
        if node.name in colors:
            node.color(colors[node.name])

    for node in nodes:
        print(node)
        node.plot(g)
        for key, value in node.has.items():
            g.edge(node.superclass().name, value.superclass().name, label=key)
        if node.base:
            g.edge(node.base.name, node.name, arrowhead='onormal')

    g.render(figname, format='svg')


def make_figures(render=True):
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

    if 0:
        nodes = create_nodes(A(C()), [B()],
                             lambda obj: obj.__class__.__name__ in 'ABC')
        print(nodes)
        plot_graph('abc', nodes, {'B': '#ffddff'})
        return

    fd = GPAW(mode='fd', txt=None)
    pw = GPAW(mode='pw', txt=None)
    lcao = GPAW(mode='lcao', txt=None)
    a = ase.Atoms('H', cell=[2, 2, 2], pbc=1)

    class Atoms:
        def __init__(self, calc):
            self.calc = calc

    a0 = Atoms(fd)
    fd.get_potential_energy(a)
    pw.get_potential_energy(a)
    lcao.get_potential_energy(a)
    ibzwfs = fd.calculation.state.ibzwfs
    ibzwfs.wfs_qs = ibzwfs.wfs_qs[0][0]

    colors = {'BZPoints': '#ddffdd',
              'PotentialCalculator': '#ffdddd',
              'WaveFunctions': '#ddddff',
              'Eigensolver': '#ffffdd'}

    def include(obj):
        try:
            mod = obj.__module__
        except AttributeError:
            return False

        return mod.startswith('gpaw.new')

    nodes = create_nodes(a0, [pw.calculation.pot_calc], include)
    print(nodes)
    plot_graph('code', nodes, colors)


"""
    ok.add('UniformGridAtomCenteredFunctions')
    subclasses['UniformGridAtomCenteredFunctions'] = (
        'AtomCenteredFunctions (ACF)',
        [pw.calculation.pot_calc.nct_ag], '#ddffff')
    obj = A(1)
    obj.a = fd.calculation.pot_calc.nct_aR
    g = make_graph(obj, skip_first=True)
    if render:
        g.render('acf', format='svg')

    subclasses['UniformGridFunctions'] = (
        'DistributedArrays (DA)',
        [pw.calculation.state.ibzwfs.wfs_qs[0][0].psit_nX], '#eeeeee')
    subclasses['UniformGrid'] = (
        'Domain',
        [pw.calculation.state.ibzwfs.wfs_qs[0][0].psit_nX.desc], '#ddeeff')
    obj = A(1)
    obj.a = fd.calculation.state.ibzwfs.wfs_qs.psit_nX
    g = make_graph(obj, skip_first=True)
    if render:
        g.render('da', format='svg')

    b = []
    for mode in ['fd', 'pw']:
        b.append(builder(a, {'mode': mode}))
    subclasses['FDDFTComponentsBuilder'] = (
        'DFTComponentsBuilder',
        [b[1]], '#ffeedd')
    obj = A(1)
    obj.a = b[0]
    g = make_graph(obj, skip_first=True)
    if render:
        g.render('builder', format='svg')

    ok.add('AtomArrays')
    ok.add('AtomArraysLayout')
    ok.add('AtomDistribution')
    obj = A(1)
    obj.a = AtomArraysLayout([1]).empty()
    g = make_graph(obj, skip_first=True)
"""

if __name__ == '__main__':
    make_figures()
