# -*- coding: utf-8 -*-
import os
import pickle
from sys import argv

from ase.atoms import string2symbols
from ase.data import atomic_numbers, atomic_names
from ase.data.molecules import extra, g2, rest
from ase.data.molecules import data as molecule_data

from gpaw.atom.generator import Generator, parameters
from gpaw.atom.analyse_setup import analyse
from gpaw.testing.dimer import TestAtom

bulk = {'Ti': ()}

def make(symbol):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]

    if not os.path.isfile(symbol + '.pckl'):
        g = Generator(symbol, 'PBE', scalarrel=True)
        g.run(logderiv=True, **parameters[symbol])
        tables = analyse(g, show=False)

        t = TestAtom(symbol)
        t.run(True, False)
        hmin, B, n = t.summary(show=False)

        molecules = []
        for x in extra + g2:
            if symbol in string2symbols(molecule_data[x]['symbols']):
                molecules.append(molecule_data[x]['name'])
        mols = ''
        if molecules:
            names = [rest(m) for m in molecules]
            if len(molecules) == 1:
                mols = names[0]
            elif len(molecules) > 5:
                mols = ', '.join(names[:5]) + ' ...'
            else:
                mols = ', '.join(names[:-1]) + ' and ' + names[-1]
        pickle.dump((tables, hmin, B, n, mols), open(symbol + '.pckl', 'w'))
    else:
        tables, hmin, B, n, mols = pickle.load(open(symbol + '.pckl'))
    
    f = open(symbol + '.rst', 'w')

    f.write(""".. _%s:

.. index:: %s

================
%s
================

.. default-role:: math

""" % (name, name, name))

    f.write("""
Bulk tests
==========

XXX

""" % ())

    if mols:
        f.write("""
Molecule tests
==============

See tests for %s in :ref:`molecule_tests`.


""" % mols)

    if name[0] in 'AEIOUY':
        aname = 'an ' + name.lower()
    else:
        aname = 'a ' + name.lower()
    f.write(u"""
Convergence tests
=================

The energy af %s dimer (`E_d`) and %s atom (`E_a`) is
calculated at diferent grid-spacings (`h`).

.. image:: ../_static/%s-dimer-eggbox.png

A fit to `E_a` for `h` between %.2f Å and 0.20 Å gives:

.. math::  E_a \simeq A + B (h / h_0)^n,

with *B* = %.2f eV and *n* = %.2f (`h_0` = 0.20 Å).

""" % (aname, aname, symbol, hmin, B, n))

    tables = ['\n'.join(t) for t in tables]
    f.write(u"""
Setup details
=============

Electrons
---------

%s

Cutoffs
-------

%s

Energy Contributions
--------------------

%s

Core and valence States
-----------------------

%s

Wave functions, projectors, ...
-------------------------------

.. image:: ../_static/%s-setup.png



Back to :ref:`setups`.

.. default-role::

""" % (tuple(tables) + (symbol,)))


if len(argv) == 1:
    symbols = parameters.keys()
else:
    symbols = argv[1:]
for symbol in symbols:
    if not os.path.isfile(symbol + '.rst') and os.path.isdir(symbol):
        make(symbol)
