# creates: H.rst, H.png, He.rst, He.png
# ... and all the rest
from __future__ import print_function
import matplotlib.pyplot as plt
import ase.db
from ase.data import atomic_numbers, atomic_names
from ase.units import Hartree
from ase.utils import plural

from gpaw.atom.check import summary, solve, cutoffs, all_names

con = ase.db.connect('datasets.db')


def make_rst(symbol):
    Z = atomic_numbers[symbol]
    name = atomic_names[Z]
    
    rst = """\
.. Computer generated reST (make_setup_pages.py)
.. index:: {name}
.. _{name}:
    
================
{name}
================

"""
    
    data = sorted((row.name, row.data['nlfer'])
                  for row in con.select(symbol, test='dataset'))
    if len(data) > 1:
        rst += '.. contents::\n\n'
        
    for dataset, nlfer in data:
        table = ''
        nv = 0
        for n, l, f, e, rcut in nlfer:
            n, l, f = (int(x) for x in [n, l, f])
            if n == -1:
                n = ''
            table += '    {}{},{},{:.3f},'.format(n, 'spdf'[l], f, e * Hartree)
            if rcut:
                table += '{:.2f}'.format(rcut)
                nv += f
            table += '\n'

        rst += """
{electrons}
====================

Radial cutoffs and eigenvalues:
    
.. csv-table::
    :header: id, occ, eig [eV], cutoff [Bohr]
    
{table}
        
The figure shows convergence of the absolule energy (red line)
and atomization energy (green line) of a {symbol} dimer relative
to completely converged numbers (plane-wave calculation at 1500 eV).
Also shown are finite-difference and LCAO (dzp) calculations at gridspacings
0.17 Å and 0.20 Å.

.. image:: {symbol}.png

Egg-box errors in finite-difference mode:
    
.. csv-table::
    :header: grid-spacing [Å], energy error [eV]
    
{table2}
"""
        epw, depw, efd, defd, elcao, delcao, deegg = summary(con, dataset)
        
        table2 = ''
        for h, e in zip([0.16, 0.18, 0.2], deegg):
            table2 += '    {:.2f},{:.4f}\n'.format(h, e)
            
        plt.semilogy(cutoffs[:-1], epw[:-1], 'r',
                     label='pw, absolute')
        plt.semilogy(cutoffs[:-1], depw[:-1], 'g',
                     label='pw, atomization')
        plt.semilogy([solve(epw, de) for de in efd], efd, 'rs',
                     label='fd, absolute')
        plt.semilogy([solve(depw, de) for de in defd], defd, 'gs',
                     label='fd, atomization')
        plt.semilogy([solve(epw, de) for de in elcao], elcao, 'ro',
                     label='lcao, absoulte')
        plt.semilogy([solve(depw, de) for de in delcao], delcao, 'go',
                     label='lcao, atomization')
        plt.xlabel('plane-wave cutoff [eV]')
        plt.ylabel('error [eV/atom]')
        plt.legend(loc='best')
        plt.savefig(symbol + '.png')

    with open(symbol + '.rst', 'w') as fd:
        fd.write(rst.format(electrons=plural(nv, 'valence electron'),
                            table=table, table2=table2, symbol=symbol,
                            name=name))

        
for symbol in all_names:
    if '.' not in symbol:
        make_rst(symbol)
    break
