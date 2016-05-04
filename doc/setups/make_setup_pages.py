# creates: H.rst, H.png, He.rst, He.png
# ... and all the rest
from __future__ import print_function
import os
import sys

import numpy as np
from ase.data import atomic_numbers, atomic_names

con = ase.db.connect('dataset.db')


def make_rst(symbol):
    Z = atomic_numbers[symbol]
    atomname = atomic_names[Z]
    rst = ['.. Computer generated reST (make_setup_pages.py)',
           '.. index:: ' + atomname,
           '.. _{}:'.format(atomname),
           '',
           '================',
           atomname,
           '================']
    
    name_and_data = sorted((row.name, row.data)
                           for row in con.select(symbol, test='dataset'))
    if len(name_and_data) > 1:
        rst += ['.. contents::', '']
        
    for name, data in name_and_data:
        for n, l, f, e, rcut in data['nlfer']:
            if n == -1:
                n = '\*'
            table += '%2s%s  %3d  %10.3f Ha' % (n, 'spdf'[l], f, e)
            if rcut:
                table += '  %.2f Bohr\n' % rcut
            else:
                table += '\n'
                
        rst += ['.. csv-table::',
                '    :header: id, occ, eigenvals, cutoff',
                '']

The energy of %(aname)s dimer (`E_d`) and %(aname)s atom (`E_a`) is
calculated at different grid-spacings (`h`).

.. image:: ../static/setups-data/%(symbol)s-dimer-eggbox.png


Setup details
=============

The setup has %(Nv)d valence electron%(plural)s and %(Nc)d electrons
in the frozen core.  It is based on a scalar-relativistic spin-paired
neutral all-electron PBE calculation.

Cutoffs and eigenvalues:

===  ===  =============  ==========

---  ---  -------------  ----------
%(table)s
===  ===  =============  ==========




    import pylab as plt
    dpi = 80
    fig = plt.figure(figsize=(6, 11), dpi=dpi)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.04)

    plt.subplot(311)
    plt.semilogy(h[:-1], 2 * abs(Eegg[:-1, 0] - Eegg[-1, 0]), '-o',
                 label=r'$2E_a$')
    plt.semilogy(h[:-1], abs(Edimer0[:-1] - Edimer0[-1]), '-o',
                 label=r'$E_d$')
    plt.semilogy(h[:-1], abs(Ediss[:-1] - Ediss[-1]), '-o',
                 label=r'$2E_a-E_d$')
    plt.semilogy(h, Eegg.ptp(axis=1), '-o', label=r'$E_{egg}$')
    #plt.title('Energy differences')
    plt.xlabel(r'h [\AA]')
    plt.ylabel('energy [eV]')
    plt.legend(loc='best')

    plt.subplot(312)
    plt.semilogy(h, np.abs(data['Fegg']).max(axis=1), '-o',
                 label=r'$|\mathbf{F}_{egg}|$')
    plt.semilogy(h, np.abs(data['Fdimer'].sum(axis=2)).max(axis=1), '-o',
                 label=r'$|\mathbf{F}_1 + \mathbf{F}_2|$')
    #plt.title('Forces')
    plt.xlabel(r'h [\AA]')
    plt.ylabel(r'force [eV/\AA]')
    plt.legend(loc='best')

    plt.subplot(313)
    d = ddimer0[-1]
    plt.plot(h, ddimer0, '-o')
    #plt.title('Bond length')
    plt.xlabel(r'h [\AA]')
    plt.axis(ymin=d * 0.98, ymax=d * 1.02)
    plt.ylabel(r'bond length [\AA]')

    plt.savefig('../static/setups-data/%s-dimer-eggbox.png' % symbol, dpi=dpi)

    with open(symbol + '.rst', 'w') as fd:
        fd.write('\n'.join(rst))
    
