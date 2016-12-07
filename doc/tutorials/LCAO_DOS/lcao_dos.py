from gpaw import GPAW, FermiDirac
import numpy as np
import pickle
from ase import Atoms
from gpaw.utilities.dos import LCAODOS, RestartLCAODOS, fold
from ase.io import *
import matplotlib.pyplot as plt
from ase.units import Hartree
import matplotlib

matplotlib.rc('font', family='normal', serif='cm10')

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

font = {'family' : 'normal',
        'size'   : 12}
matplotlib.rc('font', **font)

compound = 'GaAs'

atoms = Atoms('GaAs', scaled_positions = [[ 0.25,  0.25,  0.25], [ 0.,    0.,    0.  ]], cell = [[ 4.0248,      0.,          0.        ], [ 2.0124,      3.48557905,  0.        ], [ 2.0124,      1.16185968,  3.28623544]], pbc = True)


calc= GPAW(mode='lcao',
            #h = 0.14,
            basis='dzp',
            xc='PBEsol',
            parallel={'band':1},
            kpts=(5,5,5),
            txt = compound+'_LCAO.out',
            occupations=FermiDirac(width=0.01))
atoms.set_calculator(calc)
en = atoms.get_potential_energy()
calc.write(compound+'_LCAO.gpw')


dos = RestartLCAODOS(calc)
homo, lumo = calc.get_homo_lumo()
e = atoms.get_chemical_symbols()
elements = np.unique(e)

sh = {}
l_no = {}
des = {}

for ind,atom in enumerate(elements):
	sz = calc.wfs.setups[ind].basis.get_l_numbers()
	d = [sz[i][1].split()[0].split('-')[0] for i in range(len(sz))]	
	ud = []
	[ud.append(item) for item in d if item not in ud]
	print ud
	l = [sz[i][0] for i in range(len(sz))]
	l = np.asarray(np.unique(l))
	print l, sz
	
	coeff_A = []
        for an in l:
            ind = 0
            l_value = []
            for j in range(len(l)):
                if int(sz[j][0]) == an:
                    for i in range(2*an+1):
                        l_value.append(i+ind)
                ind = ind + 2*sz[j][0]+1
            if len(l_value)> 0:
                coeff_A.append(l_value)	
	sh[atom]= coeff_A
	l_no[atom] = l
	des[atom] =ud

npts = 100
width = 0.1

m_all =np.array(())
ls = np.array(())

projections = []
description = []
des_uni = []

for i in range(len(e)):
    indices_atom = dos.get_atom_indices(i)
    print indices_atom
    print 'indices_atom'
    m_all = np.concatenate((m_all,indices_atom),axis=0)
    sz_indices = sh[e[i]] # Which atom
    ls = np.concatenate((ls,l_no[e[i]]),axis=0)
        

    for ii, spd in enumerate(sz_indices): # which orbital
	print 'spd'
	print spd
	M = []
	for x in spd:
            M.append(indices_atom[x])
	description.append(e[i]+'-'+des[e[i]][ii])

    	print 'M'
    	print M
	projections.append(M)
print description
print projections



for state in np.unique(description):
    stp = []
    ss = 0
    for state2 in description:
	if state == state2:
	    stp = np.concatenate((stp,projections[ss]))
	ss = ss +1
    print stp
    e, w = dos.get_subspace_pdos(map(int,stp))
    en_bl, wt = fold(energies=e*Hartree, weights=w, npts=npts, width=width)
    plt.plot(en_bl-homo, wt, label=r'\textbf{%s}'%(state), lw=1)	

en_1, dos_bl1 = dos.get_subspace_pdos(map(int,m_all))
en_2, dos_bl = fold(energies=en_1*Hartree,weights=dos_bl1, npts =npts, width=width)
plt.plot(en_2-homo, dos_bl, label=r'\textbf{Total DOS}', lw=1.2, color='k')


plt.legend()
plt.ylabel('DOS')
plt.xlabel('E[eV]')
plt.xlim(-5,5)
plt.save('dos_GaAs.png')

