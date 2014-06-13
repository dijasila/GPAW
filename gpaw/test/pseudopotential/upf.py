from gpaw.upf import UPFSetupData, upfplot
from gpaw.pseudopotential import pseudoplot, PseudoPotential

def get(fname):
    print 'load', fname
    s = UPFSetupData(fname)
    print 'loaded ok'
    return s

s = get('H.pz-hgh.UPF')
#upfplot(s.data)
bfs = s.create_basis_functions()
pp = PseudoPotential(s, bfs)
#pseudoplot(pp)

from gpaw.atom.atompaw import AtomPAW

if 1:
    f = 1.0 #1e-12
    c = AtomPAW('H', [[[f]]],
                charge=1 - f,
                h=0.03,
                #setups='paw',
                #setups='hgh',
                setups={'H': s}
                )

    import pylab as pl
    #pl.plot(c.wfs.gd.r_g, c.wfs.gd.r_g * c.hamiltonian.vbar_g)
    #pl.plot(c.wfs.gd.r_g, c.wfs.gd.r_g * c.hamiltonian.vt_sg[0])

    pl.plot(c.wfs.gd.r_g, c.density.rhot_g)
    pl.plot(c.wfs.gd.r_g, c.density.nt_sg.sum(axis=0))
    pl.plot(c.wfs.gd.r_g, c.wfs.gd.r_g * c.hamiltonian.vt_sg.sum(axis=0))
    pl.show()

    #b = c.extract_basis_functions()
    #from gpaw.basis_data import BasisPlotter

    #bp = BasisPlotter(premultiply=False, show=True)
    #bp.plot(b)

    raise SystemExit

#print 'test v201 Au.pz-d-hgh.UPF'
#s = UPFSetupData('Au.pz-d-hgh.UPF')
#print 'v201 ok'

#print 'test horrible version O.pz-mt.UPF'
#s = UPFSetupData('O.pz-mt.UPF')
#print 'horrible version ok, relatively speaking'

if 1:
    from ase import Atoms
    from gpaw import GPAW, PoissonSolver
    from gpaw.utilities import h2gpts

    #s = UPFSetupData('/home/askhl/parse-upf/h_lda_v1.uspp.F.UPF')

    system = Atoms('H')
    system.center(vacuum=4.5)
    calc = GPAW(#setups='hgh', 
                setups={'H': s},
                #charge=1-1e-12,
                eigensolver='rmm-diis',
                gpts=h2gpts(0.15, system.get_cell(), idiv=8),
                poissonsolver=PoissonSolver(relax='GS', eps=1e-7),
                xc='oldLDA',
                nbands=4)

    system.set_calculator(calc)
    system.get_potential_energy()
