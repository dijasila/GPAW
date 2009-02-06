from math import pi, sqrt
import numpy as npy
from ase.units import Hartree
from ase.parallel import paropen
from gpaw.utilities import pack, wignerseitz
from gpaw.setup_data import SetupData
from gpaw.gauss import Gauss

import gpaw.mpi as mpi

def print_projectors(setup):
    """Print information on the projectors of input nucleus object.

    If nucleus is a string, treat this as an element name.
    """
    if type(setup) is str:
        setup = SetupData(setup, 'LDA', 'paw')
        n_j = setup.n_j
        l_j = setup.l_j
    else:
        n_j = setup.n_j
        l_j = setup.l_j
    
    angular = [['1'],
               ['y', 'z', 'x'],
               ['xy', 'yz', '3z^2-r^2', 'xz', 'x^2-y^2'],
               ['3x^2y-y^3', 'xyz', '5yz^2-yr^2', '5z^3-3zr^2',
                '5xz^2-xr^2', 'x^2z-y^2z', 'x^3-3xy^2'],
               ]
    print ' i n l m'
    print '--------'
    i = 0
    for n, l in zip(n_j, l_j):
        for m in range(2*l+1):
            if n == -1:
                n = '*'
            print '%2s %s %s_%s' % (i, n, 'spdf'[l], angular[l][m])
            i += 1

def get_angular_projectors(setup, angular, type='bound'):
    """Determine the projector indices which have specified angula
    quantum number.

    angular can be s, p, d, f, or a list of these.
    If type is 'bound', only bound state projectors are considered, otherwise
    all projectors are included.
    """
    # Get the number of relevant j values
    if type == 'bound':
        nj = 0
        while setup.n_j[nj] != -1: nj += 1
    else:
        nj = len(setup.n_j)
            

    # Choose the relevant projectors
    projectors = []
    i = j = 0
    for j in range(nj):
        m = 2 * setup.l_j[j] + 1
        if 'spdf'[setup.l_j[j]] in angular:
            projectors.extend(range(i, i + m))
        j += 1
        i += m

    return projectors

def delta(x, x0, width):
    """Return a gaussian of given width centered at x0."""
    return npy.exp(npy.clip(-((x - x0) / width)**2,
                            -100.0, 100.0)) / (sqrt(pi) * width)

def fold(energies, weights, npts, width):
    """Take a list of energies and weights, and sum a delta function
    for each."""
    emin = min(energies) - 5 * width
    emax = max(energies) + 5 * width
    e = npy.linspace(emin, emax, npts)
    dos_e = npy.zeros(npts)
    for e0, w in zip(energies, weights):
        dos_e += w * delta(e, e0, width)
    return e, dos_e

def raw_orbital_LDOS(paw, a, spin, angular='spdf'):
    """Return a list of eigenvalues, and their weight on the specified atom.

    angular can be s, p, d, f, or a list of these.
    If angular is None, the raw weight for each projector is returned.

    An integer value for ``angular`` can also be used to specify a specific
    projector function.
    """
    wfs = paw.wfs
    w_k = wfs.weight_k
    nk = len(w_k)
    nb = wfs.nbands
    setup = wfs.setups[a]

    energies = npy.empty(nb * nk)
    weights_xi = npy.empty((nb * nk, setup.ni))
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = wfs.collect_eigenvalues(k=k, s=spin)
        u = spin * nk + k
        weights_xi[x:x + nb, :] = w * npy.absolute(wfs.kpt_u[u].P_ani[a])**2
        x += nb

    if angular is None:
        return energies, weights_xi
    elif type(angular) is int:
        return energies, weights_xi[angular]
    else:
        projectors = get_angular_projectors(setup, angular, type='bound')
        weights = npy.sum(npy.take(weights_xi,
                                   indices=projectors, axis=1), axis=1)
        return energies, weights

def all_electron_LDOS(paw, mol, spin, lc=None, wf_k=None, P_aui=None):
    """Returns a list of eigenvalues, and their weights on a given molecule
    
       If wf_k is None, the weights are calculated as linear combinations of
       atomic orbitals using P_uni. lc should then be a list of weights
       for each atom. For example, the pure 2pi_x orbital of a
       molecule can be obtained with lc=[[0,0,0,1.0],[0,0,0,-1.0]]. mol
       should be a list of atom numbers contributing to the molecule.

       If wf_k is not none, it should be a list of wavefunctions
       corresponding to different kpoints and a specified band. It should
       be accompanied by a list of arrays: P_uai=nucleus[a].P_uni for the
       band n and a in mol. The weights are then calculated as the overlap
       of all-electron KS wavefunctions with wf_k"""

    wfs = paw.wfs
    w_k = wfs.weight_k
    nk = len(w_k)
    nb = wfs.nbands
    ns = wfs.nspins
    
    P_un = npy.zeros((nk*ns, nb), npy.complex)

    if wf_k is None:
        P_auni = npy.array([wfs.nuclei[a].P_uni for a in mol])
        if lc is None:
            lc = [[1,0,0,0] for a in mol]
        N = 0
        for atom, w_a in zip(range(len(mol)), lc):
            i=0
            for w_o in w_a:
                P_un += w_o * P_auni[atom,:,:,i]
                N += abs(w_o)**2
                i +=1
        P_un /= sqrt(N)

    else:
        P_aui = [npy.conjugate(P_aui[a]) for a in range(len(mol))]
        for kpt in wfs.kpt_u[spin*nk:(spin+1)*nk]:
            w = npy.reshape(npy.conjugate(wf_k)[kpt.k], -1)
            for n in range(nb):
                psit_nG = npy.reshape(kpt.psit_nG[n], -1)
                P_un[kpt.u][n] = npy.dot(w, psit_nG) * wfs.gd.dv * wfs.a0**1.5
                for a, b in zip(mol, range(len(mol))):
                    atom = wfs.nuclei[a]
                    p_i = atom.P_uni[kpt.u][n]
                    for i in range(len(p_i)):
                        for j in range(len(p_i)):
                            P_un[kpt.u][n] += (P_aui[b][kpt.u][i] *
                                               atom.setup.O_ii[i][j] * p_i[j])
                print n, abs(P_un)[kpt.u][n]**2
                
            print 'Kpoint', kpt.u, ' Sum: ',  sum(abs(P_un[kpt.u])**2)
            
    energies = npy.empty(nb * nk)
    weights = npy.empty(nb * nk)
    x = 0
    for k, w in enumerate(w_k):
        energies[x:x + nb] = wfs.collect_eigenvalues(k=k, s=spin)
        u = spin * nk + k
        weights[x:x + nb] = w * npy.absolute(P_un[u])**2
        x += nb

    return energies, weights

                    
def raw_wignerseitz_LDOS(paw, a, spin):
    """Return a list of eigenvalues, and their weight on the specified atom"""
    wfs = paw.wfs
    gd = wfs.gd
    atom_index = gd.empty(dtype=int)
    atom_ac = paw.atoms.get_scaled_positions() * gd.N_c
    wignerseitz(atom_index, atom_ac, gd.beg_c, gd.end_c)

    w_k = wfs.weight_k
    nk = len(w_k)
    nb = wfs.nbands

    energies = npy.empty(nb * nk)
    weights = npy.empty(nb * nk)
    x = 0
    for k, w in enumerate(w_k):
        u = spin * nk + k
        energies[x:x + nb] = wfs.collect_eigenvalues(k=k, s=spin)
        for n, psit_G in enumerate(wfs.kpt_u[u].psit_nG):
            P_i = wfs.kpt_u[u].P_ani[a][n]
            P_p = pack(npy.outer(P_i, P_i))
            Delta_p = sqrt(4 * pi) * wfs.setups[a].Delta_pL[:, 0]
            weights[x + n] = w * (gd.integrate(npy.absolute(
                npy.where(atom_index == a, psit_G, 0.0))**2)
                                  + npy.dot(Delta_p, P_p))
        x += nb
    return energies, weights


class RawLDOS:
    """Class to get the unfolded LDOS"""
    def __init__(self, calc):
        self.paw = calc
        for setup in calc.wfs.setups.setups.values():
            if not hasattr(setup, 'l_i'):
                # get the mapping
                l_i = []
                for l in setup.l_j:
                    for m in range(2 * l + 1):
                        l_i.append(l)
                setup.l_i = l_i

    def get(self, atom):
        """Return the s,p,d weights for each state"""
        wfs = self.paw.wfs
        nibzkpts = len(wfs.ibzk_kc)
        spd = npy.zeros((wfs.nspins, nibzkpts, wfs.nbands, 3))

        if hasattr(atom, '__iter__'):
            # atom is a list of atom indicies 
            for a in atom:
                spd += self.get(a)
            return spd

        l_i = wfs.setups[atom].l_i
        for kpt in self.paw.wfs.kpt_u:
            if atom in kpt.P_ani:
                for i, P_n in enumerate(kpt.P_ani[atom].T):
                    spd[kpt.s, kpt.k, :, l_i[i]] += npy.abs(P_n)**2

        wfs.gd.comm.sum(spd)
        wfs.kpt_comm.sum(spd)
        return spd

    def by_element(self):
        # get element indicees
        elemi = {}
        for i,a in enumerate(self.paw.atoms):
            symbol = a.symbol
            if elemi.has_key(symbol):
                elemi[symbol].append(i)
            else:
                elemi[symbol] = [i]
        for key in elemi:
            elemi[key] = self.get(elemi[key])
        return elemi

    def by_element_to_file(self, 
                           filename='ldos_by_element.dat',
                           width=None,
                           shift=True):
        """Write the LDOS by element to a file

        If a width is given, the LDOS will be Gaussian folded and shifted to set 
        Fermi energy to 0 eV. The latter can be avoided by setting shift=False. 
        """
        ldbe = self.by_element()

        f = paropen(filename,'w')

        wfs = self.paw.wfs
        if width is None:
            # unfolded ldos
            eu = '[eV]'
            print >> f, '# e_i' + eu + '  spin  kpt     n   kptwght',
            for key in ldbe:
                if len(key) == 1: key=' '+key
                print  >> f, ' '+key+':s     p        d      ',
            print  >> f,' sum'
            for k in range(wfs.nibzkpts):
                for s in range(wfs.nspins):
                    e_n = self.paw.get_eigenvalues(kpt=k, spin=s)
                    if e_n is None:
                        continue
                    w = wfs.weight_k[k]
                    for n in range(wfs.nbands):
                        sum = 0.0
                        print >> f, '%10.5f %2d %5d' % (e_n[n], s, k), 
                        print >> f, '%6d %8.4f' % (n, w),
                        for key in ldbe:
                            spd = ldbe[key][s, k, n]
                            for l in range(3):
                                sum += spd[l]
                                print >> f, '%8.4f' % spd[l],
                        print >> f, '%8.4f' % sum
        else:
            # folded ldos

            gauss = Gauss(width)

            # minimal and maximal energies
            emin = 1.e32
            emax = -1.e32
            for k in range(wfs.nibzkpts):
                for s in range(wfs.nspins):
                    e_n = self.paw.get_eigenvalues(kpt=k, spin=s,
                                                   broadcast=True)
                    emin = min(e_n.min(), emin)
                    emax = min(e_n.max(), emax)
            emin -= 4 * width
            emax += 4 * width

            # Fermi energy
            try:
                efermi = self.paw.get_fermi_level()
            except:
                # set Fermi level half way between HOMO and LUMO
                hl = self.paw.occupations.get_homo_lumo(wfs.kpt_u)
                efermi = (hl[0] + hl[1]) * Hartree / 2

            eshift = 0.0
            if shift:
                eshift = -efermi

            # set de to sample 4 points in the width
            de = width / 4
            
            for s in range(wfs.nspins):
                print >> f, '# Gauss folded, width=%g [eV]' % width
                if shift:
                    print >> f, '# shifted to Fermi energy = 0'
                    print >> f, '# Fermi energy was', 
                else:
                    print >> f, '# Fermi energy',
                print  >> f, efermi, 'eV'
                print >> f, '# e[eV]  spin ',
                for key in ldbe:
                    if len(key) == 1: key=' '+key
                    print  >> f, ' '+key+':s     p        d      ',
                print  >> f

                # loop over energies
                emax=emax+.5*de
                e=emin
                while e<emax:
                    val = {}
                    for key in ldbe:
                        val[key] = npy.zeros((3))
                    for k in range(wfs.nibzkpts):
                        w = wfs.weight_[k]
                        e_n = self.paw.get_eigenvalues(kpt=k, spin=s,
                                                       broadcast=True)
                        for n in range(wfs.nbands):
                            w_i = w * gauss.get(e_n[n] - e)
                            for key in ldbe:
                                val[key] += w_i * ldbe[key][s, k, n]

                    print >> f, '%10.5f %2d' % (e + eshift, s), 
                    for key in val:
                        spd = val[key]
                        for l in range(3):
                            print >> f, '%8.4f' % spd[l],
                    print >> f
                    e += de
                            

        f.close()
