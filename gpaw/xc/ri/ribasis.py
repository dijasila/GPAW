from gpaw.basis_data import Basis, BasisFunction
from collections import defaultdict


class RIBasis(Basis):
    def __init__(self, symbol, name, readxml=True, rgd=None, world=None):
        Basis.__init__(self, symbol, name, readxml, rgd, world)
        self.ribf_j = []

    @property
    def nrio(self):
        return sum([2 * ribf.l + 1 for ribf in self.ribf_j])

    def append_ri(self, ribf):
        self.ribf_j.append(ribf)

    def write(self, out):
        Basis.write(self, out)
        for ribf in self.ribf_j:
            out(ribf.xml(indentation='  '))

    def ritosplines(self):
        return [self.rgd.spline(ribf.phit_g, ribf.rc, ribf.l, points=400)
                for ribf in self.ribf_j]
   
    def get_description(self):
        desc = Basis.get_description(self)
        desc += f'\n  Number of RI-basis functions {self.nrio}'
        
        ribf_lines = []
        for ribf in self.ribf_j:
            line = '\n    l=%d %s' % (ribf.l, ribf.type)
            ribf_lines.append(line)
        desc += '\n'.join(ribf_lines)

        return desc

    def generate_ri_basis(self, accuracy):
        lmax = 4

        # TODO: Hartree
        def poisson(n_g, l):
            return Hartree(self.rgd, n_g, l)

        # Auxiliary basis functions per angular momentum channel
        auxt_lng = defaultdict(lambda: [])
        # The Coulomb (or screened coulomb) solution to the basis functions,
        # truncated to the maximum extent of the original basis function.
        # i.e. not the fulll potential extending to infinity.
        # wauxt_lng = defaultdict(lambda: [])

        def add(aux_g, l):
            ribf = BasisFunction(None, l, None, aux_g, type='auxiliary')
            self.append_ri(ribf)
            # auxt_lng[l].append(aux_g)
            # v_g = poisson(aux_g, l)
            # wauxt_lng[l].append(v_g)

        def basisloop():
            for j, bf in enumerate(self.bf_j):
                yield j, bf.l, bf.phit_g

        # Double basis function loop to create product orbitals
        for j1, l1, phit1_g in basisloop():
            for j2, l2, phit2_g in basisloop():
                # Loop only over ordered pairs
                if j1 > j2:
                    continue

                # Loop over all possible angular momentum states what the
                # product l1 x l2 creates.
                for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                    if l > lmax:
                        continue

                    add(phit1_g * phit2_g, l)

        for l, auxt_ng in auxt_lng.items():
            print(l, auxt_ng)
            print(f'    l={l}')
            for n, auxt_g in enumerate(auxt_ng):
                print(f'        {n}')
        # Auxiliary basis functions
        # setup.auxt_j, setup.wauxt_j, setup.sauxt_j, setup.wsauxt_j,
        # setup.M_j = \
        #    get_auxiliary_splines_screened(setup,
        #         self.lmax, rcmax, threshold=self.threshold)
        print(self.get_description())


def Hartree(rgd, n_g, l):
    v_g = rgd.poisson(n_g, l)
    v_g[1:] /= rgd.r_g[1:]
    v_g[0] = v_g[1]
    return v_g


"""
def _get_auxiliary_splines(setup, lmax, cutoff, poisson, threshold=1e-2):
    rgd = setup.rgd
    print('Threshold: %.10f' % threshold)

    for j1, spline1 in enumerate(setup.phit_j):
        l1 = spline1.get_angular_momentum_number()
        for j2, spline2 in enumerate(setup.phit_j):
            if j1 > j2:
                continue
            l2 = spline2.get_angular_momentum_number()
            for l in range((l1 + l2) % 2, l1 + l2 + 1, 2):
                if l > 2:
                    continue
                aux_g = spline_to_rgd(rgd, spline1, spline2)
                add(aux_g, l)

    if setup.Z == 1:
        add(np.exp(-1.2621205398*rgd.r_g**2),1)
        add(np.exp(-0.50199775874*rgd.r_g**2),1)
        add(np.exp(-0.71290724024*rgd.r_g**2),2)
        # add(np.exp(-1.6565726132*rgd.r_g**2),3)

    # Splines
    auxt_j = []
    wauxt_j = []
    sauxt_j = []
    wsauxt_j = []
    M_j = []

    #g_lg = shape_functions(rgd, 'gauss',  0.38 , lmax) # XXX Hard coded

    for l, auxt_ng in auxt_lng.items():
        auxt_ng = np.array(auxt_ng)
        wauxt_ng = np.array(wauxt_lng[l])
        N = len(auxt_ng)
        S_nn = np.zeros( (N, N) )
        #import matplotlib.pyplot as plt
        for n1, auxt_g in enumerate(auxt_ng):
            #plt.plot(rgd.r_g, auxt_g)
            #plt.plot(rgd.r_g, wauxt_ng[n1],'x')
            for n2, wauxt_g in enumerate(wauxt_ng):
                S_nn[n1, n2] = rgd.integrate(auxt_g * wauxt_g)
        S_nn = (S_nn + S_nn.T) / 2

        print('l=%d' % l, S_nn)
        eps_i, v_ni = eigh(S_nn)
        print(eps_i)
        assert np.all(eps_i>-1e-10)
        nbasis = int((eps_i > threshold).sum())
        q_ni = np.dot(v_ni[:, -nbasis:],
                      np.diag(eps_i[-nbasis:]**-0.5))

        #plt.show()

        auxt_ig =  q_ni.T @ auxt_ng
        wauxt_ig = q_ni.T @ wauxt_ng

        g_g = spline_to_rgd(rgd, setup.ghat_l[l])

        # Evaluate reference multipole momement
        Mref = rgd.integrate(g_g * rgd.r_g**l) / (4*np.pi)

        for i in range(len(auxt_ig)):
            auxt_g = auxt_ig[i]
            auxt_j.append(rgd.spline(auxt_g, cutoff, l, 500))
            wauxt_j.append(rgd.spline(wauxt_ig[i], cutoff, l, 500))

            # Evaluate multipole moment
            if l <= 2:
                M = rgd.integrate(auxt_g * rgd.r_g**l) / (4*np.pi)
                M_j.append(M)
                sauxt_g = auxt_g - M / Mref * g_g
            else:
                M_j.append(0.0)

            sauxt_j.append(rgd.spline(sauxt_g, cutoff, l, 500))

            v_g = poisson(sauxt_g, l)
            wsauxt_j.append(rgd.spline(v_g, cutoff, l, 500))
            print('Last potential element', v_g[-1])
            assert(np.abs(v_g[-1])<1e-6)
        print('l=%d %d -> %d' % (l, len(auxt_ng), len(auxt_ig)))
    return auxt_j, wauxt_j, sauxt_j, wsauxt_j, M_j
    """
