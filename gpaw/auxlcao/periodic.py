
class RIPeriodic:
    def __init__(self, name, exx_fraction, screening_omega, N_k = (3,3,3)):
        self.name = name
        self.exx_fraction = exx_fraction
        self.screening_omega = screening_omega
        self.fix_rho = False

        self.sdisp_Rc = []
        for x in range(-N_k[0]//2, N_k[0]//2+1):
            for y in range(-N_k[0]//2, N_k[0]//2+1):
                for z in range(-N_k[0]//2, N_k[0]//2+1):
                    self.sdisp_Rc.append( (x,y,z) )
        print('Real space cell displacements', self.sdisp_Rc)

    def initialize(self, density, hamiltonian, wfs):
        self.density = density
        self.hamiltonian = hamiltonian
        self.wfs = wfs
        self.timer = hamiltonian.timer
        self.prepare_setups(density.setups)

    def prepare_setups(self, setups):
        if self.screening_omega != 0.0:
            print('Screening omega for setups')
            for setup in setups:
                setup.ri_M_pp = setup.calculate_screened_M_pp(self.screening_omega)
                setup.ri_X_p = setup.HSEX_p
        else:
            for setup in setups:
                setup.ri_M_pp = setup.M_pp
                setup.ri_X_p = setup.X_p


    def nlxc(self,
            H_MM:np.ndarray,
             dH_asp:Dict[int,np.ndarray],
             wfs,
             kpt, yy) -> Tuple[float, float, float]:

        #H_MM += kpt.exx_V_MM * yy
        print('RI periodic not doing anything at nlxc')
        pass

    def set_positions(self, spos_ac):
        self.spos_ac = spos_ac

    def calculate_non_local(self):
        ekin = -2*evv
        return evv, ekin

    def calculate_paw_correction(self, setup, D_sp, dH_sp=None, a=None):
        with self.timer('RI Local atomic corrections'):
            D_ii = unpack2(D_sp[0]) / 2 # Check 1 or 2
            ni = len(D_ii)
            V_ii = np.empty((ni, ni))
            for i1 in range(ni):
                for i2 in range(ni):
                    V = 0.0
                    for i3 in range(ni):
                        p13 = packed_index(i1, i3, ni)
                        for i4 in range(ni):
                            p24 = packed_index(i2, i4, ni)
                            V += setup.ri_M_pp[p13, p24] * D_ii[i3, i4]
                    V_ii[i1, i2] = +V*2 #XXX
            V_p = pack2(V_ii)
            dH_sp[0][:] += (-V_p - self.density.setups[a].ri_X_p) * self.exx_fraction

            evv = -self.exx_fraction * np.dot(V_p, self.density.D_asp[a][0]) / 2
            evc = -self.exx_fraction * np.dot(self.density.D_asp[a][0], self.density.setups[a].ri_X_p)
            return evv + evc, 0.0


