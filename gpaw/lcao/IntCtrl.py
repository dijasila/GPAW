import numpy as npy


class IntCtrl:
    """
    Parameters:
        kt             // Temperature : default = 0 K
        leadfermi(nLead)  // Fermi Energy of each Lead with biasV
        maxfermi, minfermi   // default = max(Fermi) & min(Fermi)    
        bias               // >0 for ul>u_r
        eqintpath,  eqinttol,  eqdelta,  eqresz     // eqInt Ctrl
        locintpath, lcointtol, locdelta, locresz    // locInt Ctrl

        neintmethod  // 0 - Manual Method : Linear (default)// 1 - Auto Method
        neintstep    // used in Manual Method, defalut = 1e-2
        neinttol     // used in Auto Method, default = 1e-3
        neintpath    // [ minfermi leadfermi maxfermi ] + eta ( 1e-8 )}
    """
    
    def __init__(self, kt, efermi, bias, eqintpath = None, eqinttol = None,
                 locintpath = None, locinttol = None, neintmethod = 0,
                 neintstep = 0, neintpath = None, neinttol = None):
        
        #if u_l>u_r,bias>0
        self.kt = kt
        self.leadfermi = [efermi + bias / 2, efermi - bias / 2]
        self.minfermi = min(efermi + bias / 2, efermi - bias / 2)
        self.maxfermi = max(efermi + bias / 2, efermi - bias / 2)
        self.eqinttol = 1e-8
        self.kttol = 1e-5
        self.biastol = 1e-10
        
        #eq-Integral Path : 
        #default = [-100  -100+20i  10i  1e-8i] + minfermi
        #        = [-100  -100+20i  10i-20kt  5i*2*pi*kt-20kt
        #           5i*2*pi*kt+20kt] + minfermi
        
        if self.kt < self.kttol: #T=0K
            self.eqintpath = [-120, -120 + 20.j, 10.j, 1e-8j]
            self.eqdelta = 0
            self.eqresz = []
            print '--eqIntCtrl:  Tol =', self.eqinttol
        else:        #T>0K
            nkt = 20 * self.kt
            dkt = 10 * npy.pi * self.kt
            self.eqintpath = [-350, -350 + 20.j, 10.j - nkt, dkt * 1.j - nkt,
                              dkt * 1.j + nkt]
            self.eqdelta = dkt
            nRes = 10
            if abs( nRes - (npy.round((nRes - 1) / 2) * 2 + 1)) < 1e-3 :
                print 'Warning: Residue Point too close to IntPath!'
            self.eqresz = range(1, nRes, 2)
            for i in range(len(self.eqresz)):
                self.eqresz[i] *=  1.j * npy.pi * self.kt
            
            print '--eqIntCtrl: Tol = ', self.eqinttol, 'Delta =', \
                                          self.eqdelta, ' nRes =', self.eqresz

        for i in range(len(self.eqintpath)):
            self.eqintpath[i] = self.eqintpath[i] + self.minfermi
            
        for i in range(len(self.eqresz)):
            self.eqresz[i] = self.eqresz[i] + self.minfermi        

        # read bound-Integral Path : 
        # default = [ minfermi+1e-8i   minfermi+1i
        #           maxfermi+1i   maxfermi+1e-8i ] 
        #         = [ minfermi-20kt+5i*2*pi*kt   maxfermi+20kt+5i*2*pi*kt ]
        #         = [ minfermi-20kt+5i*2*pi*kt   Mid+1i
        #            maxfermi+20kt+5i*2*pi*kt ]

        self.locinttol = self.eqinttol
        if (self.maxfermi - self.minfermi)< self.biastol:
            self.locPath = None
            self.locdelta = 0
            self.locresz = 0
            print '--locInt: None'
        elif self.kt < self.kttol: #T=0K
            self.locPath = [self.minfermi + 1e-8j, self.minfermi + 1.j,
                            self.maxfermi + 1.j, self.maxfermi + 1e-8j]
            self.locdelta = 0
            self.locresz = 0
            print '--locInt: Tol', self.lcointtol    
        else:
            nkt = 20 * self.kt
            dkt = 10 * npy.pi * self.kt

            if self.maxfermi-self.minfermi < 0.2 + 2 * nkt or dkt > 0.5:
                self.locintpath = [self.minfermi - nkt + dkt * 1.j,
                                   self.maxfermi + nkt + dkt * 1.j]
            else:
                self.locintpath = [self.minfermi - nkt + dkt * 1.j,
                                   self.maxfermi + nkt + dkt * 1.j,
                                   (self.maxfermi + self.minfermi) / 2 + 1.j,
                                   self.maxfermi - nkt + dkt * 1.j,
                                   self.maxfermi + nkt + dkt * 1.j]
            self.locdelta = dkt
            nRes = 10
            self.locresz = npy.array(range(1, nRes, 2)
                                                   ) * 1.j * npy.pi * self.kt
            tmp = len(range(1, nRes, 2))
            self.locresz = npy.resize(self.locresz, [2, tmp])
            for i in range(tmp):
                self.locresz[0][i] += self.minfermi
                self.locresz[1][i] += self.maxfermi
            print '--locInt: Tol =', self.locinttol, 'Delta =', \
                               self.locdelta, 'nRes=', len(self.locresz[0])

        #ne-Integral Path : 
        # default = [ minfermi  leadfermi  maxfermi ] 
        # IntMethod = Manual Method
        # -------------------------------------------------- 
        # -- Integral Method -- 
        self.neinttol = 1e-3        
        self.neintmethod= 1 # 0: Linear 1: Auto

        # -- Integral Step--

        self.neintstep = 1e-2                    
        
        # -- Integral Path --

        if self.maxfermi -self.minfermi < self.biastol:
            self.neintpath = []
        elif self.kt < self.kttol : #T=0K
            self.neintpath = [self.minfermi, self.maxfermi]
        else :
            nkt = 20 * kt
            self.neintpath = [self.minfermi - nkt, self.maxfermi + nkt]

        # -- Integral eta --

        for i in range(len(self.neintpath)):                         
            self.neintpath[i] += 1e-8j 

        if len(self.neintpath) == 0:
            print ' --neInt: None'
        elif self.neintmethod == 0:
            print ' --neInt: ManualEp -> Step=', self.neintstep, 'Eta =', \
                                              npy.imag(self.neintpath[0])
        elif self.neintmethod == 1:
            print ' --neInt: AutoEp   -> Tol =', self.neinttol,  'Eta =', \
                                              npy.imag(self.neintpath[0])

        
            

