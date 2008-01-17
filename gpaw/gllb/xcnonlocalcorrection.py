from gpaw.gllb.sphericalexpander import SphericalExpander
from gpaw.gllb.gllb import find_nucleus
from gpaw.mpi import world

class DummyXC:
    def set_functional(self, xc):
        pass

class XCNonLocalCorrection:
    def __init__(self,
                 xcfunc, # radial exchange-correlation object
                 w_j,    # wave functions
                 wt_j,   # smooth wavefunctions
                 nc,     # core density
                 nct,    # smooth core density
                 rgd,    # radial grid edscriptor
                 jl,     # jl-indices
                 lmax,   # maximal angular momentum to consider
                 Exc0,   # Exc contribution already taken in to account in setups
                 extra_xc_data): # The response parts of core orbitals


        # Some part's of code access xc.xcfunc.hydrid, this is to ensure
        # that is does not cause error
        self.xc = DummyXC()
        self.xc.xcfunc = DummyXC()
        self.xc.xcfunc.hybrid = 0.0
        
        self.extra_xc_data = extra_xc_data
        self.gllb_xc = xcfunc
        self.Exc0 = Exc0
        self.sphere_n = SphericalExpander(rgd, lmax, jl, w_j, nc)
        self.sphere_nt = SphericalExpander(rgd, lmax, jl, wt_j, nct)

    def calculate_energy_and_derivatives(self, D_sp, H_sp, a):
        # Use special method to find nucleus in parallel calculations
        nucleus = find_nucleus(self.gllb_xc.nuclei, a)
        print "Hello! I am processor ", world.rank, " and I am calculating PAW corrections for atom ", a
        # The GLLB-functional class will perform the corrections
        E = self.gllb_xc.calculate_non_local_paw_correction( \
                D_sp, H_sp, self.sphere_nt, self.sphere_n, nucleus, self.extra_xc_data, a)

        # Substract the Exc contribution already in setup-energies
        return E - self.Exc0
    
