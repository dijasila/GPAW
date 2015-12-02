from __future__ import print_function
from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA

def param_ignore_warning(parameters):
    """Many of functionals do not accept parameters and these are ignored.
       This function prints a warning if parameters are ignored.
    """
    if parameters is not None and len(parameters) > 0:
        print("Warning: XC functional ignoring parameters", parameters)

def get_kernel_by_name(name, parameters):
    """Return xc kernel for a given name.

    Libxc will be used for default, unless the name is one of the following:
    BEE1, BEE2, LB94, TPSS, M06L, revTPSS, old... PPLDA, pyPBE, pyPBEsol,
    pyRPBE, pyxzPBEsol.
    
    xc kernel object provides the basic computational functionality for 
    LDA, GGA and MGGA type of functionals. It needs to be wrapped by
    LDA, GGA or MGGA object, which inherit the XCFunctional object.
    
    """
    if name == 'BEE1':
        from gpaw.xc.bee import BEE1
        return BEE1(parameters)
    if name == 'BEE2':
        from gpaw.xc.bee import BEE2
        return BEE2(parameters)
    if name == 'LB94':
        from gpaw.xc.lb94 import LB94
        return LB94()
    if name == 'TPSS' or name == 'M06L' or name == 'revTPSS':
        from gpaw.xc.kernel import XCKernel
        return XCKernel(name)
    if name.startswith('old'):
        from gpaw.xc.kernel import XCKernel
        return XCKernel(name[3:])
    if name == 'PPLDA':
        from gpaw.xc.lda import PurePythonLDAKernel
        return PurePythonLDAKernel()
    if name in ['pyPBE', 'pyPBEsol', 'pyRPBE', 'pyzvPBEsol']:
        from gpaw.xc.gga import PurePythonGGAKernel
        return PurePythonGGAKernel(name)
    if name == '2D-MGGA':
        from gpaw.xc.mgga import PurePython2DMGGAKernel
        return PurePython2DMGGAKernel(name, parameters)
    if name[0].isdigit():
        from gpaw.xc.parametrizedxc import ParametrizedKernel
        return ParametrizedKernel(name)

    # libXC is used by default
    return LibXC(name)

def XC(kernel, parameters=None):
    """Create XCFunctional object.

    kernel: XCKernel object or str
        Kernel object or name of functional.
    parameters: ndarray
        Parameters for BEE functional.

    Recognized names are: LDA, PW91, PBE, revPBE, RPBE, BLYP, HCTH407,
    TPSS, M06L, revTPSS, new_vdW-DF, vdW-DF, vdW-DF2, EXX, PBE0, B3LYP, BEE,
    GLLBSC.  One can also use equivalent libxc names, for example
    GGA_X_PBE+GGA_C_PBE is equivalent to PBE, and LDA_X to the LDA exchange.
    In this way one has access to all the functionals defined in libxc.
    See xc_funcs.h for the complete list.  """

    if isinstance(kernel, str):
        name = kernel

        if name not in ['BEE1','BEE2','2D-MGGA']:
            param_ignore_warning(parameters)

        # New vdW-DF implementations via libvdwxc
        # Temporary name until we decide how to deal with old and new versions.
        if name in ['new_vdW-DF']:
            from gpaw.xc.libvdwxc import VDWDF
            return VDWDF()

        # Old vdW-DF family implementations
        if name in ['vdW-DF', 'vdW-DF2', 'optPBE-vdW', 'optB88-vdW',
                    'C09-vdW', 'mBEEF-vdW', 'BEEF-vdW']:
            from gpaw.xc.vdw import VDWFunctional
            return VDWFunctional(name)

        # GLLB family potentials
        if name.startswith('GLLB'):
            from gpaw.xc.gllb.nonlocalfunctionalfactory import \
                NonLocalFunctionalFactory
            xc = NonLocalFunctionalFactory().get_functional_by_name(name)
            xc.print_functional()
            return xc

        # Hybrid functionals
        if name in ['EXX', 'PBE0', 'B3LYP']:
            from gpaw.xc.hybrid import HybridXC
            return HybridXC(name)

        if name in ['HSE03', 'HSE06']:
            from gpaw.xc.exx import EXX
            return EXX(name)

        if name == 'TB09':
            from gpaw.xc.tb09 import TB09
            return TB09()

        # Orbital dependent functionals
        if name.startswith('ODD_'):
            from ODD import ODDFunctional
            return ODDFunctional(name[4:])

        if name.endswith('PZ-SIC'):
            try:
                from ODD import PerdewZungerSIC as SIC
                return SIC(xc=name[:-7])
            except: # XXX Dangerous except
                from gpaw.xc.sic import SIC
                return SIC(xc=name[:-7])

        # If this point is reached, the functional string is 
        # either LDA, GGA or MGGA kernel        
        kernel = get_kernel_by_name(name, parameters)
    else:
        param_ignore_warning(parameters)

    if kernel.type == 'LDA':
        return LDA(kernel)
    elif kernel.type == 'GGA':
        return GGA(kernel)
    else:
        return MGGA(kernel)

        
def xc(filename, xc, ecut=None):
    """Calculate non self-consitent energy.
    
    filename: str
        Name of restart-file.
    xc: str
        Functional
    ecut: float
        Plane-wave cutoff for exact exchange.
    """
    name, ext = filename.rsplit('.', 1)
    assert ext == 'gpw'
    if xc in ['EXX', 'PBE0', 'B3LYP']:
        from gpaw.xc.exx import EXX
        exx = EXX(filename, xc, ecut=ecut, txt=name + '-exx.txt')
        exx.calculate()
        e = exx.get_total_energy()
    else:
        from gpaw import GPAW
        calc = GPAW(filename, txt=None)
        e = calc.get_potential_energy() + calc.get_xc_difference(xc)
    print(e, 'eV')
