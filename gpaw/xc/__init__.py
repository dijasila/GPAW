from __future__ import print_function
import warnings

from ase.utils import basestring

from gpaw.xc.libxc import LibXC
from gpaw.xc.lda import LDA
from gpaw.xc.gga import GGA
from gpaw.xc.mgga import MGGA


def XC(kernel, parameters=None):
    """Create XCFunctional object.

    kernel: XCKernel object, dict or str
        Kernel object or name of functional.
    parameters: ndarray
        Parameters for BEE functional.

    Recognized names are: LDA, PW91, PBE, revPBE, RPBE, BLYP, HCTH407,
    TPSS, M06-L, revTPSS, vdW-DF, vdW-DF2, EXX, PBE0, B3LYP, BEE,
    GLLBSC.  One can also use equivalent libxc names, for example
    GGA_X_PBE+GGA_C_PBE is equivalent to PBE, and LDA_X to the LDA exchange.
    In this way one has access to all the functionals defined in libxc.
    See xc_funcs.h for the complete list.  """

    if isinstance(kernel, basestring):
        # We have the option of implementing a string specification
        # minilanguage for xc keywords like 'vdW-DF:type=libvdwxc'
        kernel = {'name': kernel}

    kwargs = {}
    if isinstance(kernel, dict):
        kwargs = kernel.copy()
        name = kwargs.pop('name')
        xctype = kwargs.pop('type', None)

        if xctype == 'libvdwxc':
            # Must handle libvdwxc before old vdw implementation to override
            # behaviour for 'name'
            from gpaw.xc.libvdwxc import VDWXC
            return VDWXC(name=name, **kwargs)

        if name in ['vdW-DF', 'vdW-DF2', 'optPBE-vdW', 'optB88-vdW',
                    'C09-vdW', 'mBEEF-vdW', 'BEEF-vdW']:
            from gpaw.xc.vdw import VDWFunctional
            return VDWFunctional(name, **kwargs)
        elif name in ['EXX', 'PBE0', 'B3LYP']:
            from gpaw.xc.hybrid import HybridXC
            return HybridXC(name, **kwargs)
        elif name in ['HSE03', 'HSE06']:
            from gpaw.xc.exx import EXX
            return EXX(name, **kwargs)
        elif name == 'BEE1':
            from gpaw.xc.bee import BEE1
            kernel = BEE1(parameters)
        elif name == 'BEE2':
            from gpaw.xc.bee import BEE2
            kernel = BEE2(parameters)
        elif name == 'TOFU':
            from gpaw.xc.bee import BEE2
            if parameters is None:
              parameters2 = (
                [6.5124, 0.0] + list(range(12)) +
                [
				1.401999997714607e+00,
				4.019999934886556e-01,
				-9.689439551991570e-09,
				-1.108494993689335e-08,
				-1.101985898709578e-08,
				-9.646236399820265e-09,
				-7.162871093499266e-09,
				-4.725756885942376e-09,
				-2.215939332815698e-09,
				-8.417487361008392e-10,
				4.234628587572872e-10,
				1.064977169956315e-09,
				1.0])
		# last number is prefactor for pbesol gradient correction
		# to semilocal correlation
            else:
              parameters2 = parameters
            kernel = BEE2(parameters2, cpbesol=True)
            kernel.name = 'TOFU'
        elif name.startswith('GLLB'):
            from gpaw.xc.gllb.nonlocalfunctionalfactory import \
                NonLocalFunctionalFactory
            # Pass kwargs somewhere?
            xc = NonLocalFunctionalFactory().get_functional_by_name(name)
            xc.print_functional()
            return xc
        elif name == 'LB94':
            from gpaw.xc.lb94 import LB94
            kernel = LB94()
        elif name == 'TB09':
            from gpaw.xc.tb09 import TB09
            return TB09(**kwargs)
        elif name.startswith('ODD_'):
            from ODD import ODDFunctional
            return ODDFunctional(name[4:], **kwargs)
        elif name.endswith('PZ-SIC'):
            try:
                from ODD import PerdewZungerSIC as SIC
                return SIC(xc=name[:-7], **kwargs)
            except:
                from gpaw.xc.sic import SIC
                return SIC(xc=name[:-7], **kwargs)
        elif name in ['TPSS', 'M06-L', 'M06L', 'revTPSS']:
            if name == 'M06L':
                name = 'M06-L'
                warnings.warn('Please use M06-L instead of M06L')
            from gpaw.xc.kernel import XCKernel
            kernel = XCKernel(name)
        elif name.startswith('old'):
            from gpaw.xc.kernel import XCKernel
            kernel = XCKernel(name[3:])
        elif name == 'PPLDA':
            from gpaw.xc.lda import PurePythonLDAKernel
            kernel = PurePythonLDAKernel()
        elif name in ['pyPBE', 'pyPBEsol', 'pyRPBE', 'pyzvPBEsol']:
            from gpaw.xc.gga import PurePythonGGAKernel
            kernel = PurePythonGGAKernel(name)
        elif name == '2D-MGGA':
            from gpaw.xc.mgga import PurePython2DMGGAKernel
            kernel = PurePython2DMGGAKernel(name, parameters)
        elif name[0].isdigit():
            from gpaw.xc.parametrizedxc import ParametrizedKernel
            kernel = ParametrizedKernel(name)
        else:
            kernel = LibXC(name)

    if kernel.type == 'LDA':
        return LDA(kernel, **kwargs)
    elif kernel.type == 'GGA':
        return GGA(kernel, **kwargs)
    else:
        return MGGA(kernel, **kwargs)


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
