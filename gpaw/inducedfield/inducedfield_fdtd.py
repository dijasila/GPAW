import numpy as np

from ase.parallel import parprint
from ase.units import Bohr

from gpaw import debug
from gpaw.analyse.observers import Observer
from gpaw.transformers import Transformer
from gpaw.lfc import BasisFunctions
from gpaw.utilities import unpack2, is_contiguous

from gpaw.inducedfield.inducedfield_base import BaseInducedField, sendreceive_dict


class FDTDInducedField(BaseInducedField, Observer):
    """Induced field class for FDTD.
    
    Attributes (see also ``BaseInducedField``):
    -------------------------------------------
    time: float
        Current time
    Fn_wG: ndarray (complex)
        Fourier transform of induced polarization charge density
    n0_G: ndarray (float)
        Ground state charge density
    """
    
    def __init__(self, filename=None, paw=None,
                  frequencies=None, folding='Gauss', width=0.08,
                  interval=1, restart_file=None
                  ):
        """
        Parameters (see also ``BaseInducedField``):
        -------------------------------------------
        paw: TDDFT object
            TDDFT object for time propagation
        width: float
            Width in eV for the Gaussian (sigma) or Lorentzian (eta) folding
            Gaussian   = exp(- (1/2) * sigma^2 * t^2)
            Lorentzian = exp(- eta * t)
        interval: int
            Number of timesteps between calls (used when attaching)
        restart_file: string
            Name of the restart file
        """

        Observer.__init__(self, interval)
        # From observer:
        # self.niter
        # self.interval
        
        # Restart file
        self.restart_file = restart_file
        
        # These are allocated in allocate()
        self.Fn_wsG = None
        self.n0_G = None


        self.readwritemode_str_to_list = \
        {'': ['Fn', 'n0', 'atoms'],
         'all': ['Fn', 'n0',
                 'Frho', 'Fphi', 'Fef', 'Ffe', 'eps0', 'atoms'],
         'field': ['Frho', 'Fphi', 'Fef', 'Ffe', 'eps0', 'atoms']}

        BaseInducedField.__init__(self, filename, paw,
                                  frequencies, folding, width)
        
    def initialize(self, paw, allocate=True):
        BaseInducedField.initialize(self, paw, allocate)

        if self.has_paw:
            # FDTD replacements and overwrites
            self.fdtd = paw.hamiltonian.poisson
            #self.gd = self.fdtd.cl.gd.refine()
            self.gd = self.fdtd.cl.gd

            assert hasattr(paw, 'time') and hasattr(paw, 'niter'), 'Use TDDFT!'
            self.time = paw.time
            self.niter = paw.niter

            # TODO: remove this requirement
            assert np.count_nonzero(paw.kick_strength) > 0, \
            'Apply absorption kick before %s' % self.__class__.__name__

            # Background electric field
            self.Fbgef_v = paw.kick_strength

            # Attach to PAW-type object
            paw.attach(self, self.interval)
            # TODO: write more details (folding, freqs, etc)
            parprint('%s: Attached ' % self.__class__.__name__)

    def set_folding(self, folding, width):
        BaseInducedField.set_folding(self, folding, width)

        if self.folding is None:
            self.envelope = lambda t: 1.0
        else:
            if self.folding == 'Gauss':
                self.envelope = lambda t: np.exp(- 0.5 * self.width**2 * t**2)
            elif self.folding == 'Lorentz':
                self.envelope = lambda t: np.exp(- self.width * t)
            else:
                raise RuntimeError('unknown folding "' + self.folding + '"')

    def allocate(self):
        if not self.allocated:

            # Ground state charge density
            self.n0_G = -1.0 * self.paw.hamiltonian.poisson.classical_material.sign * \
                        self.paw.hamiltonian.poisson.classical_material.charge_density.copy()

            # Fourier transformed charge density
            self.Fn_wG = self.paw.hamiltonian.poisson.cl.gd.zeros((self.nw,),
                                                                   dtype=self.dtype)
            self.allocated = True

        if debug:
            assert is_contiguous(self.Fn_wG, self.dtype)

    def deallocate(self):
        BaseInducedField.deallocate(self)
        self.n0_G = None
        self.Fn_wG = None
        
    def update(self):
        # Update time
        self.time = self.paw.time
        time_step = self.paw.time_step

        # Complex exponential with envelope
        f_w = np.exp(1.0j * self.omega_w * self.time) * \
              self.envelope(self.time) * time_step

        # Time-dependent quantities
        n_G = -1.0 * self.fdtd.classical_material.sign * \
              self.fdtd.classical_material.charge_density

        # Update Fourier transforms
        for w in range(self.nw):
            self.Fn_wG[w] += (n_G - self.n0_G) * f_w[w]

        # Restart file
        if self.restart_file is not None and \
           self.niter % self.paw.dump_interval == 0:
            self.write(self.restart_file)
            parprint('%s: Wrote restart file %s' % (self.__class__.__name__, self.restart_file))

    def get_induced_density(self, from_density, gridrefinement):
        if self.gd == self.fdtd.cl.gd:
            Frho_wg = self.Fn_wG.copy()
        else:
            Frho_wg = Transformer(self.fdtd.cl.gd,
                                  self.gd,
                                  self.stencil,
                                  dtype=self.dtype).apply(self.Fn_wG)

        Frho_wg, gd = self.interpolate_density(self.gd, Frho_wg, gridrefinement)
        return Frho_wg, gd

    def interpolate_density(self, gd, Fn_wg, gridrefinement=2):
        
        # Find m for
        # gridrefinement = 2**m
        m1 = np.log(gridrefinement) / np.log(2.)
        m = int(np.round(m1))

        # Check if m is really integer
        if np.absolute(m - m1) < 1e-8:
            for i in range(m):
                gd2 = gd.refine()
                
                # Interpolate
                interpolator = Transformer(gd, gd2, self.stencil,
                                           dtype=self.dtype)
                Fn2_wg = gd2.empty((self.nw,), dtype=self.dtype)
                for w in range(self.nw):
                    interpolator.apply(Fn_wg[w], Fn2_wg[w],
                                       np.ones((3, 2), dtype=complex))

                gd = gd2
                Fn_wg = Fn2_wg
        else:
            raise NotImplementedError
        
        return Fn_wg, gd

    def _read(self, reader, reads):
        BaseInducedField._read(self, reader, reads)

        # Test time
        r = reader
        time = r.time
        if self.has_paw:
            # Test time
            if abs(time - self.time) >= 1e-9:
                raise IOError('Timestamp is incompatible with calculator.')
        else:
            self.time = time

        # Allocate
        self.allocate()

        def readarray(name):
            if name.split('_')[0] in reads:
                self.gd.distribute(r.get(name), getattr(self, name))

        # Read arrays
        readarray('n0_G')
        readarray('Fn_wG')
        readarray('eps0_G')

    def _write(self, writer, writes):
        # Swap classical and quantum cells, and shift atom positions for the time of writing
        qmcell = self.atoms.get_cell()
        self.atoms.set_cell(self.fdtd.cl.cell * Bohr) # Set classical cell
        self.atoms.positions = self.atoms.get_positions() + self.fdtd.qm.corner1 * Bohr
        BaseInducedField._write(self, writer, writes)
        self.atoms.set_cell(qmcell) # Restore quantum cell to the atoms object
        self.atoms.positions = self.atoms.get_positions() - self.fdtd.qm.corner1 * Bohr

        # Write time propagation status
        writer.write(time=self.time)

        # Mask, interpolation approach:
        #self.eps0_G = self.fdtd.classical_material.permittivityValue(omega=0.0) - self.fdtd.classical_material.epsInfty
        #self.eps0_G= -interpolator.apply(self.eps0_G)
        
        # Mask, direct approach:
        self.eps0_G = self.fdtd.cl.gd.zeros()
        for component in self.fdtd.classical_material.components:
            self.eps0_G += 1.0 * component.get_mask(gd=self.fdtd.cl.gd)

#        def writearray(name, shape, dtype):
#            if name.split('_')[0] in writes:
#                writer.add_array(name, shape, dtype)
#            a_wxg = getattr(self, name)
#            for w in range(self.nw):
#                if self.fdtd.cl.gd == self.gd:
#                    writer.fill(self.gd.collect(a_wxg[w]))
#                else:
#                    writer.fill(self.gd.collect(Transformer(self.fdtd.cl.gd, self.gd, self.stencil, self.dtype).apply(a_wxg[w])))

        def writearray(name):
            if name.split('_')[0] in writes:
                a_xg = getattr(self, name)
                if self.fdtd.cl.gd != self.gd:
                    a_xg = Transformer(self.fdtd.cl.gd,
                                       self.gd,
                                       self.stencil,
                                       a_xg.dtype).apply(a_xg)
                writer.write(**{name: self.gd.collect(a_xg)})

        # Write time propagation arrays
        writearray('n0_G')
        writearray('Fn_wG')
        writearray('eps0_G')

        if hasattr(self.fdtd, 'qm') and hasattr(self.fdtd.qm, 'corner1'):
            writer.write(corner1_v=self.fdtd.qm.corner1)
            writer.write(corner2_v=self.fdtd.qm.corner2)

        self.fdtd.cl.gd.comm.barrier()
