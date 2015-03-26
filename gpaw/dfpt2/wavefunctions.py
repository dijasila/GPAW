from math import pi
import numpy as np

from gpaw.lfc import LocalizedFunctionsCollection as LFC

from gpaw.dfpt2.kpointcontainer import KPointContainer


class WaveFunctions:
    """Class for wave-function related stuff (e.g. projectors)."""
    def __init__(self, nbands, kpt_u, setups, kd, gd, spos_ac, dtype=float):
        """Store and initialize required attributes.

        Parameters
        ----------
        nbands: int
            Number of occupied bands.
        kpt_u: list of KPoints
            List of KPoint instances from a ground-state calculation (i.e. the
            attribute ``calc.wfs.kpt_u``).
        setups: Setups
            LocalizedFunctionsCollection setups.
        kd: KPointDescriptor
            K-point and symmetry related stuff.
        gd: GridDescriptor
            Descriptor for the coarse grid.
        dtype: dtype
            This is the ``dtype`` for the wave-function derivatives (same as
            the ``dtype`` for the ground-state wave-functions).
        """

        self.dtype = dtype
        # K-point related attributes
        self.kd = kd
        # Number of occupied bands
        self.nbands = nbands
        # Projectors
        # BUG The lfc is initialized with the ground-state kd aka with the
        # irreducible information, while it is later called from kpt_u, which
        # contains only the unfolded full brilluoin zone information.
        self.pt = LFC(gd, [setup.pt_j for setup in setups], kd,
                      dtype=self.dtype)
        self.pt.set_positions(spos_ac)
        # Store grid
        self.gd = gd

        # Unfold the irreducible BZ to the full BZ
        # List of KPointContainers for the k-points in the full BZ
        self.kpt_u = []

        # No symmetries or only time-reversal symmetry used
        assert kd.symmetry.point_group is False
        assert kd.symmetry.symmorphic is True

        if kd.symmetry.time_reversal is False:
            assert len(kpt_u) == kd.nbzkpts

            for k in range(kd.nbzkpts):
                kpt_ = kpt_u[k]

                psit_nG = gd.empty(nbands, dtype=self.dtype)
                for n, psit_G in enumerate(psit_nG):
                    psit_G[:] = kpt_.psit_nG[n]

                P_ani = []
                for a in range(len(kpt_.P_ani)):
                    P_ni = kpt_.P_ani[a][:nbands]
                    P_ani.append(P_ni)

                dP_aniv = self.calculate_projector_coef(n=self.nbands, q=k,
                                                        psit_nG=psit_nG)

                # Strip off k-point attributes and store in the KPointContainer
                # Note, only the occupied GS wave-functions are retained here!
                kpt = KPointContainer(weight=kpt_.weight,
                                      k=kpt_.k,
                                      ik=kpt_.k,
                                      s=kpt_.s,
                                      phase_cd=kpt_.phase_cd,
                                      eps_n=kpt_.eps_n[:nbands],
                                      psit_nG=psit_nG,
                                      psit1_nG=None,
                                      P_ani=P_ani,
                                      dP_aniv=dP_aniv)

                self.kpt_u.append(kpt)
        else:
            # If the ground-state calculation used symmetries, map now every-
            # thing back to the full BZ.
            assert len(kpt_u) == kd.nibzkpts

            for k, k_c in enumerate(kd.bzk_kc):
                # Index of symmetry related point in the irreducible BZ
                ik = kd.bz2ibz_k[k]

                # Index of point group operation
                s = kd.sym_k[k]

                # Time-reversal symmetry used
                time_reversal = kd.time_reversal_k[k]

                # Coordinates of symmetry related point in the irreducible BZ
                ik_c = kd.ibzk_kc[ik]
                # Point group operation
                op_cc = kd.symmetry.op_scc[s]

                # k-point from ground-state calculation
                kpt_ = kpt_u[ik]
                weight = 1. / kd.nbzkpts * (2 - kpt_.s)
                phase_cd = np.exp(2j * pi * gd.sdisp_cd * k_c[:, np.newaxis])

                psit_nG = gd.empty(nbands, dtype=self.dtype)

                # NOTE When we introduce point group symmetry, we actually have
                # to rotate P_ani
                P_ani = []
                for a in range(len(kpt_.P_ani)):
                    P_ni = kpt_.P_ani[a][:nbands]
                    if time_reversal:
                        P_ni = P_ni.conj()
                    P_ani.append(P_ni)

                for n, psit_G in enumerate(psit_nG):
                    # Rotate wave function of the irreducible point to the
                    # full k-point
                    #psit_G[:] = kd.symmetry.symmetrize_wavefunction(
                    #    kpt_.psit_nG[n], ik_c, k_c, op_cc, time_reversal)
                    psit_G[:] = kd.transform_wave_function(kpt_.psit_nG[n], k)

                # BUG no clue what symmetry adaption dP_aniv needs. PAW not
                # working yet
                dP_aniv = self.calculate_projector_coef(n=self.nbands, q=ik,
                                                        psit_nG=kpt_.psit_nG[:nbands])

                kpt = KPointContainer(weight=weight,
                                      k=k,
                                      ik=ik,
                                      s=kpt_.s,
                                      phase_cd=phase_cd,
                                      eps_n=kpt_.eps_n[:nbands],
                                      psit_nG=psit_nG,
                                      psit1_nG=None,
                                      P_ani=P_ani,
                                      dP_aniv=dP_aniv)

                self.kpt_u.append(kpt)

    def initialize(self, spos_ac):
        """Initialize projectors according to the ``gamma`` attribute."""

        # Set positions on LFC's
        #self.pt.set_positions(spos_ac)
        
        # Calculate projector coefficients for the GS wave-functions
        ####self.calculate_projector_coef()

    def reset(self):
        """Make fresh arrays for wave-function derivatives."""

        for kpt in self.kpt_u:
            kpt.psit1_nG = self.gd.zeros(n=self.nbands, dtype=self.dtype)
        
    def calculate_projector_coef(self, n=None, q=None, psit_nG=None):
        """Coefficients for the derivative of the non-local part of the PP.

        Parameters
        ----------
        k: int
            Index of the k-point of the Bloch state on which the non-local
            potential operates on.

        The calculated coefficients are the following (except for an overall
        sign of -1; see ``derivative`` member function of class ``LFC``):

        1. Coefficients from the projector functions::

                        /      a          
               P_ani =  | dG  p (G) Psi (G)  ,
                        /      i       n
                          
        2. Coefficients from the derivative of the projector functions::

                          /      a           
               dP_aniv =  | dG dp  (G) Psi (G)  ,
                          /      iv       n   

        where::
                       
                 a        d       a
               dp  (G) =  ---  Phi (G) .
                 iv         a     i
                          dR

        """

        if n is None:
            n = self.nbands

        if q is None:
            assert kpt is None
            for kpt in self.kpt_u:
                # K-point index and wave-functions
                ik = kpt.ik
                psit_nG = kpt.psit_nG

                # Integration dicts
                P_ani   = self.pt.dict(shape=n)
                dP_aniv = self.pt.dict(shape=n, derivative=True)

                # 1) Integrate with projectors
                self.pt.integrate(psit_nG, P_ani, q=ik)
                kpt.P_ani = P_ani

                # 2) Integrate with derivative of projectors
                self.pt.derivative(psit_nG, dP_aniv, q=ik)
                kpt.dP_aniv = dP_aniv

        else:
            assert psit_nG is not None

            # Integration dicts
            #P_ani   = self.pt.dict(shape=n)
            dP_aniv = self.pt.dict(shape=n, derivative=True)

            # XXX Skip this, should have been done in init
            # 1) Integrate with projectors
            #self.pt.integrate(psit_nG, P_ani, q=k)
            #kpt.P_ani = P_ani

            # 2) Integrate with derivative of projectors
            self.pt.derivative(psit_nG, dP_aniv, q=q)

            return dP_aniv

