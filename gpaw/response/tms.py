import numpy as np

import gpaw.mpi as mpi
from gpaw.response.susceptibility import FourComponentSusceptibilityTensor

FCST = FourComponentSusceptibilityTensor


class TransverseMagneticSusceptibility(FCST):
    """Class calculating the transverse magnetic susceptibility
    and related physical quantities."""

    def __init__(self, *args, **kwargs):
        assert kwargs['fxc'] == 'ALDA'

        # Enable scaling to fit to Goldstone theorem
        if 'fxckwargs' in kwargs and 'fxc_scaling' in kwargs['fxckwargs']:
            self.fxc_scaling = kwargs['fxckwargs']['fxc_scaling']
        else:
            self.fxc_scaling = None

        FCST.__init__(self, *args, **kwargs)

    def get_macroscopic_component(self, spincomponent, q_c, frequencies,
                                  filename=None, txt=None):
        """Calculates the spatially averaged (macroscopic) component of the
        transverse magnetic susceptibility and writes it to a file.
        
        Parameters
        ----------
        spincomponent : str
            '+-': calculate chi+-, '-+: calculate chi-+
        q_c, frequencies, filename, txt : see gpaw.response.susceptibility

        Returns
        -------
        see gpaw.response.susceptibility
        """
        assert spincomponent in ['+-', '-+']

        return FCST.get_macroscopic_component(self, spincomponent, q_c,
                                              frequencies, filename=filename,
                                              txt=txt)

    def get_component_array(self, spincomponent, q_c, frequencies,
                            array_ecut=50, filename=None, txt=None):
        """Calculates a specific spin component of the
        transverse magnetic susceptibility and writes it to a file.
        
        Parameters
        ----------
        spincomponent : str
            '+-': calculate chi+-, '-+: calculate chi-+
        q_c, frequencies,
        array_ecut, filename, txt : see gpaw.response.susceptibility

        Returns
        -------
        see gpaw.response.susceptibility
        """
        assert spincomponent in ['+-', '-+']

        return FCST.get_component_array(self, spincomponent, q_c,
                                        frequencies, array_ecut=array_ecut,
                                        filename=filename, txt=txt)

    def _calculate_component(self, spincomponent, pd, wd):
        """Calculate a transverse magnetic susceptibility element.

        Returns
        -------
        pd, wd, chiks_wGG, chi_wGG : see gpaw.response.susceptibility
        """
        chiks_wGG = self.calculate_ks_component(spincomponent, pd,
                                                wd, txt=self.cfd)
        Kxc_GG = self.get_xc_kernel(spincomponent, pd,
                                    chiks_wGG=chiks_wGG, txt=self.cfd)

        chi_wGG = self.invert_dyson(chiks_wGG, Kxc_GG)

        return pd, wd, chiks_wGG, chi_wGG

    def get_xc_kernel(self, spincomponent, pd, chiks_wGG=None, txt=None):
        """Get the exchange correlation kernel."""
        Kxc_GG = self.fxc(spincomponent, pd, txt=self.cfd)

        fxc_scaling = self.fxc_scaling

        if fxc_scaling is not None:
            assert isinstance(fxc_scaling[0], bool)
            if fxc_scaling[0]:
                if fxc_scaling[1] is None:
                    assert pd.kd.gamma
                    print('Finding rescaling of kernel to fulfill the '
                          'Goldstone theorem', file=self.fd)
                    fxc_scaling[1] = find_goldstone_scaling(self.chiks.omega_w,
                                                            chiks_wGG, Kxc_GG,
                                                            world=self.world)

                assert isinstance(fxc_scaling[1], float)
                Kxc_GG *= fxc_scaling[1]

        self.fxc_scaling = fxc_scaling

        return Kxc_GG


def find_goldstone_scaling(omega_w, chi0_wGG, Kxc_GG, world=mpi.world):
    """Find a scaling of the kernel to move the magnon peak to omega=0."""
    wgs = np.abs(omega_w).argmin()
    if not np.allclose(omega_w[wgs], 0., rtol=1.e-8):
        raise ValueError("Frequency grid needs to include"
                         + " omega=0. to allow Goldstone scaling")

    fxcs = 1.

    # Only one rank, rgs, has omega=0 and finds rescaling
    nw = len(omega_w)
    mynw = (nw + world.size - 1) // world.size
    rgs, mywgs = wgs // mynw, wgs % mynw
    fxcsbuf = np.empty(1, dtype=float)
    if world.rank == rgs:
        chi0_GG = chi0_wGG[mywgs]
        chi_GG = np.dot(np.linalg.inv(np.eye(len(chi0_GG)) +
                                      np.dot(chi0_GG, Kxc_GG * fxcs)),
                        chi0_GG)
        # Scale so that kappaM=0 in the static limit (omega=0)
        kappaM = (chi0_GG[0, 0] / chi_GG[0, 0]).real
        # If kappaM > 0, increase scaling (recall: kappaM ~ 1 - Kxc Re{chi_0})
        scaling_incr = 0.1 * np.sign(kappaM)
        while abs(kappaM) > 1.e-7 and abs(scaling_incr) > 1.e-7:
            fxcs += scaling_incr
            if fxcs <= 0.0 or fxcs >= 10.:
                raise Exception('Found an invalid fxc_scaling of %.4f' % fxcs)

            chi_GG = np.dot(np.linalg.inv(np.eye(len(chi0_GG)) +
                                          np.dot(chi0_GG, Kxc_GG * fxcs)),
                            chi0_GG)
            kappaM = (chi0_GG[0, 0] / chi_GG[0, 0]).real

            # If kappaM changes sign, change sign and refine increment
            if np.sign(kappaM) != np.sign(scaling_incr):
                scaling_incr *= -0.2
        fxcsbuf[:] = fxcs

    # Broadcast found rescaling
    world.broadcast(fxcsbuf, rgs)
    fxcs = fxcsbuf[0]

    return fxcs
