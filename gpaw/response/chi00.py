from gpaw.response.chiKS import ChiKS


class Chi0(ChiKS):
    """Class to keep backwards compability for dielectric response."""

    def __init__(self, *args, **kwargs):
        ChiKS.__init__(self, *args, **kwargs)
        self.calculate = self.old_calculate

    def old_calculate(self, q_c, spin='all', A_x=None):
        """Calculate spin susceptibility in plane wave mode.

        Parameters
        ----------
        q_c : list or ndarray
            Momentum vector.
        spin : str or int
            What susceptibility should be calculated?
            Currently, '00', 'uu', 'dd', '+-' and '-+' are implemented
            'all' is an alias for '00', kept for backwards compability
            Likewise 0 or 1, can be used for 'uu' or 'dd'
        A_x : ndarray
            Output array. If None, the output array is created.

        Returns
        -------
        pd : Planewave descriptor
            Planewave descriptor for q_c.
        chi_wGG : ndarray
            The response function.
        
        Note
        ----
        When running a '00'='all' calculation and q_c = [0., 0., 0.],
        there may be additional outputs (kept only for backwards compability):
        chi_wxvG : ndarray or None
            Wings of the density response function.
        chi_wvv : ndarray or None
            Head of the density response function.

        Future: Instead of having an optical limit, in which another response
                function is calculated, the two calculations should be
                separated.
        """
        return

    def setup_chi_wings(self, nG):  # For chi00
        optical_limit = np.allclose(q_c, 0.0)
        if optical_limit:
            chi_wxvG = np.zeros((len(self.omega_w), 2, 3, nG), complex)
            chi_wvv = np.zeros((len(self.omega_w), 3, 3), complex)
            self.plasmafreq_vv = np.zeros((3, 3), complex)
        else:
            chi_wxvG = None
            chi_wvv = None
            self.plasmafreq_vv = None
