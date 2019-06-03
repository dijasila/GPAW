from gpaw.response.susceptibility import FourComponentSusceptibilityTensor


class TransverseMagneticSusceptibility(FourComponentSusceptibilityTensor):
    """Class calculating the transverse magnetic susceptibility
    and related physical quantities."""

    def __init__(self, *args, **kwargs):
        assert kwargs['fxc'] == 'ALDA'

        FCST = FourComponentSusceptibilityTensor
        FCST.__init__(self, *args, **kwargs)

    def get_macroscopic_component(self, spincomponent, q_c, filename=None):
        """Calculates the spatially averaged (macroscopic) component of the
        transverse magnetic susceptibility and writes it to a file.
        
        Parameters
        ----------
        spincomponent : str
            '+-': calculate chi+-, '-+: calculate chi-+
        q_c, filename : see gpaw.response.susceptibility

        Returns
        -------
        see gpaw.response.susceptibility
        """
        assert spincomponent in ['+-', '-+']

        FCST = FourComponentSusceptibilityTensor

        return FCST.get_macroscopic_component(self, spincomponent, q_c,
                                              filename=filename)
