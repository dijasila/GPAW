from gpaw.response.susceptibility import FourComponentSusceptibilityTensor


class TransverseMagneticSusceptibility(FourComponentSusceptibilityTensor):
    """Class calculating the transverse magnetic susceptibility
    and related physical quantities."""

    def __init__(self, *args, **kwargs):
        FCST = FourComponentSusceptibilityTensor
        FCST.__init__(self, *args, **kwargs)

    def calculate_component(self, *args, **kwargs):
        spincomponent = args[0]
        assert spincomponent in ['+-', '-+']
        assert kwargs['fxc'] == 'ALDA'

        FCST = FourComponentSusceptibilityTensor
        FCST.calculate_component(self, *args, **kwargs)
