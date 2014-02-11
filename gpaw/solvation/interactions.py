class Interaction():
    def __init__(self):
        pass


class SurfaceInteraction(Interaction):
    def __init__(self, surface_tension):
        Interaction.__init__(self)


class VolumeInteraction(Interaction):
    def __init__(self, pressure):
        Interaction.__init__(self)


class LeakedDensityInteraction(Interaction):
    def __init__(self, charging_energy):
        Interaction.__init__(self)
