from gpaw.new.brillouin import IBZ


class Symmetry:
    def __init__(self, symmetry):
        self.symmetry = symmetry

    def reduce(self, bz):
        return IBZ(self, bz, [0], [0], [1.0])
