import numpy as np


class BZ:
    def __init__(self, points):
        self.points = points


class MonkhorstPackKPoints(BZ):
    def __init__(self, size, shift=(0, 0, 0)):
        self.size = size
        self.shift = shift
        BZ.__init__(self, np.zeros((1, 3)))


class IBZ:
    def __init__(self, symmetry, bz, indices, weights):
        self.symmetry = symmetry
        self.bz = bz
        self.weights = weights
        self.points = bz.points[indices]

    def __len__(self):
        return len(self.points)

    def ranks(self, comm):
        return [0]
