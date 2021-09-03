class ShapeDistribution:
    def __init__(self, comm, shape):
        self.comm = comm
        self.total_shape = shape
        self.shape = (shape[0] // comm.size,) + shape[1:]
