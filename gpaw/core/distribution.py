from gpaw.mpi import serial_comm


class ShapeDistribution:
    def __init__(self, comm, shape):
        self.comm = comm
        self.total_shape = shape
        if not shape:
            self.shape = ()
        else:
            self.shape = (shape[0] // comm.size,) + shape[1:]


def create_shape_distributuion(shape, dist):
    if isinstance(dist, ShapeDistribution):
        assert shape is None
    else:
        if isinstance(shape, int):
            shape = (shape,)
        dist = dist or serial_comm
        dist = ShapeDistribution(dist, shape)
    return dist
