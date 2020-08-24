import pytest
import numpy as np
from gpaw.mpi import world, send, receive, broadcast_array


def test_send_receive_object():
    if world.size == 1:
        return
    obj = (42, 'hello')
    if world.rank == 0:
        send(obj, 1, world)
    elif world.rank == 1:
        assert obj == receive(0, world)


@pytest.mark.intel
def test_bcast_array():
    new = world.new_communicator

    if world.size == 2:
        comms = [world]
    elif world.size == 4:
        ranks = np.array([[0, 1], [2, 3]])
        comms = [new(ranks[world.rank // 2]),
                 new(ranks[:, world.rank % 2])]
    elif world.size == 8:
        ranks = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])
        comms = [new(ranks[world.rank // 4].ravel()),
                 new(ranks[:, world.rank // 2 % 2].ravel()),
                 new(ranks[:, :, world.rank % 2].ravel())]
    else:
        return

    array = np.zeros(3, int)
    if world.rank == 0:
        array[:] = 42

    out = broadcast_array(array, *comms)
    assert (out == 42).all()
