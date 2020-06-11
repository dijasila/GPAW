def test_send_receive_object():
    from gpaw.mpi import world, send, receive
    if world.size == 1:
        return
    obj = (42, 'hello')
    if world.rank == 0:
        send(obj, 1, world)
    elif world.rank == 1:
        assert obj == receive(1, world)
