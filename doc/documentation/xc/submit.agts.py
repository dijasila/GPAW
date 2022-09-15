from myqueue.workflow import run
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.xc.libvdwxc import libvdwxc_has_pfft


def workflow():
    with run(script='s22_set.py', cores=8, tmax='1d'):
        run(function=check_s22)
    run(script='hydrogen_atom.py', cores=16)
    if compiled_with_libvdwxc():
        run(script='libvdwxc-example.py')
        if libvdwxc_has_pfft():
            run(script='libvdwxc-pfft-example.py', cores=8)


def check_s22():
    1 / 0  # todo ...
