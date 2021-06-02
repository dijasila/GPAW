from myqueue.workflow import run
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.xc.libvdwxc import libvdwxc_has_pfft


def workflow():
    if compiled_with_libvdwxc():
        run(script='libvdwxc-example.py')
        if libvdwxc_has_pfft():
            run(script='libvdwxc-pfft-example.py', cores=8)
