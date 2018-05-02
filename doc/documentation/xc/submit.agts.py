from myqueue.job import Job
from gpaw.utilities import compiled_with_libvdwxc
from gpaw.xc.libvdwxc import libvdwxc_has_pfft

def workflow():
    jobs = []
    if compiled_with_libvdwxc():
        jobs.apend(Job('libvdwxc-example.py'))
        if libvdwxc_has_pfft():
            jobs.append(Job('libvdwxc-pfft-example.py', cores=8))
    return jobs
