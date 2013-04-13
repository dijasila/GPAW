===
GPU
===

Aalto Effort
============

(Samuli Hakala, Ville Havu, Jussi Enkovaara (CSC) )
We have been implementing the most performance critical C-kernels
in the finite-difference mode to utilize GPUs. Implementation is done
using CUDA and cuBLAS libraries when possible, Python interface to CUDA,
PyCUDA is also utilized. Code supports using multiple GPUs with MPI. 
First tests indicate promising speedups compared
to CPU-only code, and we are hoping to test larger systems (where
the benefits are expected to be larger) soon. Currently, we are extending the
CUDA implementation to real-time TDDFT.

Code is not in full production level yet.

Stanford/SUNCAT Effort
======================

(Lin Li, Jun Yan, Christopher O'Grady)

We believe that GPAW has two areas where significant improvement would
be very helpful: ease-of-convergence and performance.

We think the first of those is going to be significantly improved (for
small systems that are largely of interest to SUNCAT) by the addition
of the planewave-basis mode.  To complement the GPU grid-mode work of
Samuli Hakala et. al. at Aalto University we are seeing if it is possible to
significantly improve the planewave performance (and
performance-per-dollar) of GPAW using GPUs.  We are also using GPUs to
see if the Random Phase Approximation performance can be improved.

Installation of cuda on Fedora 18 x86_64
========================================

First, cuda_5.0.35 does not support gcc version 4.7 and up,
either kernel 3.7 and up https://bugzilla.redhat.com/show_bug.cgi?format=multiple&id=902045, however there exist workarounds.

Proceed as follows:

0. boot a 3.6 kernel

1. yum -y install wget make gcc-c++ freeglut-devel libXi-devel libXmu-devel mesa-libGLU-devel

2. configure rpmfusion-nonfree http://rpmfusion.org/Configuration

3. yum -y install xorg-x11-drv-nvidia-libs

4. disable X::

     rm /etc/systemd/system/default.target&& ln -s /lib/systemd/system/multi-user.target /etc/systemd/system/default.target

5. select a 3.6 version (see https://bugzilla.redhat.com/show_bug.cgi?format=multiple&id=902045 ) of kernel in ``/etc/grub2.cfg`` and make sure the *linux* line contains::

     nouveau.modeset=0 rd.driver.blacklist=nouveau

   See http://www.pimp-my-rig.com/2012/01/unload-nouveau-install-nvidia-driver.html

6. disable kernel and nvidia updates in ``/etc/yum.conf``::

     exclude=kernel* *nvidia*

7. reboot

8. download and install cuda::

     wget http://developer.download.nvidia.com/compute/cuda/5_0/rel-update-1/installers/cuda_5.0.35_linux_64_fedora16-1.run
     sh cuda_5.0.35_linux_64_fedora16-1.run -override compiler

   Keep the recommended installation paths (``/usr/local/cuda-5.0``, ...),
   and after the installation create a link::

     ln -s /usr/local/cuda /opt/cuda

9. add to ``~/bashrc``::

     export PATH=/opt/cuda/bin:${PATH}
     export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

   and source it.

10. convert error into a warning in ``/usr/local/cuda-5.0/include/host_config.h``::

      // #error — unsupported GNU version! gcc 4.7 and up are not supported!
      #warning — unsupported GNU version! gcc 4.7 and up are not supported!


    See https://www.udacity.com/wiki/cs344/troubleshoot_gcc47

11. test::

      cd NVIDIA_CUDA-5.0_Samples/1_Utilities/deviceQuery
      make&& ./deviceQuery

12. install http://mathema.tician.de/software/pycuda (needed only for *Aalto Effort*)::

      cd&& git clone https://github.com/inducer/pycuda.git
      cd ~/pycuda
      PATH=$PATH:/opt/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/lib64 python configure.py --update-user --boost-compiler=gcc
      /bin/mv -f ~/.aksetup-defaults.py siteconf.py
      sed -i "s/boost_python-py27/boost_python/" siteconf.py
      sed -i 's/boost_thread/boost_thread-mt/' siteconf.py
      sed -i "s#'\${CUDA_ROOT}/lib', ##" siteconf.py
      PATH=$PATH:/opt/cuda/bin LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/lib64 python setup.py install --root=~/pycuda-fc18-1

    and add to ``~/.bashrc``::

      export PYTHONPATH=~/pycuda-fc18-1/usr/lib64/python2.7/site-packages:${PYTHONPATH}

