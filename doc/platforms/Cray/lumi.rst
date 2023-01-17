.. _lumi:

============================
lumi.csc.fi
============================


.. note::
   These instructions are up-to-date as of January 2023.

As all the GPU related libraries are not available in the CPU only partition, separate 
installation is needed for LUMI-C and LUMI-G.

GPAW for LUMI-C
===============

Following instructions are preliminary guidelines on how to install GPAW to be utilized with the
CPU only partition of LUMI-C.
To begin, add a following file to your current directory and call it `siteconfig-lumi.py`.

.. literalinclude:: siteconfig-lumi.py

Following script will install latest GPAW under `gpaw-cpu`` subdirectory (within the current directory).
It is recommended to perform installation under the `project` directory (see 
`LUMI user documentation <https://docs.lumi-supercomputer.eu/runjobs/lumi_env/storing-data/>`_)::
        module load cray-python/3.9.12.1 PrgEnv-gnu/8.3.3
        
        export GPAW_CONFIG=$PWD/siteconfig-lumi.py
        export PYTHONUSERBASE=$PWD/gpaw-cpu
        pip install --user git+https://gitlab.com/gpaw/gpaw.git 2>&1 | tee install.log


In order to use GPAW, one needs to set also `PATH`::
        export PATH=$PYTHONUSERBASE/bin:$PATH

GPAW for LUMI-G
===============

GPU version of GPAW depends on `cupy <https://cupy.dev/>`_ which needs to be installed first.
Currently, the version of `rocm` library differs in the login and compute nodes, and thus the
installation needs to be done in the compute node as follows::
        module load cray-python/3.9.12.1 PrgEnv-gnu/8.3.3
        module load craype-accel-amd-gfx90a rocm

        export PYTHONUSERBASE=$PWD/gpaw-gpu
        # Start shell in GPU node
        srun -p dev-g --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 -A <project> -t 0:30:00 --pty bash        
        # Set environment variables for cupy installation
        export CUPY_INSTALL_USE_HIP=1
        export ROCM_HOME=$ROCM_PATH
        export HCC_AMDGPU_TARGET=gfx90a
        pip install --user git+https://github.com/cupy/cupy.git@v11.2.0

