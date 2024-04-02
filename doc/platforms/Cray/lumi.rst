.. _lumi:

=================================
The ``lumi.csc.fi`` supercomputer
=================================

.. note::
   These instructions are up-to-date as of September 2023.

It is recommended to perform installation under the
``/projappl/project_...`` directory (see `LUMI user documentation
<https://docs.lumi-supercomputer.eu/storage/>`_). A separate installation
is needed for LUMI-C and LUMI-G.


GPAW for LUMI-G
===============

Load the following modules:

.. code-block:: bash

  export EBU_USER_PREFIX=/scratch/project_465000538/GPAW/EasyBuild
  module load LUMI/22.12 partition/G
  module load cpeGNU/22.12
  module load craype-accel-amd-gfx90a
  module load rocm/5.2.3
  module load cray-python/3.9.13.1
  module load cray-fftw/3.3.10.1
  module load ASE/3.22.1-cpeGNU-22.12
  module load CuPy/12.2.0-cpeGNU-22.12
  module load ELPA/2023.05.001-cpeGNU-22.12-GPU
  module load libxc/6.2.2-cpeGNU-22.12

Create a virtual environment and activate it::

  python3 -m venv venv
  source venv/bin/activate

Clone the GPAW source code::

  git clone git@gitlab.com:gpaw/gpaw

Copy this :git:`~doc/platforms/Cray/siteconfig-lumi-gpu.py` to
``gpaw/siteconfig.py`` and compile the C-code and the GPU kernels with::

  pip install -v -e gpaw/

Now insert the ``export EBU_USER_PREFIX=...`` line and all the ``module load``
lines from above into the start of your ``venv/bin/activate`` script so that
the modules are always loaded when you activate your new environment.

Interactive jobs can be run like this::

  srun -A project_465000538 -p small-g --nodes=1 --ntasks-per-node=2 --gpus-per-node=1 -t 0:30:00 --pty bash

To use Omnitrace, source this file???::

  source /scratch/project_465000538/GPAW/omnitrace-1.10.2-opensuse-15.4-ROCm-50200-PAPI-OMPT-Python3/share/omnitrace/setup-env.sh


Configuring MyQueue
===================

Use the following MyQueue_ :file:`config.py` file:

.. literalinclude:: config.py

and submit jobs like this::

  mq submit job.py -R 128:standard:2h

.. _MyQueue: https://myqueue.readthedocs.io/en/latest/
