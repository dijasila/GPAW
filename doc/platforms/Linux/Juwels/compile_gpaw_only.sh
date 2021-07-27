#! /bin/bash
# Run this script for installing GPAW only, without a virtual environment
# It should be run from the gpaw repositories base directory
# It is important that the PYTHONPATH to ASE is set before the installation
# is started

module purge
module load StdEnv/2020
module load Python/3.8.5
module load SciPy-Stack/2020-Python-3.8.5
module load intel-para
module load FFTW/3.3.8
module load libxc/4.3.4
module load ELPA/2020.05.001

cp doc/platforms/Linux/Juwels/siteconfig_juwels.py siteconfig.py

python setup.py build
