# necessary MKL libs
libraries += ['mkl_intel_lp64', 'mkl_sequential', 'mkl_core', 'mkl_avx2', 'mkl_def', 'svml']
# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')
include_dirs = os.getenv('INCLUDE').split(':')
