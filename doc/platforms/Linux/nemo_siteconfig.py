# necessary MKL libs
libraries += ['mkl_intel_lp64', 'mkl_sequential', 'mkl_core', 'svml']
# Now add a EasyBuild "cover-all-bases" library_dirs
library_dirs = os.getenv('LD_LIBRARY_PATH').split(':')
