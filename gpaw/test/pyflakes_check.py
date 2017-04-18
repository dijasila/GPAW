import subprocess
import sys
# Run flake8 on code in GPAW:

# Errors to ignore
ignore = ['E',  # pep8 errors
          'W',  # pep8 warnings
          'F812',  # list comprehension redefines
          'F811',  # redefinition of unused variable
          'F841']  # variable is assigned to but never used
try:
    output = subprocess.check_output(['flake8',
                                      '--ignore={}'.format(','.join(ignore)),
                                      'gpaw'])
except subprocess.CalledProcessError as ex:
    output = ex.output.decode()

lines = []
for line in output.splitlines():
    # Exclude these files and directories:
    exclude = ['doc/',
               'gpaw/analyse/wignerseitz.py',
               'gpaw/atom/configurations.py',
               'gpaw/coding_style.py',
               'gpaw/dfpt/',
               'gpaw/factory.py',
               'gpaw/fdtd/polarizable_material.py',
               'gpaw/lcao/pwf2.py',
               'gpaw/lcaotddft/split.py',
               'gpaw/spherical_harmonics.py',
               'gpaw/test/big/',
               'gpaw/test/parallel',
               'gpaw/test/solvation/solvation_api.py',
               'gpaw/utilities/blas.py',
               'gpaw/wavefunctions/mode.py']
    for txt in exclude:
        if txt in line:
            break
    else:  # no break
        lines.append(line)
if lines:
    print('\n'.join(lines))
    sys.exit(1)
