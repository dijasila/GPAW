# creates: na2_td_Ffe.png  na2_td_Frho.png na2_casida_Ffe.png na2_casida_Frho.png
import os
from sys import executable
assert os.system('%s timepropagation_calculate.py' % executable) == 0
assert os.system('%s timepropagation_continue.py' % executable) == 0
assert os.system('%s timepropagation_postprocess.py' % executable) == 0
assert os.system('%s timepropagation_plot.py' % executable) == 0
os.system('cp na2_td_spectrum.png na2_td_Ffe.png na2_td_Frho.png ../../../_build')

assert os.system('%s casida_calculate.py' % executable) == 0
assert os.system('%s casida_postprocess.py' % executable) == 0
assert os.system('%s casida_plot.py' % executable) == 0

os.system('cp na2_casida_Ffe.png na2_casida_Frho.png ../../../_build')
