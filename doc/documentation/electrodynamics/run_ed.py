# creates: qsfdtd_vs_mie.png
import os
from sys import executable
assert os.system('%s gold_nanosphere_calculate.py' % executable) == 0
assert os.system('%s plot.py' % executable) == 0
os.system('cp qsfdtd_vs_mie.png ../../_build')
