import sys
from numpy import test
_stdout = sys.stdout
_stderr = sys.stderr
# numpy tests write to stderr
sys.stderr = sys.stdout
assert test().wasSuccessful()
sys.stdout = _stdout
sys.stderr = _stderr
