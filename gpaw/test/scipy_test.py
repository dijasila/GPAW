import sys
from scipy import test
_stdout = sys.stdout
_stderr = sys.stderr
# scipy tests write to stderr
sys.stderr = sys.stdout
assert test().wasSuccessful()
sys.stdout = _stdout
sys.stderr = _stderr
