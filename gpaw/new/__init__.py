from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from time import time


def cached_property(method):
    """Quick'n'dirty implementation of cached_property coming in Python 3.8."""
    return property(lru_cache(maxsize=None)(method))


class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.times['Total'] = -time()

    @contextmanager
    def __call__(self, name):
        t1 = time()
        try:
            yield
        finally:
            t2 = time()
            self.times[name] += t2 - t1

    def write(self, log):
        self.times['Total'] += time()
        log('\n' +
            '\n'.join(f'Time ({name + "):":12}{t:10.3f} seconds'
                      for name, t in self.times.items()))
