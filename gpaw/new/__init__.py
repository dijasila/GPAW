from collections import defaultdict
from contextlib import contextmanager
from time import time


def cached_property(method):
    """Quick'n'dirty implementation of cached_property coming in Python 3.8."""
    name = f'__{method.__name__}'

    def new_method(self):
        if not hasattr(self, name):
            setattr(self, name, method(self))
        return getattr(self, name)

    return property(new_method)


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
        total = self.times['Total']
        log()
        for name, t in self.times.items():
            n = int(round(40 * t / total))
            if n == 0:
                bar = '|'
            else:
                bar = '|' + (n - 1) * '-' + '|'
            log(f'Time ({name + "):":12}{t:10.3f} seconds', bar)
