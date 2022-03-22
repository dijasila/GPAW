from collections import defaultdict
from contextlib import contextmanager
from time import time


def prod(iterable):
    result = 1
    for x in iterable:
        result *= x
    return result


def cached_property(method):
    """Quick'n'dirty implementation of cached_property coming in Python 3.8."""
    name = f'__{method.__name__}'

    def new_method(self):
        if not hasattr(self, name):
            setattr(self, name, method(self))
        return getattr(self, name)

    return property(new_method)


def zip_strict(*iterables):
    """From PEP 618."""
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        pass
    if items:
        i = len(items)
        plural = " " if i == 1 else "s 1-"
        msg = f"zip() argument {i+1} is shorter than argument{plural}{i}"
        raise ValueError(msg)
    sentinel = object()
    for i, iterator in enumerate(iterators[1:], 1):
        if next(iterator, sentinel) is not sentinel:
            plural = " " if i == 1 else "s 1-"
            msg = f"zip() argument {i+1} is longer than argument{plural}{i}"
            raise ValueError(msg)


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
            bar = '━' * n
            log(f'Time ({name + "):":12}{t:10.3f} seconds', bar)
