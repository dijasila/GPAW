from __future__ import print_function, division
import sys
import functools
from time import time, sleep

from ase.utils import devnull


class ProgressBar:
    def __init__(self, fd=sys.stdout):
        """Progress-bar.
        
        Usage::
            
            pb = ProgressBar()
            for i in range(10):
                pb.update(i / 10.0)
                do_stuff()
            pb.finish()
        """
        self.fd = fd
        
        try:
            self.tty = fd.isatty()
        except AttributeError:
            self.tty = False
        
        self.done = False
        self.n = None
        self.t = time()
        
        print('ETA: ', end='', file=fd)
        fd.flush()
        
    def update(self, x):
        """Update progress-bar (0 <= x <= 1)."""
        if x == 0 or self.done:
            return
        N = 50
        n = int(N * x)
        t = time() - self.t
        p = functools.partial(print, file=self.fd)
        if self.tty:
            p('\rETA: {0:.3f}s |{1:50}| Time: {2:.3f}s'
              .format(t / x, '-' * n, t), end='')
            if x == 1:
                p()
                self.done = True
            self.fd.flush()
        else:
            if self.n is None:
                p('{0:.3f}s |'.format(t / x), end='')
                self.n = 0
            if n > self.n:
                p('-' * (n - self.n), end='')
                self.fd.flush()
                self.n = n
            if x == 1:
                p('| Time: {0:.3f}s'.format(t))
                self.fd.flush()
                self.done = True
                
    def finish(self):
        self.update(1)

        
def test():
    for fd in [sys.stdout, devnull, open('pb.txt', 'w')]:
        print(fd)
        pb = ProgressBar(fd)
        for i in range(20):
            pb.update(i / 20)
            sleep(0.03)
            pb.update((i + 1) / 20)
        pb.finish()


if __name__ == '__main__':
    test()
