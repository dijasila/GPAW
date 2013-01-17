from time import time
import _gpaw

class Timer:
    def __init__(self):
        self.timers = {}
        self.t0 = {}
        
    def start(self, name, sync=True):
        if sync:
            _gpaw.cuDevSynch()
        self.t0[name] = time()
    
    def end(self, name, sync=True):
        if sync:
            _gpaw.cuDevSynch()
        if name in self.timers.keys():
            self.timers[name] += time() - self.t0[name]
        else:
            self.timers[name] = time() - self.t0[name]

    def get_timing(self,name):
        return self.timers[name]

    def get_tot_timing(self):
        tot = 0
        for key in self.timers.keys():
            tot += self.timers[key]
        return tot
