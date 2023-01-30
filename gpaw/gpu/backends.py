from gpaw.gpu import parameters
from gpaw.gpu.arrays import HostArrayInterface


class BaseBackend:
    label = 'base'
    device_no = None
    device_ctx = None

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.api = [x for x in dir(self) if not x.startswith('_')]

    @property
    def enabled(self):
        return parameters.enabled

    @enabled.setter
    def enabled(self, x):
        parameters.enabled = bool(x)

    @property
    def debug(self):
        return parameters.debug

    @debug.setter
    def debug(self, x):
        parameters.debug = bool(x)

    @property
    def debug_sync(self):
        return parameters.debug_sync

    @debug_sync.setter
    def debug_sync(self, x):
        parameters.debug_sync = bool(x)

    def init(self, rank=0):
        pass

    def delete(self):
        pass

    def get_context(self):
        return self.device_ctx

    def copy_to_host(self, x):
        raise NotImplementedError

    def copy_to_device(self, x):
        raise NotImplementedError

    def to_host(self, x):
        if self.is_device_array(x):
            return self.copy_to_host(x)
        else:
            return x

    def to_device(self, x):
        if self.is_host_array(x):
            return self.copy_to_device(x)
        else:
            return x

    def is_host_array(self, x):
        return not self.is_device_array(x)

    def is_device_array(self, x):
        raise NotImplementedError

    def memcpy_dtod(self, tgt, src, n):
        raise NotImplementedError

    def synchronize(self):
        pass


class DummyBackend(BaseBackend):
    label = 'dummy'
    array = HostArrayInterface()

    # catch-all method to ignore anything not defined explicitly
    def _pass(self, *args, **kwargs):
        pass

    # accept undefined attributes, but ignore them
    def __getattr__(self, key):
        if self.debug:
            print("DummyBackend: ignoring undefined method '{0}'".format(key))
        return self._pass

    def copy_to_host(self, x):
        return x

    def copy_to_device(self, x):
        return x

    def is_device_array(self, x):
        return self.enabled

    def memcpy_dtod(self, tgt, src, n):
        pass


class HostBackend(BaseBackend):
    label = 'host'
    array = HostArrayInterface()

    def copy_to_host(self, src, tgt=None, stream=None):
        return self.array.copy_to_host(src, tgt=tgt, stream=stream)

    def copy_to_device(self, src, tgt=None, stream=None):
        return self.array.copy_to_device(src, tgt=tgt, stream=stream)

    def is_device_array(self, x):
        return False

    def memcpy_dtod(self, tgt, src, n):
        pass
