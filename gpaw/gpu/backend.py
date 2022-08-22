class DummyBackend:
    enabled = False
    debug = False
    debug_sync = False
    use_hybrid_blas = False
    device_no = None
    device_ctx = None

    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    # catch-all method to ignore anything not defined explicitly
    def _pass(self, *args, **kwargs):
        pass

    def __getattr__(self, key):
        if self.debug:
            print("DummyBackend: ignoring undefined method '{0}'".format(key))
        return self._pass
