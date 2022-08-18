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

    def init(self, rank=0):
        pass

    def delete():
        pass

    def get_context():
        pass
