from gpaw.gpu import cupy_is_fake

if not cupy_is_fake:
    from _gpaw_gpu import pwlfc_expand_gpu as pwlfc_expand
