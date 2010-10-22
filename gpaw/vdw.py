def FFTVDWFunctional(**kwargs):
    import warnings
    warnings.warn('Please import FFTVDWFunctional from gpaw.xc.vdw.', 
                  DeprecationWarning)
    from gpaw.xc.vdw import FFTVDWFunctional as XC
    return XC(**kwargs)
