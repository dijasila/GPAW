def get_nonlocal_functional(name):
    """Function for building GLLB functionals.

    Recognized names and implied parameters:
    * GLLB (Contains screening part from GGA functional
            and response part based on simple square root expression
            of orbital energy differences)
    * GLLBC (GLLB with screening part from PBE + PBE Correlation)
    * GLLBSC (GLLB with screening part from PBE_SOL + PBE Correlation)
    * GLLBNORESP (Just GLLB Screening)
    * GLLBLDA (A test functional, which is just LDA but via
               NonLocalFunctional framework)
    * GLLBGGA (A test functional, which is just GGA but via
               NonLocalFunctional framework)
    """
    from gpaw.xc.gllb.nonlocalfunctional import NonLocalFunctional
    functional = NonLocalFunctional(name)

    if name == 'GLLB':
        from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
        from gpaw.xc.gllb.c_response import C_Response
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0).get_coefficient_calculator())
        return functional
    elif name == 'GLLBM':
        from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
        from gpaw.xc.gllb.c_response import C_Response
        from gpaw.xc.gllb.c_xc import C_XC
        C_Response(functional, 1.0, C_GLLBScr(
            functional, 1.0, metallic=True).get_coefficient_calculator())
        return functional
    elif name.startswith('GLLBSC'):
        from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
        from gpaw.xc.gllb.c_response import C_Response
        from gpaw.xc.gllb.c_xc import C_XC

        if name == 'GLLBSC':
            kwargs = dict()
        elif name == 'GLLBSCM':
            kwargs = dict(metallic=True)
        elif name.startswith('GLLBSC_W'):
            kwargs = dict(width=float(name.split('GLLBSC_W')[1]))
        elif name.startswith('GLLBSCM_W'):
            kwargs = dict(metallic=True,
                          width=float(name.split('GLLBSCM_W')[1]))

        functional = NonLocalFunctional(name, setup_name='GLLBSC')
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0, 'GGA_X_PBE_SOL', **kwargs)
                   .get_coefficient_calculator())
        C_XC(functional, 1.0, 'GGA_C_PBE_SOL')
        return functional
    elif name == 'GLLBC':
        from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
        from gpaw.xc.gllb.c_response import C_Response
        from gpaw.xc.gllb.c_xc import C_XC
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0, 'GGA_X_PBE')
                   .get_coefficient_calculator())
        C_XC(functional, 1.0, 'GGA_C_PBE')
        return functional
    elif name == 'GLLBCP86':
        from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
        from gpaw.xc.gllb.c_response import C_Response
        from gpaw.xc.gllb.c_xc import C_XC
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0).get_coefficient_calculator())
        C_XC(functional, 1.0, 'GGA_C_P86')
        return functional
    elif name == 'GLLBLDA':
        from gpaw.xc.gllb.c_xc import C_XC
        C_XC(functional, 1.0,'LDA')
        return functional
    elif name == 'GLLBGGA':
        from gpaw.xc.gllb.c_xc import C_XC
        C_XC(functional, 1.0,'PBE')
        return functional
    elif name == 'GLLBNORESP':
        from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
        C_GLLBScr(functional, 1.0)
        return functional
    elif name == 'KLI':
        raise RuntimeError('KLI functional not implemented')
        from gpaw.xc.gllb.c_slater import C_Slater
        from gpaw.xc.gllb.c_response import C_Response
        C_Response(functional, 1.0,
                   C_Slater(functional, 1.0).get_coefficient_calculator())
        return functional
    else:
        raise RuntimeError('Unkown NonLocal density functional: ' + name)
