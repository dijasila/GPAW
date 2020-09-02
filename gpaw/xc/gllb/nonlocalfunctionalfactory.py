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
    from gpaw.xc.gllb.c_gllbscr import C_GLLBScr
    from gpaw.xc.gllb.c_response import C_Response
    from gpaw.xc.gllb.c_xc import C_XC

    functional = NonLocalFunctional(name)

    if name == 'GLLB':
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0).get_coefficient_calculator())
        return functional
    elif name == 'GLLBM':
        C_Response(functional, 1.0, C_GLLBScr(
            functional, 1.0, metallic=True).get_coefficient_calculator())
        return functional
    elif name.startswith('GLLBSC'):
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
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0, 'GGA_X_PBE')
                   .get_coefficient_calculator())
        C_XC(functional, 1.0, 'GGA_C_PBE')
        return functional
    elif name == 'GLLBCP86':
        C_Response(functional, 1.0,
                   C_GLLBScr(functional, 1.0).get_coefficient_calculator())
        C_XC(functional, 1.0, 'GGA_C_P86')
        return functional
    elif name == 'GLLBLDA':
        C_XC(functional, 1.0,'LDA')
        return functional
    elif name == 'GLLBGGA':
        C_XC(functional, 1.0,'PBE')
        return functional
    elif name == 'GLLBNORESP':
        C_GLLBScr(functional, 1.0)
        return functional
    else:
        raise RuntimeError('Unkown NonLocal density functional: ' + name)
