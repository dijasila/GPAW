def get_nonlocal_functional(name,
                            metallic=False,
                            width=None):
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

    scr_functional = None
    xc_functional = None
    response = True
    setup_name = None

    # Set parameters based on the name
    if name == 'GLLB':
        scr_functional = 'GGA_X_B88'
    elif name == 'GLLBM':
        scr_functional = 'GGA_X_B88'
        setup_name = 'GLLB'
        metallic = True
    elif name == 'GLLBSC':
        scr_functional = 'GGA_X_PBE_SOL'
        xc_functional = 'GGA_C_PBE_SOL'
        setup_name = 'GLLBSC'
    elif name == 'GLLBSCM':
        scr_functional = 'GGA_X_PBE_SOL'
        xc_functional = 'GGA_C_PBE_SOL'
        setup_name = 'GLLBSC'
        metallic = True
    elif name.startswith('GLLBSC_W'):
        scr_functional = 'GGA_X_PBE_SOL'
        xc_functional = 'GGA_C_PBE_SOL'
        setup_name = 'GLLBSC'
        width = float(name.split('GLLBSC_W')[1])
    elif name.startswith('GLLBSCM_W'):
        scr_functional = 'GGA_X_PBE_SOL'
        xc_functional = 'GGA_C_PBE_SOL'
        setup_name = 'GLLBSC'
        metallic = True
        width = float(name.split('GLLBSCM_W')[1])
    elif name == 'GLLBC':
        scr_functional = 'GGA_X_PBE'
        xc_functional = 'GGA_C_PBE'
    elif name == 'GLLBCP86':
        scr_functional = 'GGA_X_B88'
        xc_functional = 'GGA_C_P86'
    elif name == 'GLLBLDA':
        xc_functional = 'LDA'
        setup_name = 'LDA'
    elif name == 'GLLBGGA':
        xc_functional = 'PBE'
        setup_name = 'PBE'
    elif name == 'GLLBNORESP':
        scr_functional = 'GGA_X_B88'
        response = False
    else:
        raise RuntimeError('Unkown nonlocal density functional: ' + name)

    # Construct functional
    functional = NonLocalFunctional(name, setup_name=setup_name)
    if scr_functional is not None:
        scr = C_GLLBScr(functional, 1.0, functional=scr_functional,
                        width=width, metallic=metallic)
        if response:
            C_Response(functional, 1.0, scr.get_coefficient_calculator())
    if xc_functional is not None:
        C_XC(functional, 1.0, functional=xc_functional)
    return functional
