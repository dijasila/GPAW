import numpy as np


def npp_conv(np_xK, p_xK):
    assert np_xK.shape[-3:] == p_xK.shape[-3:], f"np_xk shape: {np_xK.shape}, p_xk shape: {p_xK.shape}"

    return np_xK * p_xK


def pp_conv(p1_xK, p2_xK, volume):
    assert p1_xK.shape[-3:] == p2_xK.shape[-3:]
    
    return volume * p1_xK * p2_xK


