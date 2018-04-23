from myqueue.job import Job


def workflow():
    return [
        Job('C_ecut_k_conv_GW.py@8x20m'),
        Job('C_ecut_k_conv_plot_GW.py', deps=['C_ecut_k_conv_GW.py']),
        Job('C_ecut_extrap.py', deps=['C_ecut_k_conv_GW.py']),
        Job('C_frequency_conv.py@1x30s'),
        Job('C_frequency_conv_plot.py', deps=['C_frequency_conv.py']),
        Job('C_equal_test.py', deps=['C_ecut_k_conv_GW.py', 'C_frequency_conv.py']),
        Job('BN_GW0.py@1x1m'),
        Job('BN_GW0_plot.py', deps=['BN_GW0.py']),
        Job('MoS2_gs_GW.py@1x1m'),
        Job('MoS2_GWG.py@8x20s', deps=['MoS2_gs_GW.py']),
        Job('MoS2_bs_plot.py', deps=['MoS2_GWG.py']),
        Job('check_gw.py', deps=['MoS2_GWG.py'])]
