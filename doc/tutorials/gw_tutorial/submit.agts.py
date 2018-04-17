from q2.job import Job


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

def agts(queue):
    conv = queue.add('C_ecut_k_conv_GW.py', ncpus=8, walltime=20 * 60)
    queue.add('C_ecut_k_conv_plot_GW.py', deps=conv, creates='C_GW.png')
    queue.add('C_ecut_extrap.py', deps=conv, creates='C_GW_k8_extrap.png')

    freq = queue.add('C_frequency_conv.py', walltime=30)
    queue.add('C_frequency_conv_plot.py', deps=freq, creates='C_freq.png')

    queue.add('C_equal_test.py', deps=[conv, freq])

    bn = queue.add('BN_GW0.py', walltime=70)
    queue.add('BN_GW0_plot.py', deps=bn, creates='BN_GW0.png')

    mos2 = queue.add('MoS2_gs_GW.py', walltime=70)
    mos2_gwg = queue.add('MoS2_GWG.py', deps=mos2, ncpus=8, walltime=20)
    queue.add('MoS2_bs_plot.py', deps=mos2_gwg, creates='MoS2_bs.png')
    queue.add('check_gw.py', deps=mos2_gwg)
