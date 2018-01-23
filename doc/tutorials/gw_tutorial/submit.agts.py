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
    mos2_gwg = queue.add('MoS2_GWG.py', ncpus=8, walltime=20)
    queue.add('MoS2_bs_plot.py', deps=[mos2, mos2_gwg], creates='MoS2_bs.png')
    queue.add('check_gw.py', deps=[mos2, mos2_gwg])
