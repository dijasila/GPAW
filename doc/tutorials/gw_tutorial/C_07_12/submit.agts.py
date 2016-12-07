def agts(queue):
    conv = queue.add('C_ecut_k_conv_GW.py', ncpus=8, walltime=10 * 60)
    queue.add('C_ecut_k_conv_plot_GW.py', deps=conv, creates=['C_GW.png'])
    freq = queue.add('C_frequency_conv.py', walltime=30)
    queue.add('C_frequency_conv_plot.py', deps=freq, creates=['C_freq.png'])
    queue.add('C_equal_test.py', deps=[conv, freq])
