def agts(queue):
    conv = queue.add('Si_ecut_k_conv_GW.py', ncpus=8, walltime=25 * 60)
    queue.add('Si_ecut_k_conv_plot_GW.py', deps=conv, creates=['Si_GW.png'])
    freq = queue.add('Si_frequency_conv.py', ncpus=8, walltime=10 * 60)
    queue.add('Si_frequency_conv_plot.py', deps=freq, creates=['Si_freq.png'])
    queue.add('Si_equal_test.py', deps=[conv, freq])
