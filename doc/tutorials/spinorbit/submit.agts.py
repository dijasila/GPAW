def agts(queue):
    Pt_gs = queue.add('Pt_gs.py', ncpus=4, walltime=2000)
    Pt_bands = queue.add('Pt_bands.py', deps=Pt_gs, ncpus=32, walltime=10000)
    plot_Pt = queue.add('plot_Pt_bands.py', ncpus=1, deps=Pt_bands, walltime=1000, 
                        creates='Pt_bands.png')
    WS2_gs = queue.add('WS2_gs.py', ncpus=4, walltime=2000)
    WS2_bands = queue.add('WS2_bands.py', deps=WS2_gs, ncpus=32, walltime=10000)
    plot_WS2 = queue.add('plot_WS2_bands.py', ncpus=1, deps=WS2_bands, walltime=1000, 
                         creates='WS2_bands.png')
    Fe_gs = queue.add('Fe_gs.py', ncpus=4, walltime=2000)
    Fe_bands = queue.add('Fe_bands.py', deps=Fe_gs, ncpus=32, walltime=10000)
    plot_Fe = queue.add('plot_Fe_bands.py', ncpus=1, deps=Fe_bands, walltime=1000, 
                        creates='Fe_bands.png')
