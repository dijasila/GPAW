from gpaw.xc.rpa import RPACorrelation

rpa = RPACorrelation('H.ralda.lda_wfcs.gpw',
                     ecut=300,
                     txt='H.ralda_02_rpa_at_lda.output.txt')
rpa.calculate()
