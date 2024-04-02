from gpaw.xc.rpa import RPACorrelation

rpa = RPACorrelation('H.ralda.pbe_wfcs.gpw',
                     ecut=300,
                     txt='H.ralda_05_rpa_at_pbe.output.txt')
rpa.calculate()
