from gpaw.xc.rpa import RPACorrelation

rpa1 = RPACorrelation('bulk.all.gpw', txt='si.rpa.rpa_output.txt',
                      ecut=[164.0, 140.0, 120.0, 100.0, 80.0])
E1_i = rpa1.calculate()
