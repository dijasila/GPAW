from gpaw.response.df import DielectricFunction

df = DielectricFunction('gs_MoS2.gpw',
                        ecut=100,
                        frequencies=(0.,),
                        nbands=50,
                        intraband=False,
                        hilbert=False,
                        eta=0.1)

alpha = df.get_polarizability(filename=None)[1][0].real
print('alpha = ', alpha, 'AA')
