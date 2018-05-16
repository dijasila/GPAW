from myqueue.job import Job


def workflow():
    return [
        Job('Pt_gs.py@4:20m'),
        Job('Pt_bands.py@32:1h', deps=['Pt_gs.py']),
        Job('plot_Pt_bands.py@1:10m', deps=['Pt_bands.py']),
        Job('WS2_gs.py@4:20h'),
        Job('WS2_bands.py@32:3h', deps=['WS2_gs.py']),
        Job('plot_WS2_bands.py@1:10m', deps=['WS2_bands.py']),
        Job('Fe_gs.py@4:20m'),
        Job('Fe_bands.py@32:1h', deps=['Fe_gs.py']),
        Job('plot_Fe_bands.py@1:10m', deps=['Fe_bands.py']),
        Job('gs_Bi2Se3.py@4:40m'),
        Job('Bi2Se3_bands.py@32:5h', deps=['gs_Bi2Se3.py']),
        Job('high_sym.py@4:30h', deps=['gs_Bi2Se3.py']),
        Job('parity.py@1:5h', deps=['high_sym.py']),
        Job('plot_Bi2Se3_bands.py@1:2h', deps=['Bi2Se3_bands.py']),
        Job('gs_Co.py@32:2h'),
        Job('anisotropy.py@1:5h', deps=['gs_Co.py']),
        Job('plot_anisotropy.py@1:2m', deps=['anisotropy.py'])]
