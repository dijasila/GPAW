from q2.job import Job


def workflow():
    return [
        Job('Pt_gs.py@4x20s'),
        Job('Pt_bands.py@32x1m', deps=['Pt_gs.py']),
        Job('plot_Pt_bands.py@1x10s', deps=['Pt_bands.py']),
        Job('WS2_gs.py@4x20s'),
        Job('WS2_bands.py@32x3m', deps=['WS2_gs.py']),
        Job('plot_WS2_bands.py@1x10s', deps=['WS2_bands.py']),
        Job('Fe_gs.py@4x20s'),
        Job('Fe_bands.py@32x1m', deps=['Fe_gs.py']),
        Job('plot_Fe_bands.py@1x10s', deps=['Fe_bands.py']),
        Job('gs_Bi2Se3.py@4x40s'),
        Job('Bi2Se3_bands.py@32x5m', deps=['gs_Bi2Se3.py']),
        Job('high_sym.py@4x30s', deps=['gs_Bi2Se3.py']),
        Job('parity.py@1x5s', deps=['high_sym.py']),
        Job('plot_Bi2Se3_bands.py@1x2s', deps=['Bi2Se3_bands.py']),
        Job('gs_Co.py@32x2m'),
        Job('anisotropy.py@1x5m', deps=['gs_Co.py']),
        Job('plot_anisotropy.py@1x2s', deps=['anisotropy.py'])]
