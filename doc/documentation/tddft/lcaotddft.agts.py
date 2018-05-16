from myqueue.job import Job


def workflow():
    return [
        Job('lcaotddft_basis.py@1:10m'),
        Job('lcaotddft_ag55.py@48:2h', deps=['lcaotddft_basis.py']),
        Job('lcaotddft_fig1.py', deps=['lcaotddft_ag55.py']),
        Job('lcaotddft.py@4:40m')]
