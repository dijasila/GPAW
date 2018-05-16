from myqueue.task import task


def workflow():
    return [
        task('lcaotddft_basis.py@1:10m'),
        task('lcaotddft_ag55.py@48:2h', deps=['lcaotddft_basis.py']),
        task('lcaotddft_fig1.py', deps=['lcaotddft_ag55.py']),
        task('lcaotddft.py@4:40m')]
