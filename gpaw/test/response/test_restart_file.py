import pytest
from gpaw.response.g0w0 import G0W0


@pytest.mark.response
def test_do_GW_too(in_tmp_dir, gpw_files):
    restart = False
    gw = G0W0(gpw_files['bn_pw_wfs'],
              bands=(3, 5),
              nbands=9,
              nblocks=1,
              ecut=40,
              kpts=[0, 1],
              restartfile='restartfile')

    gw.calculate()
    gw.save_restart_file(3)
    restart = gw.load_restart_file()
    assert restart
