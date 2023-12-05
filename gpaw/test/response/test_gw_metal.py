from gpaw.response.g0w0 import G0W0
import pytest


@pytest.mark.response
def test_gw_metal(gpw_files):
    gpwfile = gpw_files['srvo3_pw']
    with pytest.raises(NotImplementedError) as excinfo:
        _ = G0W0(gpwfile, nbands=15)

    errormsg = ('The current GW implementation cannot'
                ' handle intraband screening and is'
                ' therefore not applicable to metals')
    assert str(excinfo.value) == errormsg
