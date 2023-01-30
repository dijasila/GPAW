import pytest
from gpaw.response.berryology import get_orbital_magnetization


@pytest.mark.response
def test_fe_bcc(in_tmp_dir, gpw_files):

    calc_file = gpw_files['fe_cheap_pw_wfs']

    # ----- Check results with default parameters. ----------#
    test_M_x_def, test_M_y_def, test_M_z_def \
        = get_orbital_magnetization(calc_file)

    M_x_def = [-2.76987700e-32, 6.77625564e-10, 1.26921221e-03,
               -5.44642998e-10, -5.44642870e-10, -1.26921221e-03,
               -6.77625612e-10, -2.22312060e-03]
    M_y_def = [7.60665920e-31, -6.77625595e-10, 5.44642799e-10,
               -1.26921221e-03, 1.26921221e-03, 5.44642954e-10,
               -6.77625576e-10, 2.22312057e-03]
    M_z_def = [1.37139630e-30, -3.18938286e-03, -1.21292465e-03,
               -1.21292465e-03, -1.21292465e-03, -1.21292465e-03,
               -3.18938286e-03, 6.09181302e-04]

    assert test_M_x_def == pytest.approx(
        M_x_def, rel=1e-12, abs=1e-3)
    assert test_M_y_def == pytest.approx(
        M_y_def, rel=1e-12, abs=1e-3)
    assert test_M_z_def == pytest.approx(
        M_z_def, rel=1e-12, abs=1e-3)

    # ----- Check results with reduced spin-orbit coupling. -------#
    test_M_x_wr_so, test_M_y_wr_so, test_M_z_wr_so \
        = get_orbital_magnetization(calc_file, scale=0.5)

    M_x_wr_so = [3.46095953e-32, -6.14684541e-11, 6.31021165e-04,
                 -2.97312911e-10, -2.97313249e-10, -6.31021165e-04,
                 6.14683833e-11, -1.13285376e-03]
    M_y_wr_so = [2.47346006e-32, 6.14684503e-11, 2.97313019e-10,
                 -6.31021165e-04, 6.31021165e-04, 2.97312691e-10,
                 6.14683171e-11, 1.13285375e-03]
    M_z_wr_so = [-8.02610117e-32, -1.45348416e-03, -6.16515199e-04,
                 -6.16515199e-04, -6.16515199e-04, -6.16515199e-04,
                 -1.45348416e-03, 2.89134048e-04]

    assert test_M_x_wr_so == pytest.approx(
        M_x_wr_so, rel=1e-12, abs=1e-3)
    assert test_M_y_wr_so == pytest.approx(
        M_y_wr_so, rel=1e-12, abs=1e-3)
    assert test_M_z_wr_so == pytest.approx(
        M_z_wr_so, rel=1e-12, abs=1e-3)

    # ---- Check results when spins are oriented along the x-axis. -----#
    test_M_x_s_x, test_M_y_s_x, test_M_z_s_x \
        = get_orbital_magnetization(calc_file, theta=90, phi=0)

    M_x_s_x = [2.23625528e-30, -1.21292465e-03, -1.21292465e-03,
               -3.18938286e-03, -3.18938286e-03, -1.21292465e-03,
               -1.21292465e-03, 6.09181248e-04]
    M_y_s_x = [9.93132815e-31, 1.26921221e-03, -5.44642805e-10,
               6.77625600e-10, -6.77625563e-10, 5.44642852e-10,
               -1.26921221e-03, -2.22312060e-03]
    M_z_s_x = [-4.67493368e-32, -5.44642993e-10, 1.26921221e-03,
               6.77625530e-10, 6.77625527e-10, -1.26921221e-03,
               5.44642990e-10, -2.22312060e-03]

    assert test_M_x_s_x == pytest.approx(
        M_x_s_x, rel=1e-12, abs=1e-3)
    assert test_M_y_s_x == pytest.approx(
        M_y_s_x, rel=1e-12, abs=1e-3)
    assert test_M_z_s_x == pytest.approx(
        M_z_s_x, rel=1e-12, abs=1e-3)

    # ----- Check results when spins are oriented along the y-axis. ----#
    test_M_x_s_y, test_M_y_s_y, test_M_z_s_y \
        = get_orbital_magnetization(calc_file, theta=90, phi=90)

    M_x_s_y = [1.81873705e-30, 1.26921221e-03, 6.77625647e-10,
               -5.44642845e-10, 5.44642964e-10, -6.77625521e-10,
               -1.26921221e-03, -2.22312060e-03]
    M_y_s_y = [9.95981597e-31, -1.21292465e-03, -3.18938286e-03,
               -1.21292465e-03, -1.21292465e-03, -3.18938286e-03,
               -1.21292465e-03, 6.09181302e-04]
    M_z_s_y = [1.71447934e-30, 5.44642787e-10, -6.77625637e-10,
               -1.26921221e-03, 1.26921221e-03, -6.77625553e-10,
               5.44642878e-10, 2.22312057e-03]
    
    assert test_M_x_s_y == pytest.approx(
        M_x_s_y, rel=1e-12, abs=1e-3)
    assert test_M_y_s_y == pytest.approx(
        M_y_s_y, rel=1e-12, abs=1e-3)
    assert test_M_z_s_y == pytest.approx(
        M_z_s_y, rel=1e-12, abs=1e-3)

    # --- Test results when the contribution from 3 bands --- #
    # --- (occupied and unoccupied) are included in matrix elements etc. ---- #

    n1 = 1
    n2 = 6
    test_M_x_n6, test_M_y_n6, test_M_z_n6 \
        = get_orbital_magnetization(calc_file, n1=n1, n2=n2)

    M_x_n6 = [-7.04288559e-45, 1.35633762e-04, -1.10562561e-01,
              -3.15288233e-03, -3.15288804e-03, 1.11595571e-01,
              -1.11511764e-04, -7.87927469e-03]
    M_y_n6 = [3.92252677e-46, 1.11464347e-04, 3.15288804e-03,
              1.11595571e-01, -1.10562561e-01, 3.15288233e-03,
              1.35586345e-04, 7.84404289e-03]
    M_z_n6 = [6.74127924e-46, -1.17532148e-01, -1.15124694e-01,
              -1.16196726e-01, -1.15124694e-01, -1.16196726e-01,
              -1.17532143e-01, 2.30705817e-03]

    assert test_M_x_n6 == pytest.approx(
        M_x_n6, rel=1e-12, abs=1e-1)
    assert test_M_y_n6 == pytest.approx(
        M_y_n6, rel=1e-12, abs=1e-1)
    assert test_M_z_n6 == pytest.approx(
        M_z_n6, rel=1e-12, abs=1e-1)
    
    # ---- Test results when occupation numbers are  ---- #
    # ----- changed such that only 2 bands are occupied ---- #
    mi = 2
    mf = 3
    test_M_x_bandocc2, test_M_y_bandocc2, test_M_z_bandocc2 \
        = get_orbital_magnetization(calc_file, mi=mi, mf=mf)

    M_x_bandocc2 = [6.03860123e-28, -4.41656305e-06, 6.75622496e-02,
                    -3.75404894e-06, -3.75404879e-06, -6.75622496e-02,
                    4.41656305e-06, 2.67070952e-05]
    M_y_bandocc2 = [-7.29449523e-28, 4.41656304e-06, 3.75404879e-06,
                    -6.75622496e-02, 6.75622496e-02, 3.75404893e-06,
                    4.41656305e-06, -2.67070951e-05]
    M_z_bandocc2 = [8.79333319e-29, 1.33821069e+00, -2.60610423e-02,
                    -2.60610423e-02, -2.60610423e-02, -2.60610423e-02,
                    1.33821069e+00, 1.10350807e-03]

    assert test_M_x_bandocc2 == pytest.approx(
        M_x_bandocc2, rel=1e-12, abs=1e-2)
    assert test_M_y_bandocc2 == pytest.approx(
        M_y_bandocc2, rel=1e-12, abs=1e-2)
    assert test_M_z_bandocc2 == pytest.approx(
        M_z_bandocc2, rel=1e-12, abs=1e-2)
