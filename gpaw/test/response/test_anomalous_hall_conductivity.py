import pytest
from gpaw.response.berryology import get_hall_conductivity


@pytest.mark.response
def test_fe_bcc(in_tmp_dir, gpw_files):

    calc_file = gpw_files['fe2_pw_wfs']

    # ----------- Check results with default parameters. ------- #
    test_sigma_xy_def, test_sigma_yz_def, test_sigma_zx_def \
        = get_hall_conductivity(calc_file)

    sigma_xy_def = [-7.03325011e-32, 2.55683148e-07, 3.47873186e-07,
                    3.47873186e-07, 3.47873186e-07, 3.47873186e-07,
                    2.55683148e-07, 6.52075901e-08]
    sigma_yz_def = [6.95671942e-32, -4.70688597e-12, -2.72226471e-07,
                    -6.85077200e-12, -6.85077190e-12, 2.72226471e-07,
                    4.70688593e-12, -1.04109688e-07]
    sigma_zx_def = [5.01550705e-32, 4.70688597e-12, 6.85077195e-12,
                    2.72226471e-07, -2.72226471e-07, 6.85077200e-12,
                    4.70688588e-12, 1.04109687e-07]

    assert test_sigma_xy_def == pytest.approx(
        sigma_xy_def, rel=1e-12, abs=1e-6)
    assert test_sigma_yz_def == pytest.approx(
        sigma_yz_def, rel=1e-12, abs=1e-6)
    assert test_sigma_zx_def == pytest.approx(
        sigma_zx_def, rel=1e-12, abs=1e-6)

    # ---- Check results with reduced spin-orbit coupling. ----- #
    test_sigma_xy_wr_so, test_sigma_yz_wr_so, test_sigma_zx_wr_so \
        = get_hall_conductivity(calc_file, scale=0.5)

    sigma_xy_wr_so = [2.65945571e-35, 9.09123982e-21, -1.51837123e-21,
                      -8.20896458e-21, -2.39734970e-21, -3.15988374e-21,
                      -3.30625648e-21, 5.45657831e-18]
    sigma_yz_wr_so = [-2.34745867e-35, -1.22830874e-21, 1.50917611e-21,
                      4.12587328e-22, -1.15692559e-20, -3.16713755e-21,
                      1.21494356e-20, -2.41266576e-18]
    sigma_zx_wr_so = [1.09714122e-34, 1.23861601e-21, -1.69421491e-20,
                      -8.21831398e-21, 2.38723898e-21, 5.37227383e-21,
                      1.21423014e-20, -7.90097766e-18]

    assert test_sigma_xy_wr_so == pytest.approx(
        sigma_xy_wr_so, rel=1e-12, abs=1e-6)
    assert test_sigma_yz_wr_so == pytest.approx(
        sigma_yz_wr_so, rel=1e-12, abs=1e-6)
    assert test_sigma_zx_wr_so == pytest.approx(
        sigma_zx_wr_so, rel=1e-12, abs=1e-6)

    # ----- Check results when M is oriented along the x-axis. ------------- #
    test_sigma_xy_M_x, test_sigma_yz_M_x, test_sigma_zx_M_x \
        = get_hall_conductivity(calc_file, theta=90, phi=0)

    sigma_xy_M_x = [2.02056944e-32, -6.85077191e-12,
                    -2.72226471e-07, -4.70688596e-12,
                    -4.70688592e-12, 2.72226471e-07,
                    6.85077194e-12, -1.04109688e-07]
    sigma_yz_M_x = [-4.69041620e-33, 3.47873186e-07,
                    3.47873186e-07, 2.55683148e-07,
                    2.55683148e-07, 3.47873186e-07,
                    3.47873186e-07, 6.52075888e-08]
    sigma_zx_M_x = [2.11868104e-32, -2.72226471e-07,
                    -6.85077201e-12, -4.70688590e-12,
                    4.70688592e-12, 6.85077202e-12,
                    2.72226471e-07, -1.04109688e-07]

    assert test_sigma_xy_M_x == pytest.approx(
        sigma_xy_M_x, rel=1e-12, abs=1e-6)
    assert test_sigma_yz_M_x == pytest.approx(
        sigma_yz_M_x, rel=1e-12, abs=1e-6)
    assert test_sigma_zx_M_x == pytest.approx(
        sigma_zx_M_x, rel=1e-12, abs=1e-6)

    # ------- Check results when M is oriented along the y-axis. ------ #
    test_sigma_xy_M_y, test_sigma_yz_M_y, test_sigma_zx_M_y \
        = get_hall_conductivity(calc_file, theta=90, phi=90)

    sigma_xy_M_y = [8.42146901e-32, 6.85077202e-12, 4.70688589e-12,
                    2.72226471e-07, -2.72226471e-07, 4.70688593e-12,
                    6.85077193e-12, 1.04109687e-07]
    sigma_yz_M_y = [-3.20258099e-32, -2.72226471e-07, -4.70688585e-12,
                    -6.85077203e-12, 6.85077200e-12, 4.70688586e-12,
                    2.72226471e-07, -1.04109688e-07]
    sigma_zx_M_y = [-7.88077284e-32, 3.47873186e-07, 2.55683148e-07,
                    3.47873186e-07, 3.47873186e-07, 2.55683148e-07,
                    3.47873186e-07, 6.52075902e-08]
    
    assert test_sigma_xy_M_y == pytest.approx(
        sigma_xy_M_y, rel=1e-12, abs=1e-6)
    assert test_sigma_yz_M_y == pytest.approx(
        sigma_yz_M_y, rel=1e-12, abs=1e-6)
    assert test_sigma_zx_M_y == pytest.approx(
        sigma_zx_M_y, rel=1e-12, abs=1e-6)

    # ----- Test results when the contribution from 3 bands ------ #
    # -----(occupied and unoccupied) are included in matrix elements etc. --- #
    n1 = 1
    n2 = 6

    test_sigma_xy_n6, test_sigma_yz_n6, test_sigma_zx_n6 \
        = get_hall_conductivity(calc_file, n1=n1, n2=n2)

    sigma_xy_n6 = [-1.57755951e-49, 3.83301771e-04, 2.98900587e-04,
                   3.02384790e-04, 2.98900587e-04, 3.02384790e-04,
                   3.83301756e-04, 9.56402602e-07]
    sigma_yz_n6 = [1.82606166e-48, -5.54517452e-07, 3.01298109e-04,
                   1.21287525e-06, 1.21289425e-06, -3.04773305e-04,
                   5.68102614e-07, -4.30542259e-07]
    sigma_zx_n6 = [-1.03075846e-49, -5.67835032e-07, -1.21289425e-06,
                   -3.04773305e-04, 3.01298109e-04, -1.21287525e-06,
                   -5.54249870e-07, 4.17036218e-07]
    
    assert test_sigma_xy_n6 == pytest.approx(
        sigma_xy_n6, rel=1e-12, abs=1e-5)
    assert test_sigma_yz_n6 == pytest.approx(
        sigma_yz_n6, rel=1e-12, abs=1e-5)
    assert test_sigma_zx_n6 == pytest.approx(
        sigma_zx_n6, rel=1e-12, abs=1e-5)
    
    # ------ Test results when occupation numbers  ----------#
    # ------ are changed such that only 2 bands are occupied -------- #
    mi = 2
    mf = 3

    test_sigma_xy_bandocc2, test_sigma_yz_bandocc2, test_sigma_zx_bandocc2 \
        = get_hall_conductivity(calc_file, mi=mi, mf=mf)

    sigma_xy_bandocc2 = [-9.42778557e-33, -3.48313049e-04, 6.83259931e-06,
                         6.83259931e-06, 6.83259931e-06, 6.83259931e-06,
                         -3.48313049e-04, 8.26705496e-08]
    sigma_yz_bandocc2 = [-6.45456155e-32, 1.14912373e-09, -1.78199461e-05,
                         9.88020731e-10, 9.88020692e-10, 1.78199461e-05,
                         -1.14912373e-09, 2.52643923e-09]
    sigma_zx_bandocc2 = [7.79251365e-32, -1.14912372e-09, -9.88020693e-10,
                         1.78199461e-05, -1.78199461e-05, -9.88020729e-10,
                         -1.14912373e-09, -2.52643925e-09]

    assert test_sigma_xy_bandocc2 == pytest.approx(
        sigma_xy_bandocc2, rel=1e-12, abs=1e-5)
    assert test_sigma_yz_bandocc2 == pytest.approx(
        sigma_yz_bandocc2, rel=1e-12, abs=1e-5)
    assert test_sigma_zx_bandocc2 == pytest.approx(
        sigma_zx_bandocc2, rel=1e-12, abs=1e-5)
