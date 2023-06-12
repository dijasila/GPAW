from pathlib import Path

from myqueue.workflow import run


def workflow():
    with run(script='H2_instability.py', cores=8):
        run(function=check_instability)
    with run(script='tPP.py', cores=8, tmax='1h'):
        run(function=check_tPP)
    with run(script='ethylene.py', cores=8):
        run(function=check_ethylene)


def check_ethylene():
    text = Path('Ethylene_EX_DO-GMF.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Extrapolated:'):
            gmf = float(line.split()[-1])
    assert abs(gmf + 18.783) < 0.01


def check_instability():
    get = 10
    text = Path('davidson_H2_S.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Eigenvalues:'):
            get = 0
            continue
        get += 1
        if get == 3:
            temp = line.split()
            eigv_s = [float(temp[0]), float(temp[1])]
        if get == 8:
            temp = line.split()
            if temp[0][-1] == 'c' and temp[1][-1] == 'c':
                break

    get = 10
    text = Path('davidson_H2_BS.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Eigenvalues:'):
            get = 0
            continue
        get += 1
        if get == 3:
            temp = line.split()
            eigv_bs = [float(temp[0]), float(temp[1])]
        if get == 8:
            temp = line.split()
            if temp[0][-1] == 'c' and temp[1][-1] == 'c':
                break

    assert abs(eigv_s[0] + 0.118) < 0.01
    assert abs(eigv_s[1] - 0.891) < 0.01
    assert abs(eigv_bs[0] - 0.198) < 0.01
    assert abs(eigv_bs[1] - 0.492) < 0.01


def check_tPP():
    text = Path('N-Phenylpyrrole_EX_DO-GMF.txt').read_text()
    for line in text.splitlines():
        if line.startswith('Dipole moment:'):
            gmf = float(line.split()[-2].replace(')', ''))
    assert abs(gmf * 4.803 + 10.227) < 0.01
