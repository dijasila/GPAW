from q2.job import Job


def workflow():
    return [
        Job('water/h2o.py'),
        Job('wavefunctions/CO.py@8x15s'),
        Job('aluminium/Al_fcc.py@2x15s'),
        Job('aluminium/Al_bcc.py@2x15s'),
        Job('aluminium/Al_fcc_vs_bcc.py@2x15s'),
        Job('aluminium/Al_fcc_modified.py@2x15s'),
        Job('diffusion/initial.py@2x15s'),
        Job('diffusion/solution.py@2x15s'),
        Job('diffusion/densitydiff.py', deps=['diffusion/solution.py']),
        Job('wannier/si.py@8x15s'),
        Job('wannier/wannier-si.py', deps=['wannier/si.py']),
        Job('wannier/benzene.py@8x15s'),
        Job('wannier/wannier-benzene.py', deps=['wannier/benzene.py']),
        Job('band_structure/ag.py'),
        Job('vibrations/h2o.py@8x15s'),
        Job('vibrations/H2O_vib.py@8x15s', deps=['vibrations/h2o.py']),
        Job('vibrations/H2O_vib_2.py@4x15s', deps=['vibrations/H2O_vib.py']),
        Job('iron/ferro.py@4x15s'),
        Job('iron/anti.py@4x15s'),
        Job('iron/non.py@2x15s'),
        Job('iron/PBE.py', deps=['iron/ferro.py', 'iron/anti.py', 'iron/non.py']),
        Job('eels/test.py', deps=['band_structure/ag.py']),
        Job('gw/test.py'),
        Job('rpa/si.pbe.py'),
        Job('rpa/si.pbe+exx.py@4x15s', deps=['rpa/si.pbe.py']),
        Job('rpa/si.rpa_init_pbe.py'),
        Job('rpa/si.rpa.py@4x15s', deps=['rpa/si.rpa_init_pbe.py']),
        Job('stress/con_pw.py'),
        Job('stress/stress.py'),
        Job('transport/pt_h2_tb_transport.py'),
        Job('transport/pt_h2_lcao_manual.py'),
        Job('transport/pt_h2_lcao_transport.py', deps=['transport/pt_h2_lcao_manual.py'])]

def agts(queue):
    queue.add('water/h2o.py', ncpus=1)
    queue.add('wavefunctions/CO.py', ncpus=8)
    queue.add('aluminium/Al_fcc.py', ncpus=2)
    queue.add('aluminium/Al_bcc.py', ncpus=2)
    queue.add('aluminium/Al_fcc_vs_bcc.py', ncpus=2)
    queue.add('aluminium/Al_fcc_modified.py', ncpus=2)
    queue.add('diffusion/initial.py', ncpus=2)
    sol = queue.add('diffusion/solution.py', ncpus=2)
    queue.add('diffusion/densitydiff.py', deps=[sol])
    si = queue.add('wannier/si.py', ncpus=8)
    queue.add('wannier/wannier-si.py', deps=[si])
    benzene = queue.add('wannier/benzene.py', ncpus=8)
    queue.add('wannier/wannier-benzene.py', deps=[benzene])
    band = queue.add('band_structure/ag.py', ncpus=1, creates='Ag.png')
    h2o = queue.add('vibrations/h2o.py', ncpus=8)
    h2ovib = queue.add('vibrations/H2O_vib.py', ncpus=8, deps=[h2o])
    queue.add('vibrations/H2O_vib_2.py', ncpus=4, deps=[h2ovib])
    ferro = queue.add('iron/ferro.py', ncpus=4)
    anti = queue.add('iron/anti.py', ncpus=4)
    non = queue.add('iron/non.py', ncpus=2)
    queue.add('iron/PBE.py', deps=[ferro, anti, non])
    queue.add('eels/test.py', deps=band)
    queue.add('gw/test.py')
    rpa_si_exxgs = queue.add('rpa/si.pbe.py')
    queue.add('rpa/si.pbe+exx.py', deps=rpa_si_exxgs, ncpus=4)
    rpa_si_rpags = queue.add('rpa/si.rpa_init_pbe.py')
    queue.add('rpa/si.rpa.py', deps=rpa_si_rpags, ncpus=4)
    queue.add('stress/con_pw.py', ncpus=1)
    queue.add('stress/stress.py', ncpus=1)
    queue.add('transport/pt_h2_tb_transport.py')
    t1 = queue.add('transport/pt_h2_lcao_manual.py')
    queue.add('transport/pt_h2_lcao_transport.py', deps=t1)
