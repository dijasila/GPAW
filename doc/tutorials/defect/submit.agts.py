from myqueue.task import task


def create_tasks():
    return [
        task('calc_gs.py@20:00m'), # 6 cores
        task('vacancy_run@3:00h', deps='calc_gs.py'), # 6 cores
        task('vacancy_supercell.py@10:00m', deps='vacancy_run.py'), # 5 cores
        task('defect_types.py@2:00m', deps='calc_gs.py'), # 1 core
        task('vacancy_V_vs_k.py@20:00m', deps='vacancy_supercell.py'),
        ]
