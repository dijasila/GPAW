from q2.job import Job


def workflow():
    jobs = [Job('g2_dzp.py+{}@4x13m'.format(i)) for i in range(10)]
    return jobs + [Job('g2_dzp_csv.py', deps=jobs)]
