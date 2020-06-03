def create_tasks():
    from myqueue.task import task
    return [task('gaps.py@16:5h'),
            task('submit.agts.py', deps='gaps.py'),
            task('molecules.py', cores=8, tmax='5h')]


def test_gaps():
    import ase.db
    c = ase.db.connect('gaps.db')
    for d in sorted(c.select(), key=lambda d: d.name):
        print((d.name))
        for k, e in d.data.items():
            epbe = max(abs(e[0] - e[2]), abs(e[0] - e[4]))
            epbe0 = max(abs(e[1] - e[3]), abs(e[1] - e[5]))
            r = (e[:2].tolist() +
                 (e[:2] - e[2:4]).tolist() +
                 (e[:2] - e[4:]).tolist() +
                 [epbe, epbe0])
            print(k, ' '.join(['%6.3f' % x for x in r]))
            assert epbe < 0.26
            assert epbe0 < 0.26


if __name__ == '__main__':
    test_gaps()
