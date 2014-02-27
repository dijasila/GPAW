def agts(queue):
    gaps = queue.add('gaps.py', ncpus=16, walltime=5 * 60)
    queue.add('submit.agts.py', deps=gaps)
 

if __name__ == '__main__':
    import ase.db
    c = ase.db.connect('gaps.db')
    for d in c.select():
        print(d.name)
        for k, e in d.data.items():
            print(k, e.tolist())
