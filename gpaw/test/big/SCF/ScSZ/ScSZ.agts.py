def agts(queue):
    run = queue.add('ScSZ.py',
                    queueopts='-l nodes=1:ppn=8:xeon5570',
                    walltime=13*60, deps=[])
