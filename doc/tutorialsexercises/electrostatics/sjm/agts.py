from myqueue.workflow import run


def workflow():
    Au111 = run(script='run-Au111.py', cores=24, tmax='1h')
    run(script='plot-traces.py', cores=1, tmax='1h', deps=[Au111])
    H_seq = run(script='run-Au111-H-seq.py', cores=24, tmax='5h', deps=[Au111])
    H_sim = run(script='run-Au111-H-sim.py', cores=24, tmax='5h', deps=[Au111])
    run(script='plot-seq-sim.py', cores=1, tmax='1h', deps=[H_seq, H_sim])
    H_hollow = run(script='run-Au111-H-hollow.py', cores=24, tmax='5h',
                   deps=[H_sim])
    neb = run(script='run-neb.py', cores=24, tmax='24h',
              deps=[H_sim, H_hollow])
    run(script='plot-neb.py', cores=1, tmax='1h', deps=[neb])
