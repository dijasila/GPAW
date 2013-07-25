def agts(queue):
    al = queue.add('al100.py')
    queue.add('stm.py', deps=al, creates=['2d.png', 'line.png'])
