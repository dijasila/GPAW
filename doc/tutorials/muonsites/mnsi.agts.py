def agts(queue):
    mnsi = queue.add('mnsi.py')
    queue.add('plot2d.py', deps=mnsi, creates='pot_contour.png')
