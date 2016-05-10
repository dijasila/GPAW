def agts(queue):
    # Only run this when datasets change:
    if 0:
        queue.add('make_setup_pages_data.py', creates='datasets.json')
    else:
        # We need this in order for Sphinx to download the json file:
        queue.add('datasets.agts.py', creates='datasets.json')
