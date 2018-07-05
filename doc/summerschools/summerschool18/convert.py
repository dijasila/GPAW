# creates: catalysis/n2_on_metal.ipynb
# ... and other .ipynb files

import json
from pathlib import Path


def convert(path):
    assert path.name.endswith('.master.ipynb')
    data = json.loads(path.read_text())
    cells = []
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            if lines and lines[0].startswith('# teacher'):
                continue
            for i, line in enumerate(lines):
                if ' # student: ' in line:
                    a, b = (x.strip() for x in line.split('# student:'))
                    lines[i] = line.split(a)[0] + b + '\n'
        cells.append(cell)
    data['cells'] = cells
    new = path.with_name(path.name.rsplit('.', 2)[0] + '.ipynb')
    new.write_text(json.dumps(data, indent=1))


for path in Path().glob('*/*.master.ipynb'):
    print(path)
    convert(path)
