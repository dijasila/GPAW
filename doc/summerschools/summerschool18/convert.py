# creates: catalysis/n2-on-metal.ipynb
# ... and other .ipynb files

import json
from pathlib import Path


def f(path):
    data = json.loads(path.read_text())

    lines = [f'# Converted from {path}\n']
    n = 1
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            lines.extend(['\n', f'# In [{n}]:\n'])
            for line in cell['source']:
                if line.startswith('%') or line.startswith('!'):
                    line = '# ' + line
                lines.append(line)
            lines.append('\n')
            n += 1

    path.with_suffix('.py').write_text(''.join(lines))


def convert(path):
    assert path.name.endswith('.master.ipynb')
    data = json.loads(path.read_text())
    cells = []
    for cell in data['cells']:
        if cell['cell_type'] == 'code':
            lines = cell['source']
            if (lines and
                lines[0].replace(' ', '').lower().startswith('#teacher')):
                continue
        cells.append(cell)
    data['cells'] = cells
    new = path.with_name(path.name.rsplit('.', 2)[0] + '.ipynb')
    new.write_text(json.dumps(data, indent=1))


if __name__ == '__main__':
    for path in Path().glob('*/*.master.ipynb'):
        print(path)
        convert(path)
