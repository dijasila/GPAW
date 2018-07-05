import json
from pathlib import Path


def convert(path):
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

    code = ''.join(lines)

    path.with_suffix('.py').write_text(code)

    return code


def run(name):
    code = convert(Path(name + '.master.ipynb'))
    exec(code)


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
