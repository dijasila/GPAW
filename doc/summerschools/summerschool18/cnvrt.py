import json
from pathlib import Path


header_comment = '# %%\n'


def nb2py(notebook):
    result = []
    cells = notebook['cells']

    for cell in cells:
        cell_type = cell['cell_type']

        if cell_type == 'markdown':
            result.append('%s"""\n%s\n"""'%
                          (header_comment, ''.join(cell['source'])))

        if cell_type == 'code':
            cell['source'] = ['# magic: ' + line
                              if line.startswith(('!', '%'))
                              else line
                              for line in cell['source']]
            result.append("%s%s" % (header_comment, ''.join(cell['source'])))

    return '\n\n'.join(result)


for path in Path().glob('*/*.ipynb'):
    with open(path, 'r') as f:
        notebook = json.load(f)
        py_str = nb2py(notebook)
        out_file = str(path).replace('.ipynb', '.py').replace('.master', '')
        with open(out_file, 'w') as f:
            f.write(py_str)
