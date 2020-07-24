"""Code for summerschool notebooks."""
import json
import sys
from pathlib import Path
from runpy import run_path


def main(path: Path) -> None:
    """Run code with ase.visualize.view() disabled."""
    import ase.visualize as visualize

    def view(atoms, repeat=None):
        pass

    visualize.view = view
    run_path(path)


def py2ipynb(path: Path) -> None:
    cells = []
    text = path.read_text()
    assert text.startswith('# %%\n')
    chunks = text[5:].split('\n\n# %%\n')

    for chunk in chunks:
        cell_type = 'code'
        if chunk.startswith(("'''", '"""')):
            chunk = chunk.strip('"\n')
            cell_type = 'markdown'

        cell = {
            'cell_type': cell_type,
            'metadata': {},
            'source': chunk.splitlines(True)}

        if cell_type == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            lines = cell['source']
            for i, line in enumerate(lines):
                if ' # student:' in line:
                    a, b = (x.strip() for x in line.split('# student:'))
                    lines[i] = line.split(a)[0] + b + '\n'
                elif line.startswith('# magic: '):
                    lines[i] = line[9:]
                elif line.lower().startswith('# teacher'):
                    del lines[i:]
                    break

        cells.append(cell)

    outpath = path.with_suffix('.ipynb')
    outpath.write_text(
        json.dumps(
            {'cells': cells,
             'metadata': {
                 'kernelspec': {
                     'display_name': 'Python 3',
                     'language': 'python',
                     'name': 'python3'},
                 'language_info': {
                     'codemirror_mode': {'name': 'ipython', 'version': 3},
                     'file_extension': '.py',
                     'mimetype': 'text/x-python',
                     'name': 'python',
                     'nbconvert_exporter': 'python',
                     'pygments_lexer': 'ipython3',
                     'version': '3.6.1'}},
             'nbformat': 4,
             'nbformat_minor': 1},
            indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
