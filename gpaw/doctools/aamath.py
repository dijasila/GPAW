from __future__ import annotations


def prep(lines: list[str]) -> list[str]:
    """Preprocess lines.

    * Remove leading and trailing empty lines.
    * Make all lines have the same length (pad with spaces).
    * Remove spaces from beginning of lines.
    """
    lines = [line for line in lines if line.strip()]
    if not lines:
        return []
    n = min(len(line) - len(line.lstrip()) for line in lines)
    lines = [line[n:] for line in lines]
    n = max(len(lines) for line in lines)
    return [line.ljust(n) for line in lines]


class ParseError(Exception):
    """Bad ascii-art math."""


def cut(lines: list[str], i1: int, i2: int = None) -> list[str]:
    """Cut out block.

    >>> cut(['012345', 'abcdef'], 1, 3)
    ['12', 'bc']
    """
    index = slice(i1, i2)
    return [line[index] for line in lines]


def block(lines: list[str]) -> dict[int, str]:
    r"""Find superscript/subscript blocks.

    >>> block([' 2  _ ',
    ...        '    k '])
    {0: '2', 3: '\\mathbf{k}'}
    """
    if not lines:
        return {}
    blocks = {}
    i1 = None
    for i in range(len(lines[0])):
        if all(line[i] == ' ' for line in lines):
            if i1 is not None:
                blocks[i1 - 1] = parse(cut(lines, i1, i))
                i1 = None
        else:
            if i1 is None:
                i1 = i
    if i1 is not None:
        blocks[i1 - 1] = parse(cut(lines, i1))
    return blocks


def parse(lines: str | list[str], n: int = None) -> str:
    r"""Parse ascii-art math to LaTeX.

    >>> parse([' /   ~      ',
    ...        ' |dx p  (x) ',
    ...        ' /    ai    '])
    '\\int dx \\tilde{p}_{ai}  (x)'
    >>> parse(['   _ _ ',
    ...        '  ik.r ',
    ...        ' e     '])
    'e^{i\\mathbf{k}.\\mathbf{r}}'
    """
    if isinstance(lines, str):
        lines = lines.splitlines()
    lines = prep(lines)
    if not lines:
        return ''
    if n is None:
        N = max((len(line.replace(' ', '')), n)
                for n, line in enumerate(lines))[1]
        for n in [N] + [n for n in range(len(lines)) if n != N]:
            try:
                latex = parse(lines, n)
            except ParseError:
                continue
            return latex
        raise ParseError

    line = lines[n]
    i1 = line.find('--')
    if i1 != -1:
        i2 = len(line) - i1 - len(line[i1:].lstrip('-'))
        p1 = parse(cut(lines, 0, i1))
        p2 = parse(cut(lines[:n], i1, i2))
        p3 = parse(cut(lines[n + 1:], i1, i2))
        p4 = parse(cut(lines, i2 + 1))
        return rf'{p1} \frac{{{p2}}}{{{p3}}} {p4}'.strip()

    i = line.find('|')
    if i != -1:
        if n > 0 and lines[n - 1][i] == '/':
            p1 = parse(cut(lines, 0, i))
            p2 = parse(cut(lines, i + 1))
            return rf'{p1} \int {p2}'.strip()
        i1 = line.find('<')
        i2 = line.find('>')
        if i1 == -1 or i1 > i or i2 == -1 or i2 < i:
            raise ParseError
        p1 = parse(cut(lines, 0, i1))
        p2 = parse(cut(lines, i1 + 1, i))
        p3 = parse(cut(lines, i + 1, i2))
        p4 = parse(cut(lines, i2 + 1))
        return rf'{p1} \langle {p2}|{p3} \rangle {p4}'.strip()

    hats = {}
    if n > 0:
        new = []
        for i, c in enumerate(lines[n - 1]):
            if c in '^~_':
                hats[i] = c
                c = ' '
            new.append(c)
        lines[n - 1] = ''.join(new)

    superscripts = block(lines[:n])
    subscripts = block(lines[n + 1:])

    latex = []
    for i, c in enumerate(line):
        if c.isalpha():
            if i in hats:
                hat = {'^': 'hat', '~': 'tilde', '_': 'mathbf'}[hats[i]]
                c = rf'\{hat}{{{c}}}'
            sup = superscripts.pop(i, None)
            if sup:
                c = rf'{c}^{{{sup}}}'
            sub = subscripts.pop(i, None)
            if sub:
                c = rf'{c}_{{{sub}}}'
        latex.append(c)

    if superscripts or subscripts:
        raise ParseError

    return ''.join(latex).strip()


def autodoc_process_docstring(lines):
    """Hook-function for Sphinx."""
    for i1, line in enumerate(lines):
        if line.endswith(':::'):
            for i2, line in enumerate(lines[i1 + 2:], i1 + 2):
                if not line:
                    break
            else:
                i2 += 1
            latex = parse(lines[i1 + 1:i2])
            lines[i1:i2] = [f'.. math:: {latex}']
            return


examples = r"""
1+2
---
 3
\frac{1+2}{3}

<a|b>
\langle a|b \rangle
"""


def test_examples():
    for example in examples[1:].split('\n\n'):
        lines = example.splitlines()
        print(lines)
        assert parse(lines[:-1]) == lines[-1]


def main():
    import sys
    lines = sys.stdin.read().splitlines()
    print(parse(lines))


if __name__ == '__main__':
    main()
