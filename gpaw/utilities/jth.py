"""Create softlinks to JTH datasets."""
import re
import sys
from pathlib import Path

for path in Path(sys.argv[1]).glob('*.*-JTH.xml'):
    m = re.match(r'([A-Z][a-z]?)\.[A-Z]+_([A-Z]+)-JTH\.xml', path.name)
    symbol = m[1]
    xc = m[2]
    Path(f'{symbol}.jth.{xc}').symlink_to(path)
