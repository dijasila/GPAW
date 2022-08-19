"""Check URL's in Python files."""
import re
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen, Request

OK = {'https://doi.org/%s',
      'https://arxiv.org/abs/%s',
      'https://xkcd.com/%s'}
USERAGENT = 'Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11'


def check(root: Path) -> None:
    """Chech URL's in Python files inside root."""
    for path in root.glob('**/*.py'):
        for n, line in enumerate(path.read_text().splitlines()):
            for url in re.findall(r'https?://\S+', line):
                url = url.rstrip(",.')")
                if url not in OK and 'html/_downloads' not in str(path):
                    check1(path, n, url)


def check1(path: Path, n: int, url: str) -> None:
    try:
        req = Request(url, headers={'User-Agent': USERAGENT})
        urlopen(req)
    except (HTTPError, ConnectionResetError) as e:
        print(f'{path}:{n + 1}')
        print(url)
        print(e)
        print()


if __name__ == '__main__':
    root = Path(sys.argv[1])
    check(root)
