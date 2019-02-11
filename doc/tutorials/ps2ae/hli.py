# creates: hli-wfs.png, hli-pot.png
import sys
sys.path.insert(0, '.')
try:
    import hli_wfs  # noqa
    import hli_pot  # noqa
except ImportError:
    from pathlib import Path
    print(Path.cwd(), sys.path)
    raise
