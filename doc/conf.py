import sys
import sphinx_rtd_theme
from gpaw import __version__

assert sys.version_info >= (3, 6)

sys.path.append('.')

extensions = ['images',
              'ext',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx']
templates_path = ['templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'GPAW'
copyright = '2020, GPAW developers'
exclude_patterns = ['build']
default_role = 'math'
pygments_style = 'sphinx'
autoclass_content = 'both'
modindex_common_prefix = ['gpaw.']
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.8', None),
    'ase': ('https://wiki.fysik.dtu.dk/ase', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'pytest': ('https://docs.pytest.org/en/stable', None),
    'mayavi': ('http://docs.enthought.com/mayavi/mayavi', None)}

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_style = 'gpaw.css'
html_title = 'GPAW'
html_favicon = 'static/gpaw_favicon.ico'
html_static_path = ['static']
html_last_updated_fmt = '%a, %d %b %Y %H:%M:%S'
dev_version = '20.1.1b1'  # This line auto-edited by newrelease script
stable_version = '20.1.0'  # This line auto-edited by newrelease script
html_context = {
    'current_version': __version__,
    'versions':
        [('{} (development)'.format(dev_version),
          'https://wiki.fysik.dtu.dk/gpaw/dev'),
         ('{} (latest stable)'.format(stable_version),
          'https://wiki.fysik.dtu.dk/gpaw')]}
mathjax_config = {
    'TeX': {
        'Macros': {
            'br': '{\\mathbf r}',
            'bk': '{\\mathbf k}',
            'bG': '{\\mathbf G}'}}}
