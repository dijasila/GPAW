import sys
sys.path.append('.')

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

extensions = [#'ytp',
              'ext', 'images', 'sitelink',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.mathjax',
              'sphinx.ext.intersphinx']
#templates_path = ['templates']
source_suffix = '.rst'
master_doc = 'contents'
project = 'GPAW'
copyright = 'CAMd et al.'
exclude_patterns = ['build']
default_role = 'math'
pygments_style = 'sphinx'
autoclass_content = 'both'
intersphinx_mapping = {
    'ase': ('http://wiki.fysik.dtu.dk/ase', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'mayavi': ('http://docs.enthought.com/mayavi/mayavi', None)}

html_style = 'css/gpaw.css'
html_title = 'GPAW'
html_logo = 'static/logo-gpaw.png'
html_favicon = 'static/gpaw_favicon.ico'
html_static_path = ['static']
html_last_updated_fmt = '%a, %d %b %Y %H:%M:%S'
