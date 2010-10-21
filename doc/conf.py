# -*- coding: utf-8 -*-
#
# GPAW documentation build configuration file, created by
# sphinx-quickstart on Fri Jun 20 09:39:26 2008.
#
# This file is execfile()d with the current directory set to its containing dir.
#
# The contents of this file are pickled, so don't put values in the namespace
# that aren't pickleable (module imports are okay, they're removed automatically).
#
# All configuration values have a default value; values that are commented out
# serve to show the default value.

import sys, os

# If your extensions are in another directory, add it here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
#sys.path.append(os.path.abspath('some/directory'))
sys.path.append('.')

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
try:
    from sphinx.ext import pngmath
    ext_png_math = 'sphinx.ext.pngmath'
except ImportError:
    ext_png_math = 'mathpng'
    print 'Warning: sphinx uses custom mathpng.py: please update to sphinx >= 5.0'


# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['ext', 'images', 'sitelink',
              'sphinx.ext.autodoc',
              ext_png_math]

try:
    from sphinx.ext import intersphinx
    extensions.append('sphinx.ext.intersphinx')
except ImportError:
    print 'Warning: no sphinx.ext.intersphinx available: please update to sphinx >= 5.0'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'contents'

# General substitutions.
project = 'GPAW'
copyright = 'CAMd et al.'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
from gpaw.version import version_base as version

# The full version, including alpha/beta/rc tags.
release = version

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

# The reST default role (used for this markup: `text`) to use for all documents.
#default_role = None

# If true, '()' will be appended to :func: etc. cross-reference text.
#add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# Options for HTML output
# -----------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'gpaw.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = 'GPAW'

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = '_static/logo-gpaw.png'

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = '_static/gpaw_favicon.ico'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = ''#'%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
#html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
#html_additional_pages = {}

# If false, no module index is generated.
#html_use_modindex = True

# If false, no index is generated.
#html_use_index = True

# If true, the index is split into individual pages for each letter.
#html_split_index = False

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
html_file_suffix = '.html'

# Output file base name for HTML help builder.
htmlhelp_basename = 'GPAWdoc'


# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = 'a4'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
  ('contents', 'GPAW.tex', 'GPAW Documentation', 'CAMd et al.', 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
latex_preamble = '\usepackage{amsmath}\usepackage{amsfonts}'

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_use_modindex = True

# Example configuration for intersphinx: refer to ase.
intersphinx_mapping = {'http://wiki.fysik.dtu.dk/ase': None}

# sphinx.ext.pngmath manual configuration
# ---------------------------------------

pngmath_latex_preamble = r"""
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage[active]{preview}
\newcommand{\br}{\mathbf{r}}
"""

# Additional arguments to give to dvipng, as a list.
# The default value is ['-gamma 1.5', '-D 110']
pngmath_dvipng_args = [
    '-bgTransparent',
    '-Ttight',
    '--noghostscript',
    '-l10',
    '--depth',
    '-D 130',
    ]

# correctly aligns the baselines
pngmath_use_preview = True

autoclass_content = 'both'
