.. _homebrew:

========
Homebrew
========

Mountain Lion
=============

Install https://developer.apple.com/xcode/ and activate it from a terminal::

  xcodebuild -license

After installing xcode install also its `Command-Line Tools` (provides
`llvm-gcc compiler` on the command line).
After launching Xcode, in the top menubar, close to the `Apple`, choose
Xcode -> Preferences -> Downloads).

Follow the instructions for installing Homebrew http://mxcl.github.com/homebrew/
the famous::

  ruby <(curl -fsSkL https://raw.github.com/mxcl/homebrew/go)

and configure your init scripts `~/.bash_profile`::

  # Set architecture flags
  export ARCHFLAGS="-arch x86_64"
  # Ensure user-installed binaries take precedence
  export PATH=/usr/local/share/python:/usr/local/bin:$PATH

  # virtualenv should use Distribute instead of legacy setuptools
  export VIRTUALENV_DISTRIBUTE=true
  # Centralized location for new virtual environments
  export PIP_VIRTUALENV_BASE=$HOME/Virtualenvs
  # pip should only run if there is a virtualenv currently activated
  export PIP_REQUIRE_VIRTUALENV=true
  # cache pip-installed packages to avoid re-downloading
  export PIP_DOWNLOAD_CACHE=$HOME/.pip/cache

Verify your homebrew::

  brew doctor

Update with::

  brew update

Build homebrew python::

  brew install python --with-brewed-openssl --framework

And install the following homebrew packages::

  brew install gfortran
  brew install openmpi

Install GPAW setups::

  brew install gpaw-setups

Currently with::

  brew install https://github.com/marcindulak/homebrew/raw/gpaw-setups/Library/Formula/gpaw-setups.rb

www.virtualenv.org allows you to run different versions of python modules after
having them configured in different virtualenvs.
It is a convenient way of keeping GPAW with its corresponding
ASE version isolated form the globally installed python modules.

Numpy installed under virtualenv does not work with gpaw-python
(`ImportError: numpy.core.multiarray failed to import`), so install numpy
globally with::

  PIP_REQUIRE_VIRTUALENV=false pip install numpy

Install virtualenv::

  PIP_REQUIRE_VIRTUALENV=false pip install virtualenv && mkdir ~/Virtualenvs

Configure a virtualenv for GPAW, e.g. trunk::

  cd ~/Virtualenvs
  virtualenv gpaw-trunk && cd gpaw-trunk
  . bin/activate

Now, install ASE inside of virtualenv::

  pip install python-ase

Install GPAW (still inside of the virtualenv) for example with
:ref:`installationguide_standard`::

  python setup.py install

You may need additionally to set :envvar:`PYTHONPATH` (should not be necessary).

If you prefer to have matplotlib available you need to
install http://xquartz.macosforge.org, reboot and additionally::

  brew install freetype
  brew install libpng
  brew install pygtk

Set the PKG_CONFIG_PATH correctly https://github.com/mxcl/homebrew/issues/16891
and then, again inside of virtualenv::

  pip install matplotlib
