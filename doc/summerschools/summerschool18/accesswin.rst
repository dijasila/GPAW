.. _accesswin:

===================================================
Starting and accessing a Jupyter Notebook (Windows)
===================================================

To run a Jyputer Notebook in the DTU databar while displaying it output in your browser requires three steps.

* Starting the notebook on an interactive compute node.

* Make a connection to the relevant compute node, bypassing the firewall.

* Connecting your browser to the Jupyter Notebook process.


Logging into the databar
========================

If you are not already logged into the databar, do so by starting MobaXterm.  There should be a session available from the welcome screen of MobaXterm named ``login.gbar.dtu.dk`` or similar, created when you logged in the first time.  Click on it to log in again.

Once you are logged in on the front-end, get a session on an interactive compute node by typing the command::

  linuxsh -X

  
Starting a Jupyter Notebook
===========================

Change to the folder where you keep your notebooks (most likely ``CAMD2018``) and start the Jupyter Notebook server::

  cd CAMD2018
  camdnotebook

The command ``camdnotebook`` is a local script.  It checks that you are on a compute server (and not on the front-end), that X11 forwarding is enabled, and then it starts a jupyter notebook by running the command ``jupyter notebook --no-browser --port=40000 --ip=$HOSTNAME``  (you can also use this command yourself if you prefer).

The Notebook server replies by printing a few status lines, as seen here


