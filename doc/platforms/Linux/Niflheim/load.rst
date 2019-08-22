.. _load on niflheim:

======================================
Using a pre-installed GPAW at Niflheim
======================================

This is the guide for using the pre-installed GPAW modules on Niflheim.

Modules on Niflheim
===================

You can see which modules are available with the ``module avail [package]`` command, for example::

  $ module avail GPAW

  --------------------------- /home/modules/modules/all ---------------------------
     GPAW-setups/0.8.7929                         GPAW/1.5.1-foss-2018b-Python-3.6.6
     GPAW-setups/0.9.9672                         GPAW/1.5.1-intel-2018b-Python-3.6.6
     GPAW-setups/0.9.11271                        GPAW/1.5.2-foss-2018b-Python-3.6.6
     GPAW-setups/0.9.20000               (D)      GPAW/1.5.2-intel-2018b-Python-3.6.6
     GPAW/1.4.0-foss-2018a-Python-3.6.4  (D)      GPAW/19.8.1-foss-2018b-ASE-3.18.0-Python-3.6.6
     GPAW/1.4.0-foss-2018b-Python-3.6.6           GPAW/19.8.1-intel-2018b-ASE-3.18.0-Python-3.6.6
     GPAW/1.4.0-intel-2018b-Python-3.6.6

    Where:
     D:  Default Module

You can see which modules you have loaded with ``module list``.  You
can unload all modules to start from scratch with ``module purge``.


Choose the right version of GPAW
================================

This is a brief guide to which version of GPAW you should use. It
reflects the situation in December 2018 and will soon be updated as
the situation changes.

I am very conservative
  If you are currenlty using version 1.4.0 you can continue to do so
  by loading ``GPAW/1.4.0-foss-2018a-Python-3.6.4`` or
  ``GPAW/1.4.0-foss-2018b-Python-3.6.6``.

  The first of these is the exact build that has been used in 2018,
  loading it will break ``gedit``, ``emacs``, ``gnuplot`` and possibly
  some other graphical programs.  The second option is a newer
  (still 2018) build of the same version, it will not break anything.

I am conservative
  You should check what version you are using now
  (probably 1.4.0 or 1.5.2), and check that it is hardcoded in your
  ``.bashrc`` and/or your script files.  Be sure never to just load
  the default version.

I am a normal user
  You should load ``GPAW/19.8.1-intel-2018b-ASE-3.18.0-Python-3.6.6``

  This will give the newest version of GPAW, as recommended by the
  developers.  It has new features and is significantly faster, in
  particular on the new Xeon40 nodes.  For ongoing projects that have
  been using an older version, you may find that some values have
  changed slightly - check for consistency, or be sure to always use
  the same version for ongoing projects.  See below for a description
  on how to do that.

I am reckless
  You can just load the default version with ``module load GPAW``.

  You will have *no control* over when the default version change.  From
  midt September 2019 it will typically be the latest version.  *We do
  not recommend being reckless!*

**IMPORTANT:**  You do *not* need to load Python, ASE, matplotlib etc.
Loading GPAW pulls all that stuff in, in versions consistent with the
chosen GPAW version.

Intel or foss versions?
=======================

The versions built with the Intel compilers and the Intel Math Kernel
Library (MKL) are in average faster than the ones build with the Open
Source (GNU) compilers (FOSS = Free and Open Source Software).  On
newer hardware this difference can be very significant, and we
recommend using the Intel versions unless you have a good reason not
to.

The ``libvdwcx`` library of van der Waals exchange-correlation
potentials in incompatible with the MKL, so if you need these methods
you have to use the foss versions.  However, most van der Waals
calculations use the native van der Waals support in GPAW, and works
fine with the Intel versions.



Module consistency is important: check it.
==========================================

For a reliable computational experience, you need to make sure that
all modules come from the same toolchain (i.e. that the software is
compiled with a consistent set of tools).  **All modules you
load should belong to the same toolchain.**

Use ``module list`` to list your modules. Check for consistency:

* If you use the ``foss-2018a`` toolchain, all modules should end in
  ``foss-2018a``, ``foss-2018a-Python-3.6.4``, ``gompi-2018a`` or
  ``GCCcore-6.4.0``.

* If you use the ``foss-2018b`` toolchain, all modules should end in
  ``foss-2018b``, ``foss-2018b-Python-3.6.6``, ``gompi-2018b`` or
  ``GCCcore-7.3.0``.

* If you use the ``intel-2018b`` toolchain, all modules should end in
  ``intel-2018b``, ``intel-2018b-Python-3.6.6``, ``gompi-2018b`` or
  ``GCCcore-7.3.0``.

If your ``module load XXX`` commands give warnings about reloaded
modules, you are almost certainly mixing incompatible toolchains.


Using different versions for different projects.
================================================

You do not have to use the same modules for all your projects.  If you
want all jobs submitted from the folder ``~/ProjectAlpha`` to run with
an one version of GPAW, but everything else with a another version,
you can put this in your .bashrc::

  if [[ $SLURM_SUBMIT_DIR/ = $HOME/ProjectAlpha* ]]; then
      # Extreme consistency is important for this old project
      module purge
      module load GPAW/1.4.0-foss-2018a-Python-3.6.4
  else
      # Performance is important for everything else.
      module load GPAW/19.8.1-intel-2018b-ASE-3.18.0-Python-3.6.6
      module load scikit-learn/0.20.0-intel-2018b-Python-3.6.6.eb
  fi

The ``module purge`` command in the special branch is because SLURM
will remember which modules you have loaded when you submit the job,
and that will typically be the default version, which must then be
unloaded.
