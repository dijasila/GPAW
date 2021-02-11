.. _load on niflheim:

======================================
Using a pre-installed GPAW at Niflheim
======================================

This is the guide for using the pre-installed GPAW modules on Niflheim.

Modules on Niflheim
===================

You can see which modules are available with the ``module avail [package]`` command, for example::


  $ module avail GPAW

  -------------------------- /home/modules/modules/all --------------------------
     GPAW-setups/0.8.7929
     GPAW-setups/0.9.9672
     GPAW-setups/0.9.11271
     GPAW-setups/0.9.20000                            (D)
     GPAW/1.4.0-foss-2018a-Python-3.6.4
     GPAW/1.4.0-foss-2018b-Python-3.6.6
     GPAW/1.4.0-intel-2018b-Python-3.6.6
     GPAW/1.5.1-foss-2018b-Python-3.6.6
     GPAW/1.5.1-intel-2018b-Python-3.6.6
     GPAW/1.5.2-foss-2018b-Python-3.6.6
     GPAW/1.5.2-intel-2018b-Python-3.6.6
     GPAW/19.8.1-foss-2018b-ASE-3.18.0-Python-3.6.6
     GPAW/19.8.1-intel-2018b-ASE-3.18.0-Python-3.6.6
     GPAW/20.1.0-foss-2019b-Python-3.7.4
     GPAW/20.1.0-intel-2019b-Python-3.7.4
     GPAW/20.10.0-foss-2019b-ASE-3.20.1-Python-3.7.4
     GPAW/20.10.0-foss-2020b
     GPAW/20.10.0-intel-2019b-ASE-3.20.1-Python-3.7.4
     GPAW/20.10.0-intel-2020b 
     GPAW/21.1.0-foss-2020b-ASE-3.21.1
     GPAW/21.1.0-intel-2020b-ASE-3.21.1               (D)
    Where:
     D:  Default Module

You can see which modules you have loaded with ``module list``.  You
can unload all modules to start from scratch with ``module purge``.


Choose the right version of GPAW
================================

This is a brief guide to which version of GPAW you should use. It
reflects the situation in December 2020 and will be updated as
the situation changes.


I have an ongoing project
  You should probably continue to use the version you are using in
  that project, unless you want to change.  See the section below on
  using different versions for different project.

I am a normal user
  You should load ``GPAW/21.1.0-intel-2020b``.

  This will give the newest version of GPAW, as recommended by the
  developers.  It has new features and is significantly faster, in
  particular on the new Xeon40 nodes.  For ongoing projects that have
  been using an older version, you may find that some values have
  changed slightly - check for consistency, or be sure to always use
  the same version for ongoing projects.  See below for a description
  on how to do that.

I am sligtly conservative or need ``libvwdxc``.
  The version of GPAW compiled with the FOSS toolchain is somewhat
  slower in many situations, but is better tested and may use less
  memory.  You may also have to use this version if you want the
  functionality from ``libvwdxc`` library, but be aware that many vad
  der Waals potentials do not use ``libvwdxc``.
  

**IMPORTANT:**  You do *not* need to load Python, ASE, matplotlib etc.
Loading GPAW pulls all that stuff in, in versions consistent with the
chosen GPAW version.

If you want to generate Wannier functions with the Wannier90 module,
you need to explicitly load ``Wannier90/3.1.0-foss-2020b`` or
``Wannier90/3.1.0-intel-2020b``.


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

     
==============   ==================================
foss/2020b       foss-2020b

                 gompi-2020b
		 
                 GCCcore-10.2.0
--------------   ----------------------------------
intel/2020b      intel-2020b

                 iccifort-2020.4.304
		 
                 iimpi-2020b
		 
                 GCCcore-10.2.0
--------------   ----------------------------------
foss/2019b       foss-2019b

                 foss-2019b-[ASE-3.20.1]Python-3.7.4
		 
		 gompi-2019b
		 
		 GCC-8.3.0
--------------   ----------------------------------
intel/2019b      intel-2019b

                 intel-2019b-[ASE-3.20.1]Python-3.7.4
		 
		 iccifort-2019.5.281
		 
		 iimpi-2019b
		 
		 GCCcore-8.3.0
--------------   ----------------------------------
foss/2018b       foss-2018b

                 foss-2018b-Python-3.6.6
		 
                 gompi-2018b
		 
                 GCCcore-7.3.0
--------------   ----------------------------------
intel/2018b      intel-2018b

                 intel-2018b-Python-3.6.6
		 
                 iimpi-2018b
		 
		 iccifort-2018.3.222-GCC-7.3.0-2.30
		 
                 GCCcore-7.3.0
		 
		 *and a few variations thereof*
==============   ==================================

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
      module load GPAW/20.10.0-intel-2020b
      module load scikit-learn/0.23.2-intel-2020b
  fi

The ``module purge`` command in the special branch is because SLURM
will remember which modules you have loaded when you submit the job,
and that will typically be the default version, which must then be
unloaded.
