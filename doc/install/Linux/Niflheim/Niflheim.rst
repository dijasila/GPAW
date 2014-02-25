.. _Niflheim:

========
Niflheim
========

Information about the Niflheim cluster can be found at
`<https://wiki.fysik.dtu.dk/niflheim>`_.

Preferably use the system default installation of GPAW and setups:
to be able to do so, please do **not**
overwrite the system default :envvar:`PATH`, :envvar:`PYTHONPATH`,
nor :envvar:`GPAW_SETUP_PATH` environment variables.
When setting the environment variables **prepend** them, i.e.:

- using csh/tcsh::

    setenv PATH ${HOME}/bin:${PATH}

- using bash::

    export PATH=${HOME}/bin:${PATH}

If you decide to install a development version of GPAW, this is what you do
when installating GPAW for the first time:

1. On the ``servcamd`` filesystem (login on your workstation)
   go to a directory on the Niflheim filesystem.
   Usually users install GPAW under Niflheim's :envvar:`HOME`,
   i.e. :file:`/home/niflheim/$USER/gpaw`,
   and the instructions below assume this.

2. Checkout the GPAW source. Make **sure** that
   :file:`/home/niflheim/$USER/gpaw` does **not** exist,
   before running the checkout!::

     svn checkout https://svn.fysik.dtu.dk/projects/gpaw/trunk /home/niflheim/$USER/gpaw

   **Note**: if you are doing a heavy development (many svn checkins)
   you may consider installing a special development version on workstation's
   local disk (faster), i.e. :file:`/scratch/$USER/gpaw`,
   however this version will not be accesible from Niflheim.

3. Set the :envvar:`GPAW_HOME` environment variable:

- using csh/tcsh::

    setenv GPAW_HOME /home/niflheim/$USER/gpaw

- using bash::

    export GPAW_HOME=/home/niflheim/$USER/gpaw

4. To compile the code, run the shell script
   :svn:`~doc/install/Linux/Niflheim/compile.sh`:

   .. literalinclude:: compile.sh

   preferably from your gpaw directory, i.e.::

     cd ${GPAW_HOME}
     sh ./doc/install/Linux/Niflheim/compile.sh

   Note the  versions of gpaw compiled with `TAU Performance System <http://www.cs.uoregon.edu/research/tau/>`_.
   See `Modules in batch jobs <https://wiki.fysik.dtu.dk/niflheim/Installed_software#modules-in-batch-jobs>`_ for details of using these on Niflheim.

   If you have login passwords active,
   this will force you to type your password four times. It is
   possible to remove the need for typing passwords on internal CAMd systems,
   using the procedure described at
   https://wiki.fysik.dtu.dk/it/SshWithoutPassword.

5. **Prepend** :envvar:`PYTHONPATH` and :envvar:`PATH` environment variables:

- csh/tcsh - add to /home/niflheim/$USER/.cshrc::

    if ( "`echo $FYS_PLATFORM`" == "AMD-Opteron-el5" ) then # fjorm
        module load GPAW
        setenv GPAW_PLATFORM "linux-x86_64-opteron-2.4"
    endif
    if ( "`echo $FYS_PLATFORM`" == "Intel-Nehalem-el5" ) then # thul
        module load GPAW
        setenv GPAW_PLATFORM "linux-x86_64-xeon-2.4"
    endif
    if ( "`echo $FYS_PLATFORM`" == "x3455-el6" ) then # slid
        module load GPAW
        setenv GPAW_PLATFORM "linux-x86_64-x3455-2.6"
    endif
    if ( "`echo $FYS_PLATFORM`" == "dl160g6-el6" ) then # muspel
        module load GPAW
        setenv GPAW_PLATFORM "linux-x86_64-dl160g6-2.6"
    endif
    if ( "`echo $FYS_PLATFORM`" == "sl230s-el6" ) then # surt
        module load GPAW
        setenv GPAW_PLATFORM "linux-x86_64-sl230s-2.6"
    endif
    # GPAW_HOME must be set after loading the GPAW module!
    setenv GPAW_HOME /home/niflheim/$USER/gpaw
    setenv PATH ${GPAW_HOME}/build/bin.${GPAW_PLATFORM}:${PATH}
    setenv PATH ${GPAW_HOME}/tools:${PATH}
    setenv PYTHONPATH ${GPAW_HOME}:${PYTHONPATH}
    setenv PYTHONPATH ${GPAW_HOME}/build/lib.${GPAW_PLATFORM}:${PYTHONPATH}

- bash - add to /home/niflheim/$USER/.bashrc::

    if [ "`echo $FYS_PLATFORM`" == "AMD-Opteron-el5" ]; then # fjorm
        module load GPAW
        export GPAW_PLATFORM="linux-x86_64-opteron-2.4"
    fi
    if [ "`echo $FYS_PLATFORM`" == "Intel-Nehalem-el5" ]; then # thul
        module load GPAW
        export GPAW_PLATFORM="linux-x86_64-xeon-2.4"
    fi
    if [ "`echo $FYS_PLATFORM`" == "x3455-el6" ]; then # slid
        module load GPAW
        export GPAW_PLATFORM="linux-x86_64-x3455-2.6"
    fi
    if [ "`echo $FYS_PLATFORM`" == "dl160g6-el6" ]; then # muspel
        module load GPAW
        export GPAW_PLATFORM="linux-x86_64-dl160g6-2.6"
    fi
    if [ "`echo $FYS_PLATFORM`" == "sl230s-el6" ]; then # surt
        module load GPAW
        export GPAW_PLATFORM="linux-x86_64-sl230s-2.6"
    fi
    # GPAW_HOME must be set after loading the GPAW module!
    export GPAW_HOME=/home/niflheim/$USER/gpaw
    export PATH=${GPAW_HOME}/build/bin.${GPAW_PLATFORM}:${PATH}
    export PATH=${GPAW_HOME}/tools:${PATH}
    export PYTHONPATH=${GPAW_HOME}:${PYTHONPATH}
    export PYTHONPATH=${GPAW_HOME}/build/lib.${GPAW_PLATFORM}:${PYTHONPATH}

  Make sure that you add these settings above any line that
  causes exit when run in the batch system e.g. ``if ( { tty -s } == 0 ) exit``.
 
  **Warning**: from the time you save settings in /home/niflheim/$USER/.cshrc
  of /home/niflheim/$USER/.bashrc, your jobs (also those waiting
  currently in the queue) will start using the new version.
  Consider making such changes with no jobs in the queue.

6. If you prefer to use a personal setup's directory follow
   :ref:`installationguide_setup_files`.

7. When submitting jobs to the batch system, use the file
   :svn:`~doc/documentation/parallel_runs/gpaw-qsub` instead of the
   usual :command:`qsub`.

When updating the gpaw code in the future:

- Go to the :envvar:`GPAW_HOME` directory and run::

    svn up

- If any of the c-code changed during the update repeat step 4.

.. note::

  Please ask the Niflheim's support staff to verify that gpaw-python runs single-threaded, e.g. for a job running on ``p024`` do from ``audhumbla``::

    ssh p024 ps -fL

  Numbers higher then **1** in the **NLWP** column mean multi-threaded job.

  In case of openmpi it is necessary to set the :envvar:`OMP_NUM_THREADS` variable::

    setenv OMP_NUM_THREADS 1 # [t]csh
    export OMP_NUM_THREADS=1 # [ba]sh
