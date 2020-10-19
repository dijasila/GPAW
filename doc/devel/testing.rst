.. _testing:

============
Testing GPAW
============

Testing of gpaw is done by a nightly test suite consisting of many
small and quick tests and by a weekly set of larger test.


"Quick" test suite
==================

.. warning::

    It's not really quick - it will take almost an hour to run all the tests!

Use pytest_ and pytest-xdist_ to run the tests::

    $ pytest -v -n <number-of-processes>

The test suite consists of a large number of small and quick tests
found in the :git:`gpaw/test/` directory.  The tests run nightly in serial
and in parallel.

In order to run the tests in parallel, do this:

    $ mpiexec -n <number-of-processes> pytest -v

Please report errors to the ``gpaw-users`` mailing list so that we
can fix them (see :ref:`mail list`).


.. _pytest: http://doc.pytest.org/en/latest/contents.html
.. _pytest-xdist: https://github.com/pytest-dev/pytest-xdist


.. highlight:: python


Special fixtures and marks
--------------------------

Tests that should only run in serial can be marked like this::

    import pytest

    @pytest.mark.serial
    def test_something():
        ...

There are two special GPAW-fixtures:

.. autofunction:: gpaw.test.conftest.in_tmp_dir
.. autofunction:: gpaw.test.conftest.gpw_files

Check the :git:`~gpaw/test/conftest.py` to see which gpw-files are available.
Use a ``_wfs`` postfix to get a gpw-file that contains the wave functions.


Adding new tests
----------------

A test script should fulfill a number of requirements:

* It should be quick.  Preferably not more than a few milliseconds.
  If the test takes several minutes or more, consider making the
  test a :ref:`big test <big-test>`.

* It should not depend on other scripts.

* It should be possible to run it on 1, 2, 4, and 8 cores.

A test can produce standard output and files - it doesn't have to
clean up.  Just add the ``in_tmp_dir`` fixture as an argument::

    def test_something(in_tmp_dir):
        # make a mess ...

Here is a parametrized test that uses :func:`pytest.approx` for
comparing floating point numbers::

    import pytest

    @pytest.mark.parametrize('x', [1.0, 1.5, 2.0])
    def test_sqr(x):
        assert x**2 == pytest.approx(x * x)


.. _big-test:
.. _agts:

Big tests
=========

The directory in :git:`gpaw/test/big/` contains a set of longer and more
realistic tests that we run every weekend.  These are submitted to a
queueing system of a large computer.


Adding new tests
----------------

To add a new test, create a script somewhere in the file hierarchy ending with
``agts.py`` (e.g. ``submit.agts.py`` or just ``agts.py``). ``AGTS`` is short
for Advanced GPAW Test System (or Another Great Time Sink). This script
defines how a number of scripts should be submitted to niflheim and how they
depend on each other. Consider an example where one script, ``calculate.py``,
calculates something and saves a ``.gpw`` file and another script,
``analyse.py``, analyses this output. Then the submit script should look
something like::

    def create_tasks():
        from myqueue.task import task
        return [task('calculate.py', cores=8, tmax='25m'),
                task('analyse.py', cores=1, tmax='5m',
                     deps=['calculate.py'])]

As shown, this script has to contain the definition of the function
create_tasks_.  Start the workflow with ``mq workflow -p agts.py .``
(see https://myqueue.readthedocs.io/ for more details).

.. _create_tasks: https://myqueue.readthedocs.io/en/latest/
    workflows.html#create_tasks
