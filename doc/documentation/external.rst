.. module:: gpaw.external

External potential
==================

External potentials are applied to all charged particles, i.e. electrons
and nuclei.


Examples
--------

>>> # 2.5 eV/Ang along z:
>>> from gpaw.external import ConstantElectricField
>>> calc = GPAW(external=ConstantElectricField(2.5, [0, 0, 1]), ...)

.. autoclass:: ConstantElectricField

>>> # Two point-charges:
>>> from gpaw.external import PointChargePotential
>>> pc = PointChargePotential([-1, 1], [[4.0, 4.0, 0.0], [4.0, 4.0, 10.0]])
>>> calc = GPAW(external=pc, ...)

.. autoclass:: PointChargePotential

>>> # Step potential in z-direction
>>> from gpaw.external import StepPotentialz
>>> step = StepPotentialz(zstep=10, value_right=-7)
>>> calc = GPAW(external=step, ...)

.. autoclass:: StepPotentialz
.. autoclass:: gpaw.bfield.BField


Several potentials
------------------

A collection of potentials can be applied using :class:`PotentialCollection`

>>> from gpaw.external import ConstantElectric, PointChargePotential
>>> from gpaw.external import PotentialCollection
>>> ext1 = ConstantElectricField(1)
>>> ext2 = PointChargePotential([1, -5], positions=((0, 0, -10), (0, 0, 10)))
>>> collection = PotentialCollection([ext1, ext2])
>>> calc = GPAW(external=collection, ...)

.. autoclass:: PotentialCollection


Your own potential
------------------

See an example here: :git:`gpaw/test/ext_potential/test_harmonic.py`.
