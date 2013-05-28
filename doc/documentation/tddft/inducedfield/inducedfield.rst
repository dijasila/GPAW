.. _inducedfield:

======================================
Induced electric near field from TDDFT
======================================

Induced electric near field can be calculated for finite systems from
:ref:`inducedfield_timepropagation` or :ref:`inducedfield_casida`.

.. warning::
	This code is experimental and documentation is not complete!

.. _inducedfield_timepropagation:

Time-propagation TDDFT
===========================================

See :ref:`timepropagation` for instructions how to use time-propagation TDDFT.

Example code for time-propagation calculation

.. literalinclude:: timepropagation_calculate.py

Example code for continuing time-propagation

.. literalinclude:: timepropagation_continue.py

Induced electric potential and near field are calculated after time-propagation as follows:

.. literalinclude:: timepropagation_postprocess.py

|na2_td_Frho| |na2_td_Fphi| |na2_td_Ffe|


.. |na2_td_Frho| image:: na2_td_Frho.png
.. |na2_td_Fphi| image:: na2_td_Fphi.png
.. |na2_td_Ffe| image:: na2_td_Ffe.png


.. _inducedfield_casida:

Linear response TDDFT (Casida's equation)
===========================================

