.. _eels_exercise:

=======================================
Electron energy loss spectrum of silver
=======================================

Electron energy loss spectroscopy (EELS) is a widely used method to obtain the excitation spectrum of materials. For metallic and semiconducting materials, the energy losses in the range 0-50 eV is primarily due excitations of plasmons, that are collective electronic excitations, corresponding to oscillations in the electron density.  For a free-electron metal (Drude metal), the plasmon energy is given by the electron density, n: 

.. math:: \omega_p = \sqrt{\frac{ne^2}{\epsilon_0 m}}. 

In GPAW, the EELS can be calculated with the dielectric response module, where it is obtained from the macroscopic dielectric function: 

.. math:: \mathrm{EELS}(q, \omega) = -\mathrm{Im} \frac{1}{\epsilon_M(q,\omega)}, 

where q is the momentum transfer. (Se the tutorial :ref:`df_tutorial` for a detailed description of the dielectric response.) 

Here we will calculate the EELS for bulk silver, where band-structure effects (coupling to inter-band transitions) are seen to have a big impact on the plasmon resonance, which means that the Drude description for the plasmon energy given above is not expected to hold.
As a first step we perform a ground-state calculation for bulk silver. It is important to get a good description of the d-band. This means that we must be careful with the choice of exchange-correlation functional, since these states are generally poorly described within LDA. (Why do you think that is?) Download the script :download:`silver_bandstructure.py` that calculates the bandstructure of silver with LDA. Read the script and try to get an idea of what it will do, and run the script by typing::

  python silver_bandstructure.py

The bandstructure is plotted in the end of the script. Where is the d-band located? Experimentally it's found to be approximately 4 eV below the Fermi-level. 
Now modify the script so the bandstructure is calculated with the GLLBSC functional. Is the energy position of the d-band improved? 

Now we are ready to calculate the loss spectrum. First restart the ground state calculation and converge a larger number of bands::

  import numpy as np
  from gpaw import GPAW
  from gpaw.response.df import DielectricFunction

  calc = GPAW('Ag_GLLBSC.gpw')
  calc.diagonalize_full_hamiltonian(nbands = 30) 
  calc.write('Ag_GLLBSC_full.gpw', 'all')

Then we can set up the dielectric function, taking the ground state as input::

  df = DielectricFunction(calc='Ag_GLLBSC_full.gpw',
                          alpha=0.0,      # use linear frequency grid
                          domega0 = 0.05) # energy grid spacing

The EELS spectrum is calculated with the method ``get_eels_spectrum()``, that takes the momentum transfer q as a parameter. This parameter is restricted to be the difference between two k-points from the ground state calculation, so let's choose the smallest q possible::

  q_c = np.array([1./10., 0, 0])
  df.get_eels_spectrum(q_c=q_c)

The calculation takes some time due to a sum over k-points. To speed up the calculation download the script :download:`silver_EELS.py`, and run it in parallel::

  mpirun -np 4 gpaw-python silver_EELS.py

The calculation saves the file ``eels.csv`` by default, where the first column is the energy grid, and the second and third columns are the loss spectrum without and with local field corrections respectively. (The local field corrections takes into account that the system responds on the microscopic scale though the perturbation is macroscopic). You can plot the spectrum by like this::
 
  from numpy import genfromtxt
  import pylab as p
  
  data = genfromtxt('eels.csv', delimiter=','  )
  omega = d[:,0]
  eels = d[:,2]
  p.plot(omega, eels)
  p.xlabel('Energy (eV)')
  p.ylabel('Loss spectrum')
  p.xlim(0,20)
  p.show()

Look at the spectrum, where is the plasmon peak? Compare the result to the experimental plasmon energy :math:`\omega_P \approx 3.9 \mathrm{eV}`. Also compare the result to the Drude value for the plasmon frequency given above. (Hint: For silver there is one valence s electron pr. atom, use this to calculate the free-electron density.)

Hopefully you will find that there is large difference between the free electron and the quantum result. The plasmon is damped and shifted down in energy due to coupling to single-particle transitions (inter-band transitions). Here the d-band of silver plays a crucial role, since transitions from here up to the Fermi level defines the onset of inter-band transitions. For example you can calculate the loss spectrum from the LDA ground state result and see what is does to the spectrum. You can also investigate the single-particle spectrum by calculating the dielectric function::

  df.get_dielectric_function(q_c=q_c)

which saves a file ``df.csv``. Try plotting the imaginary part (column 5 in the data file), which corresponds to the single-particle spectrum. Compare this to the loss spectrum to see that the plasmon peak is shifted down just below the single-particle transitions. 

