.. _band_exercise:

==============
Band structure
==============

Band diagrams are usefull analysis tools. Read :download:`Na_band.py` and try
to understand what it does, then use it to construct the band diagram of bulk
Na.  Read :download:`plot_band.py` and try to understand it, then use it to
plot the band diagram.

As a next step, calculate the bandstructure of silver. Here we should be careful with the choice of exchange-correlation functional to get a good description of the d-band, which is generally poorly described within LDA. (Why do you think that is?). Download the script :download:`silver_bandstructure.py` that calculates the bandstructure of silver with LDA. 

The bandstructure is plotted in the end of the script. Where is the d-band located? Experimentally it's found to be approximately 4 eV below the Fermi-level. 
Now modify the script so the bandstructure is calculated with the GLLBSC functional. Is the energy position of the d-band improved? 
