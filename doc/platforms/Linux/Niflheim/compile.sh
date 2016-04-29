#!/bin/sh
nh=doc/platforms/Linux/Niflheim
c1=$nh/el6-x3455-tm-gfortran-openmpi-1.6.3-acml-4.4.0-sl-hdf5-1.8.10.py
c2=$hn/el6-sl230s-tm-gfortran-openmpi-1.6.3-acml-4.4.0-sl-hdf5-1.8.10.py
c3=$nh/el6-dl160g6-tm-gfortran-openmpi-1.6.3-acml-4.4.0-sl-hdf5-1.8.10.py
rm -rf build
cmd="cd $PWD && python setup.py --remove-default-flags build_ext"
ssh slid "$cmd --customize=$c1 > x3455.log 2>&1"
ssh surt "$cmd --customize=$c2 > sl230s.log 2>&1"
ssh muspel "$cmd --customize=$c3 > dl160g6.log 2>&1"
