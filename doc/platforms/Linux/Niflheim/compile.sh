#!/bin/sh
nh=doc/platforms/Linux/Niflheim
rm -rf build
cmd="cd $PWD && python setup.py --remove-default-flags build_ext"
ssh sylg "$cmd --customize=$nh/el7-iomkl.py > broadwell-el7-iomkl.log 2>&1"
ssh thul "$cmd --customize=$nh/el7-iomkl.py > sandybridge-el7-iomkl.log 2>&1"
ssh fjorm "$cmd --customize=$nh/el7-iomkl.py > nehalem-el7-iomkl.log 2>&1"
(cd build && ln -sf bin.linux-x86_64-{sandybridge,ivybridge}-el7-3.6)
(cd build && ln -sf lib.linux-x86_64-{sandybridge,ivybridge}-el7-3.6)
