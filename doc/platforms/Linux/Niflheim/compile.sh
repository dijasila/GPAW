#!/bin/sh
nh=doc/platforms/Linux/Niflheim
rm -rf build
cmd="cd $PWD && python setup.py --remove-default-flags build_ext"
ssh sylg "$cmd --customize=$nh/el7.py > broadwell-el7.log 2>&1"
ssh thul "$cmd --customize=$nh/el7.py > sandybridge-el7.log 2>&1"
ssh fjorm "$cmd --customize=$nh/el7.py > nehalem-el7.log 2>&1"
ln -sf bin.linux-x86_64-{sandybridge,ivybridge}-el7-3.5
ln -sf lib.linux-x86_64-{sandybridge,ivybridge}-el7-3.5
