#Needed paths and libraries
module load mpi/mvapich2-2.3a_intel  #Careful: OpenMPI intel compiler does not work!
module load python/3.6.6

#ase needs to be in the PYTHONPATH
#export PYTHONPATH=$HOME/data/gkastlun/opt/ase_update_for_gpaw15/ase:$PYTHONPATH
softpath=/gpfs/data/ap31/software
export PYTHONPATH=$softpath/ase_for_gpaw_1.5:$PYTHONPATH

#Let's use our custom customize.py, which includes all the paths for libs on the cluster
cp customize.py.bakk customize.py

#Compile
rm -fr build
#python setup.py build
python3 setup.py build_ext

#Reset the customize.py so it won't show up in git diff
git checkout customize.py
