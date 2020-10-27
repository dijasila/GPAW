# Import the required modules: General
import numpy as np
from pathlib import Path
import sys

# Import the required modules: GPAW/ASE
from gpaw.mpi import world, broadcast

# Load the mometum and gs file in master


def load_data(mml_name='mml.npz'):
    """
    Load the data and distribute among cores

    Input:
        mml_name        NLO data filename (default mml.npz)
    Output:

        p_kvnn          The mometum matrix elements, dimension (nk,3,nb,nb)
    """

    # Load the data to the master
    if world.rank == 0:
        nlo = np.load(mml_name)
        # print(nlo.files)
        # print(nlo['p_kvnn'])
    else:
        nlo = dict.fromkeys(['w_k', 'f_kn', 'E_kn', 'p_kvnn'])

    # Distribute the data among cores
    k_info = distribute_data(
        [nlo['w_k'], nlo['f_kn'], nlo['E_kn'], nlo['p_kvnn']])

    # Return the output dict
    return k_info


# Distribute the data among the cores

def distribute_data(arr_list):
    """
    Distribute the data among the cores

    Input:
        arr_list        A list of numpy array (the first index should be k)
    Output:
        k_info          A  dictionary of data with key of k index
    """

    # Check the array shape
    size = world.size
    rank = world.rank
    if rank == 0:
        nk = 0
        arr_shape = []
        for ii, arr in enumerate(arr_list):
            arr_shape.append(arr.shape)
            if nk == 0:
                nk = arr_shape[-1][0]
            else:
                # print(arr_shape[-1])
                assert arr_shape[-1][0] == nk, 'Wrong shape for array.'
    else:
        arr_shape = None
        nk = None
    arr_shape = broadcast(arr_shape, root=0)
    nk = broadcast(nk, root=0)

    # Distribute the data of k-points between cores
    k_info = {}

    # Loop over k points
    for kk in range(nk):
        if rank == 0:
            if kk % size == rank:
                k_info[kk] = [arr[kk] for arr in arr_list]
            else:
                for ii, arr in enumerate(arr_list):
                    data_k = np.array(arr[kk], dtype=complex)
                    world.send(
                        data_k, dest=kk % size, tag=ii * nk + kk)
        else:
            if kk % size == rank:
                dataset = []
                for ii, cshape in enumerate(arr_shape):
                    data_k = np.empty(cshape[1:], dtype=complex)
                    world.receive(data_k, src=0, tag=ii * nk + kk)
                    dataset.append(data_k)
                k_info[kk] = dataset

    # Return the dictionary
    return k_info


# Print iterations progress


def print_progressbar(iteration, total, prefix='Progress:', suffix='Complete',
                      decimals=1, length=50, fill='#', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar

    Input:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals complete
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if world.rank == 0:
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(
            '\r%s |%s| %s%% %s' %
            (prefix, bar, percent, suffix), end=printEnd)
        sys.stdout.flush()

        # Print New Line on Complete
        if iteration == total:
            print()
            sys.stdout.flush()

# Check if the file exist or not


def is_file_exist(filename):
    """
    Check if the file exist

    Input:
        filename        Filename to check
    Output:
        file_exist      Flag for file
    """
    if world.rank == 0:
        if (not Path(filename).is_file()):
            file_exist = True
        else:
            file_exist = False
    else:
        file_exist = None

    file_exist = broadcast(file_exist, 0)

    # Return the output
    return file_exist


# Prints only from master


def parprint(*args, **kwargs):
    """MPI-safe print - prints only from master. """
    if world.rank == 0:
        print(*args, **kwargs)
        sys.stdout.flush()
