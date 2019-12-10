import numpy as np

def get_nbars(n_g, npts=100):
    min_dens = np.min(n_g)
    max_dens = np.max(n_g)


    if np.allclose(min_dens, max_dens):
        nb_i = np.linspace(0.8 * min_dens, 1.2 * max_dens, npts)
    else:
        # nb_i = np.exp(np.linspace(np.log(min_dens), np.log(max_dens), npts))
        nb_i = np.linspace(min_dens, max_dens, npts)
    return nb_i


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n_g = np.random.rand(5,5,5)
    nb_i = get_nbars(n_g)
    plt.plot(nb_i, '.')
    plt.show()
