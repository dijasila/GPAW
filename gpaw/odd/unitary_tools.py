""" Evaluates atomic charges or wavefunction spread as well as 
    the derivative and cost functions nessecary for unitary optimization.

    ### DETAILS ON THE AVIALABLE COST func and DER func will follow

    Also contains the Brockett class for the unitary optimization
    methods - test class where the objective function and derivative
    are defined, determined and easily understood.

    A) R. A. Brockett, Dynamic-systems that sort lists, diagonalize
    matrices, and solve linear programming problems, Lin. Alg. Appl.
    146, 79 (1991)

"""

import numpy as np
import numpy.random as npr
from scipy.linalg import sqrtm as sqm
from scipy.linalg import inv

class Brockett:
    """ The function is

        F(W) = Tr{W*EWN}

        subject to W*W = I = WW* ( W e U(n) )

        E is here some hermitian n x n matrix, and N is the diagonal
        matrix with N.diag = [1,2,...,n].
 
        The euclidian derivative is
 
        dF(W)
        ----  = EWN
         dW* 

        Maximizing the function using the unitary optimizer gives: 
        1) W converges to the eigenvectors of E
        2) the matrix D = W*EW will converge to a diagonal matrix
        containing the eigenvalues of E sorted in the ascending order
        of the diagonal.        

        The 'convergence' criteria here is expressed in 
        .get_function_tol() - and is saved to unitary_opt.ftol
        array.

    """
    def __init__(self, E = None, size = 6, rnd = 'normal'):
        self.E = E       # if None is provided one will be built
        self.size = size # dim of the n x n matrix
        self.rnd = rnd   # random number generator to be used
        # Either 'normal' for rand, 
        # std for randn (standard normal dist).

        self.dE  = None   # hold on to dE, euclidian derivative
        self.dG  = None   # hold on to dG, reimannian derivative

        self.N   = None   # hold on to N
        self.W_k = None   # hold on to W_k


    def get_euclidian_der(self, W_k=None, io=False):
        # First step in the U(n) optimization
        # Check if E was provided
        self.W_k = W_k.copy()

        if self.E is None: # Probably initialization
            self.E = random_hermitian(self.size)
        else: # Make sure it is symmetric
            assert (np.shape(self.E) == np.shape(self.E.T)), \
                   'Objective matrix is not symmetric!'
        
        # Make sure it is hermitian?
        assert check_hermitian(self.E), \
               'Objective matrix is not Hermitian!'

        # Check the unitary matrix:
        if self.W_k is None: # Make Identity
            self.W_k = random_orthogonal(np.shape(self.E)[0])

        else: # Make sure W_k makes sense
            assert (np.shape(self.E) == np.shape(W_k)), \
                   'Unitary and objective matrices asymmetric'

        # N is n x n with diagonal 1,...,n
        if self.N is None:
            self.N  = np.diag(range(1,np.shape(self.E)[0]+1))

        # derivative
        self.dE = self.E.dot(self.W_k.dot(self.N)).copy()
 
        ## TEST ##
        if io:
            return self.dE       


    def get_reimannian_der(self, W_k = None, io=False): #TST purposes
        # Given the euclidian derivative and the unitary matrix 
        # get the Reimannian derivative
        self.W_k = W_k.copy()

        # Check matrices
        if self.E is None:
            self.get_euclidian_der(self.W_k)
        elif self.dE is None: #CHK
            self.get_euclidian_der(self.W_k)

        # Check the unitary matrix:
        if W_k is None: # Make random orthogonal matrix
            self.W_k = random_orthogonal(np.shape(self.E)[0], \
                                         rand = self.rnd)
        else: # Make sure W_k makes sense
            assert (np.shape(self.E) == np.shape(W_k)), \
                   'Unitary and objective matrices asymmetric'

        # Get Reimannian derivative
        self.dG = self.dE.dot(self.W_k.T.conj()) - \
                          self.W_k.dot(self.dE.T.conj())

        # Check and make sure the matrix is skew-hermitian
        assert check_skew_hermitian(self.dG), \
            'Reimanniand derivative is not skew-hermitian!' 

        ## TEST ##
        if io:
            return self.dG


    def get_threshold(self, W_k = None):
        # The value here is passed on to the
        # unitary.fthr array of dimension 'n'<'nsteps'
        """ Here a diagonality criterion is checked

                           off{W*EW}
            Delta = 10log ----------
                           diag{W*EW}

            where the off{-} operator computes the squared
            magnitudes of the off-diagonal elements of a matrix
            and diag{-} does the same but for the diagonal
            elements.

        """
        matrix = W_k.T.conj().dot(self.E.dot(W_k))
        top    = ((matrix - np.diag(np.diag(matrix)))**2).sum()
        bottom = (np.diag(matrix)**2).sum()
        
        return 10.0*np.log10(top/bottom)


    def get_order(self):
        # see Lehtola2013; ref A) in unitary doc
        # In short: q is the order where W appears in the Taylor 
        # series expansion of F(W) 
        # Also; see pg. 1140 ref D) in unitary doc
        return 2.0 # 


    def get_tolerance(self):
        return 1e-10


### Initial matrices; E and W_k and whatnot ###
def random_orthogonal(s, rand = 'normal', dtype = float, 
                      rattle = 0.0):
    # Make a random orthogonal matrix of dim s x s, 
    # such that WW* = I = W*W
    # 'normal': rnd generator
    # 'std'   : standard normal (~N(mu,sigma**2)) rnd generator
    # cmplx   : complex is False by default
    # rattle  :

    if rand == 'normal':
        w_r = npr.rand(s,s)
        if dtype == complex:
            w_r = w_r + 1.j * npr.rand(s,s)
        return w_r.dot(inv(sqm(w_r.T.conj().dot(w_r))))
    elif rand == 'std':
        w_r = npr.randn(s,s)
        if dtype == complex:
            w_r = w_r + 1.j * npr.randn(s,s)
        return w_r.dot(inv(sqm(w_r.T.conj().dot(w_r))))
    else:
        raise Exception('Check doc. for allowed rnd. generators!')


def random_hermitian(s, rand = 'normal', cmplx = False):
    # Make a random Hermitian matrix of dim s x s
    # such that W - W* = 0
    # 'normal': rnd generator
    # 'std'   : standar normal (~N(mu,sigma**2)) rnd generator
    # cmplx   : complex is False by default

    if rand == 'normal':
        w_r = npr.rand(s,s)
        if cmplx:
            w_r = w_r + 1.j * npr.rand(s,s)
        return 0.5 * (w_r + w_r.T.conj())
    elif rand == 'std':
        w_r = npr.randn(s,s)
        if cmplx:
            w_r = w_r + 1.j * npr.randn(s,s)
        return 0.5 * (w_r + w_r.T.conj())
    else:
        raise Exception('Check doc. for allowed rnd. generators!')


### CHECK/MAKE MATRICES
### Numerical acc at? 1e-10->1e-14...
def check_hermitian(H, crit=1e-10):
    # Check if matrix is hermitian
    val = H - H.T.conj()
    return abs(val.sum()) < crit


def make_hermitian(H):
    return 0.5 * (H + H.T.conj())


def check_skew_hermitian(H, crit=1e-10):
    # Check if matrix is skew hermitian
    val = H + H.T.conj()
    return abs(val.sum()) < crit


def make_skew_hermitian(H):
    return 0.5 * (H - H.T.conj())


def check_orthogonal(H, crit=1e-10):
    # Check if matrix is orthogonal
    I = (H.dot(H.T.conj()))
    D = (np.diag(I)).sum()
    return abs(I.sum() - D) < crit


def check_norm(H, crit=1e-13):
    # Check if matrix is unitary / normalized
    return (H.T.conj().dot(H) == H.dot(H.T.conj())).all() < crit


def make_norm(H):
    return H.dot(inv(sqm(H.T.conj().dot(H))))


def bracket(H, G):
    return 0.5 * np.real(np.trace(H.dot(G)))

