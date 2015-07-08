class Perturbation:
    """Abstract base class for various perturbation types.

    Specific perturbations must derive from this class. As a minimum a
    perturbation should provide an ``apply`` member function that implements
    the action of the perturbing potential onto a wave function.
    """

    def __init__(self):
        pass
