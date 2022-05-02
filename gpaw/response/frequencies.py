import numbers

import numpy as np
from ase.units import Ha


class FrequencyDescriptor:
    """Describes a single dimensional array."""
    def __init__(self, omega_w):
        self.omega_w = np.asarray(omega_w).copy()

    def __len__(self):
        return len(self.omega_w)

    def __repr__(self):
        emin = self.omega_w[0] * Ha
        emax = self.omega_w[-1] * Ha
        return (f'{self.__class__.__name__}'
                f'(from {emin:.3f} to {emax:.3f} eV, {len(self)} points)')

    @staticmethod
    def from_array_or_dict(input):
        if isinstance(input, dict):
            assert input['type'] == 'nonlinear'
            domega0 = input.get('domega0')
            omega2 = input.get('omega2')
            omegamax = input.get('omegamax')
            return NonLinearFrequencyDescriptor(
                (0.1 if doemga0 is None else domega0) / Ha,
                (10.0 if oemga2 is None else omega2) / Ha,
                omegamax / Ha)
        return LinearFrequencyDescriptor(np.asarray(input) / Ha)


class LinearFrequencyDescriptor(FrequencyDescriptor):
    def get_closest_index(self, scalars_w):
        """Get closest index.

        Get closest index approximating scalars from below."""
        diff_xw = self.omega_w[:, np.newaxis] - scalars_w[np.newaxis]
        return np.argmin(diff_xw, axis=0)

    def get_index_range(self, lim1_m, lim2_m):
        """Get index range. """

        i0_m = np.zeros(len(lim1_m), int)
        i1_m = np.zeros(len(lim2_m), int)

        for m, (lim1, lim2) in enumerate(zip(lim1_m, lim2_m)):
            i_x = np.logical_and(lim1 <= self.omega_w,
                                 lim2 >= self.omega_w)
            if i_x.any():
                inds = np.argwhere(i_x)
                i0_m[m] = inds.min()
                i1_m[m] = inds.max() + 1

        return i0_m, i1_m


class NonLinearFrequencyDescriptor(FrequencyDescriptor):
    def __init__(self,
                 domega0: float,
                 omega2: float,
                 omegamax: float):
        beta = (2**0.5 - 1) * domega0 / omega2
        wmax = int(omegamax / (domega0 + beta * omegamax))
        w = np.arange(wmax + 2)  # + 2 is for buffer
        omega_w = w * domega0 / (1 - beta * w)

        super().__init__(omega_w)

        self.domega0 = domega0
        self.omega2 = omega2
        self.omegamax = omegamax
        self.omegamin = 0

        self.beta = beta
        self.wmax = wmax
        self.omega_w = omega_w
        self.wmax = wmax

    def get_closest_index(self, o_m):
        beta = self.beta
        w_m = (o_m / (self.domega0 + beta * o_m)).astype(int)
        if isinstance(w_m, np.ndarray):
            w_m[w_m >= self.wmax] = self.wmax - 1
        elif isinstance(w_m, numbers.Integral):
            if w_m >= self.wmax:
                w_m = self.wmax - 1
        else:
            raise TypeError
        return w_m

    def get_index_range(self, omega1_m, omega2_m):
        omega1_m = omega1_m.copy()
        omega2_m = omega2_m.copy()
        omega1_m[omega1_m < 0] = 0
        omega2_m[omega2_m < 0] = 0
        w1_m = self.get_closest_index(omega1_m)
        w2_m = self.get_closest_index(omega2_m)
        o1_m = self.omega_w[w1_m]
        o2_m = self.omega_w[w2_m]
        w1_m[o1_m < omega1_m] += 1
        w2_m[o2_m < omega2_m] += 1
        return w1_m, w2_m
