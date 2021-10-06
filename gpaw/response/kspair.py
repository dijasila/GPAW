from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from gpaw.utilities import convert_string_to_fd
from ase.utils.timing import Timer, timer

from gpaw import disable_dry_run
from gpaw.calculator import GPAW
import gpaw.mpi as mpi
from gpaw.response.math_func import two_phi_planewave_integrals


class KohnShamKPoint:
    """Kohn-Sham orbital information for a given k-point."""
    def __init__(self, n_t, s_t, K,
                 eps_t, f_t, ut_tR, projections, shift_c):
        """K-point data is indexed by a joint spin and band index h, which is
        directly related to the transition index t."""
        self.K = K                      # BZ k-point index
        self.n_t = n_t                  # Band index
        self.s_t = s_t                  # Spin index
        self.eps_t = eps_t              # Eigenvalues
        self.f_t = f_t                  # Occupation numbers
        self.ut_tR = ut_tR              # Periodic part of wave functions
        self.projections = projections  # PAW projections

        self.shift_c = shift_c  # long story - see the
        # PairDensity.construct_symmetry_operators() method


class KohnShamKPointPair:
    """Object containing all transitions between Kohn-Sham orbitals from a
    specified k-point to another."""

    def __init__(self, kpt1, kpt2, mynt, nt, ta, tb, comm=None):
        self.kpt1 = kpt1
        self.kpt2 = kpt2

        self.mynt = mynt  # Number of transitions in this block
        self.nt = nt      # Total number of transitions between all blocks
        self.ta = ta      # First transition index of this block
        self.tb = tb      # First transition index of this block not included
        self.comm = comm  # MPI communicator between blocks of transitions

    def transition_distribution(self):
        """Get the distribution of transitions."""
        return self.mynt, self.nt, self.ta, self.tb

    def get_transitions(self):
        return self.n1_t, self.n2_t, self.s1_t, self.s2_t

    def get_all(self, A_mytx):
        """Get a certain data array with all transitions"""
        if self.comm is None or A_mytx is None:
            return A_mytx

        A_tx = np.empty((self.mynt * self.comm.size,) + A_mytx.shape[1:],
                        dtype=A_mytx.dtype)

        self.comm.all_gather(A_mytx, A_tx)

        return A_tx[:self.nt]

    @property
    def n1_t(self):
        return self.get_all(self.kpt1.n_t)

    @property
    def n2_t(self):
        return self.get_all(self.kpt2.n_t)

    @property
    def s1_t(self):
        return self.get_all(self.kpt1.s_t)

    @property
    def s2_t(self):
        return self.get_all(self.kpt2.s_t)

    @property
    def deps_t(self):
        return self.get_all(self.kpt2.eps_t) - self.get_all(self.kpt1.eps_t)

    @property
    def df_t(self):
        return self.get_all(self.kpt2.f_t) - self.get_all(self.kpt1.f_t)

    @classmethod
    def add_mytransitions_array(cls, _key, key):
        """Add a A_tx data array class attribute.
        Handles the fact, that the transitions are distributed in blocks.

        Parameters
        ----------
        _key : str
            attribute name for the A_mytx data array
        key : str
            attribute name for the A_tx data array
        """
        # In general, the data array has not been specified to instances of
        # the class. As a result, set the _key to None
        setattr(cls, _key, None)
        # self.key should return full data array
        setattr(cls, key,
                property(lambda self: self.get_all(self.__dict__[_key])))

    def attach(self, _key, key, A_mytx):
        """Attach a data array to the k-point pair.
        Used by PairMatrixElement to attach matrix elements calculated
        between the k-points for the different transitions."""
        self.add_mytransitions_array(_key, key)
        setattr(self, _key, A_mytx)


class KohnShamPair:
    """Class for extracting pairs of Kohn-Sham orbitals from a ground
    state calculation."""

    def __init__(self, gs, world=mpi.world, transitionblockscomm=None,
                 kptblockcomm=None, txt='-', timer=None):
        """
        Parameters
        ----------
        transitionblockscomm : gpaw.mpi.Communicator
            Communicator for distributing the transitions among processes
        kptblockcomm : gpaw.mpi.Communicator
            Communicator for distributing k-points among processes
        """
        self.world = world
        self.fd = convert_string_to_fd(txt, world)
        self.timer = timer or Timer()
        self.calc = get_calc(gs, fd=self.fd, timer=self.timer)
        self.calc_parallel = self.check_calc_parallelisation()

        self.transitionblockscomm = transitionblockscomm
        self.kptblockcomm = kptblockcomm

        # Prepare to distribute transitions
        self.mynt = None
        self.nt = None
        self.ta = None
        self.tb = None

        # Prepare to find k-point data from vector
        kd = self.calc.wfs.kd
        self.kdtree = cKDTree(np.mod(np.mod(kd.bzk_kc, 1).round(6), 1))

        # Prepare to use other processes' k-points
        self._pd0 = None

        # Prepare to redistribute kptdata
        self.rrequests = []
        self.srequests = []

        # Count bands so it is possible to remove null transitions
        self.nocc1 = None  # number of completely filled bands
        self.nocc2 = None  # number of non-empty bands
        self.count_occupied_bands()

    def check_calc_parallelisation(self):
        """Check how ground state calculation is distributed in memory"""
        if self.calc.world.size == 1:
            return False
        else:
            assert self.world.rank == self.calc.wfs.world.rank
            assert self.calc.wfs.gd.comm.size == 1
            return True

    def count_occupied_bands(self):
        """Count number of occupied and unoccupied bands in ground state
        calculation. Can be used to omit null-transitions between two occupied
        bands or between two unoccupied bands."""
        ftol = 1.e-9  # Could be given as input
        nocc1 = 9999999
        nocc2 = 0
        for kpt in self.calc.wfs.kpt_u:
            f_n = kpt.f_n / kpt.weight
            nocc1 = min((f_n > 1 - ftol).sum(), nocc1)
            nocc2 = max((f_n > ftol).sum(), nocc2)
        nocc1 = int(nocc1)
        nocc2 = int(nocc2)

        # Collect nocc for all k-points
        nocc1 = self.calc.wfs.kd.comm.min(nocc1)
        nocc2 = self.calc.wfs.kd.comm.max(nocc2)

        # Sum over band distribution
        nocc1 = self.calc.wfs.bd.comm.sum(nocc1)
        nocc2 = self.calc.wfs.bd.comm.sum(nocc2)

        self.nocc1 = int(nocc1)
        self.nocc2 = int(nocc2)
        print('Number of completely filled bands:', self.nocc1, file=self.fd)
        print('Number of partially filled bands:', self.nocc2, file=self.fd)
        print('Total number of bands:', self.calc.wfs.bd.nbands,
              file=self.fd)

    @property
    def pd0(self):
        """Get a PWDescriptor that includes all k-points"""
        if self._pd0 is None:
            from gpaw.pw.descriptor import PWDescriptor
            wfs = self.calc.wfs
            assert wfs.gd.comm.size == 1

            kd0 = wfs.kd.copy()
            pd, gd = wfs.pd, wfs.gd

            # Extract stuff from self.calc.wfs.pd
            ecut, dtype = pd.ecut, pd.dtype
            fftwflags, gammacentered = pd.fftwflags, pd.gammacentered

            # Initiate _pd0 with kd0
            self._pd0 = PWDescriptor(ecut, gd, dtype=dtype,
                                     kd=kd0, fftwflags=fftwflags,
                                     gammacentered=gammacentered)
        return self._pd0

    @timer('Get Kohn-Sham pairs')
    def get_kpoint_pairs(self, n1_t, n2_t, k1_pc, k2_pc, s1_t, s2_t):
        """Get all pairs of Kohn-Sham orbitals for transitions:
        (n1_t, k1_p, s1_t) -> (n2_t, k2_p, s2_t)
        Here, t is a composite band and spin transition index
        and p is indexing the different k-points to be distributed."""

        # Distribute transitions and extract data for transitions in
        # this process' block
        nt = len(n1_t)
        assert nt == len(n2_t)
        self.distribute_transitions(nt)

        kpt1 = self.get_kpoints(k1_pc, n1_t, s1_t)
        kpt2 = self.get_kpoints(k2_pc, n2_t, s2_t)

        # The process might not have any k-point pairs to evaluate, as
        # their are distributed in the kptblockcomm
        if kpt1 is None:
            assert kpt2 is None
            return None
        assert kpt2 is not None

        return KohnShamKPointPair(kpt1, kpt2,
                                  self.mynt, nt, self.ta, self.tb,
                                  comm=self.transitionblockscomm)

    def distribute_transitions(self, nt):
        """Distribute transitions between processes in block communicator"""
        if self.transitionblockscomm is None:
            mynt = nt
            ta = 0
            tb = nt
        else:
            nblocks = self.transitionblockscomm.size
            rank = self.transitionblockscomm.rank

            mynt = (nt + nblocks - 1) // nblocks
            ta = min(rank * mynt, nt)
            tb = min(ta + mynt, nt)

        self.mynt = mynt
        self.nt = nt
        self.ta = ta
        self.tb = tb

    def get_kpoints(self, k_pc, n_t, s_t):
        """Get KohnShamKPoint and help other processes extract theirs"""
        assert len(n_t) == len(s_t)
        assert len(k_pc) <= self.kptblockcomm.size
        kpt = None

        # Use the data extraction factory to extract the kptdata
        _extract_kptdata = self.create_extract_kptdata()
        kptdata = _extract_kptdata(k_pc, n_t, s_t)

        # Make local n and s arrays for the KohnShamKPoint object
        n_myt = np.empty(self.mynt, dtype=n_t.dtype)
        n_myt[:self.tb - self.ta] = n_t[self.ta:self.tb]
        s_myt = np.empty(self.mynt, dtype=s_t.dtype)
        s_myt[:self.tb - self.ta] = s_t[self.ta:self.tb]

        # Initiate k-point object.
        if self.kptblockcomm.rank in range(len(k_pc)):
            assert kptdata is not None
            kpt = KohnShamKPoint(n_myt, s_myt, *kptdata)

        return kpt

    def create_extract_kptdata(self):
        """Creator component of the data extraction factory."""
        if self.calc_parallel:
            return self.parallel_extract_kptdata
        else:
            return self.serial_extract_kptdata
            # Useful for debugging:
            # return self.parallel_extract_kptdata

    def parallel_extract_kptdata(self, k_pc, n_t, s_t):
        """Returns the input to KohnShamKPoint:
        K, n_myt, s_myt, eps_myt, f_myt, ut_mytR, projections, shift_c
        if a k-point in the given list, k_pc, belongs to the process.
        """
        # Extract the data from the ground state calculator object
        data, h_myt, myt_myt = self._parallel_extract_kptdata(k_pc, n_t, s_t)

        # If the process has a k-point to return, symmetrize and unfold
        if self.kptblockcomm.rank in range(len(k_pc)):
            assert data is not None
            # Unpack data, apply FT and symmetrization
            K, k_c, eps_h, f_h, Ph, psit_hG = data
            Ph, ut_hR, shift_c = self.transform_and_symmetrize(K, k_c, Ph,
                                                               psit_hG)

            (eps_myt, f_myt,
             P, ut_mytR) = self.unfold_arrays(eps_h, f_h, Ph, ut_hR,
                                              h_myt, myt_myt)

            data = (K, eps_myt, f_myt, ut_mytR, P, shift_c)

        # Wait for communication to finish
        with self.timer('Waiting to complete mpi.send'):
            while self.srequests:
                self.world.wait(self.srequests.pop(0))

        return data

    @timer('Extracting data from the ground state calculator object')
    def _parallel_extract_kptdata(self, k_pc, n_t, s_t):
        """In-place kptdata extraction."""
        (data, myu_eu,
         myn_eueh, ik_r2,
         nrh_r2, eh_eur2reh,
         rh_eur2reh, h_r1rh,
         h_myt, myt_myt) = self.get_extraction_protocol(k_pc, n_t, s_t)

        (eps_r1rh, f_r1rh,
         P_r1rhI, psit_r1rhG,
         eps_r2rh, f_r2rh,
         P_r2rhI, psit_r2rhG) = self.allocate_transfer_arrays(data, nrh_r2,
                                                              ik_r2, h_r1rh)

        # Do actual extraction
        for myu, myn_eh, eh_r2reh, rh_r2reh in zip(myu_eu, myn_eueh,
                                                   eh_eur2reh, rh_eur2reh):

            eps_eh, f_eh, P_ehI = self.extract_wfs_data(myu, myn_eh)

            for r2, (eh_reh, rh_reh) in enumerate(zip(eh_r2reh, rh_r2reh)):
                if eh_reh:
                    eps_r2rh[r2][rh_reh] = eps_eh[eh_reh]
                    f_r2rh[r2][rh_reh] = f_eh[eh_reh]
                    P_r2rhI[r2][rh_reh] = P_ehI[eh_reh]

            # Wavefunctions are heavy objects which can only be extracted
            # for one band index at a time, handle them seperately
            self.add_wave_function(myu, myn_eh, eh_r2reh,
                                   rh_r2reh, psit_r2rhG)

        self.distribute_extracted_data(eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG,
                                       eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG)

        data = self.collect_kptdata(data, h_r1rh, eps_r1rh,
                                    f_r1rh, P_r1rhI, psit_r1rhG)

        return data, h_myt, myt_myt

    @timer('Create data extraction protocol')
    def get_extraction_protocol(self, k_pc, n_t, s_t):
        """Figure out how to extract data efficiently.
        For the serial communicator, all processes can access all data,
        and resultantly, there is no need to send any data.
        """
        wfs = self.calc.wfs
        get_extraction_info = self.create_get_extraction_info()

        # Kpoint data
        data = (None, None, None)

        # Extraction protocol
        myu_eu = []
        myn_eueh = []

        # Data distribution protocol
        nrh_r2 = np.zeros(self.world.size, dtype=int)
        ik_r2 = [None for _ in range(self.world.size)]
        eh_eur2reh = []
        rh_eur2reh = []
        h_r1rh = [list([]) for _ in range(self.world.size)]

        # h to t index mapping
        myt_myt = np.arange(self.tb - self.ta)
        t_myt = range(self.ta, self.tb)
        n_myt, s_myt = n_t[t_myt], s_t[t_myt]
        h_myt = np.empty(self.tb - self.ta, dtype=int)

        nt = len(n_t)
        assert nt == len(s_t)
        t_t = np.arange(nt)
        nh = 0
        for p, k_c in enumerate(k_pc):  # p indicates the receiving process
            K = self.find_kpoint(k_c)
            ik = wfs.kd.bz2ibz_k[K]
            for r2 in range(p * self.transitionblockscomm.size,
                            min((p + 1) * self.transitionblockscomm.size,
                                self.world.size)):
                ik_r2[r2] = ik

            if p == self.kptblockcomm.rank:
                data = (K, k_c, ik)

            # Find out who should store the data in KSKPpoint
            r2_t, myt_t = self.map_who_has(p, t_t)

            # Find out how to extract data
            # In the ground state, kpts are indexed by u=(s, k)
            for s in set(s_t):
                thiss_myt = s_myt == s
                thiss_t = s_t == s
                t_ct = t_t[thiss_t]
                n_ct = n_t[thiss_t]
                r2_ct = r2_t[t_ct]

                # Find out where data is in wfs
                u = ik * wfs.nspins + s
                myu, r1_ct, myn_ct = get_extraction_info(u, n_ct, r2_ct)

                # If the process is extracting or receiving data,
                # figure out how to do so
                if self.world.rank in np.append(r1_ct, r2_ct):
                    # Does this process have anything to send?
                    thisr1_ct = r1_ct == self.world.rank
                    if np.any(thisr1_ct):
                        eh_r2reh = [list([]) for _ in range(self.world.size)]
                        rh_r2reh = [list([]) for _ in range(self.world.size)]
                        # Find composite indeces h = (n, s)
                        n_et = n_ct[thisr1_ct]
                        n_eh = np.unique(n_et)
                        # Find composite local band indeces
                        myn_eh = np.unique(myn_ct[thisr1_ct])

                        # Where to send the data
                        r2_et = r2_ct[thisr1_ct]
                        for r2 in np.unique(r2_et):
                            thisr2_et = r2_et == r2
                            # What ns are the process sending?
                            n_reh = np.unique(n_et[thisr2_et])
                            eh_reh = []
                            for n in n_reh:
                                eh_reh.append(np.where(n_eh == n)[0][0])
                            # How to send it
                            eh_r2reh[r2] = eh_reh
                            nreh = len(eh_reh)
                            rh_r2reh[r2] = np.arange(nreh) + nrh_r2[r2]
                            nrh_r2[r2] += nreh

                        myu_eu.append(myu)
                        myn_eueh.append(myn_eh)
                        eh_eur2reh.append(eh_r2reh)
                        rh_eur2reh.append(rh_r2reh)

                    # Does this process have anything to receive?
                    thisr2_ct = r2_ct == self.world.rank
                    if np.any(thisr2_ct):
                        # Find unique composite indeces h = (n, s)
                        n_rt = n_ct[thisr2_ct]
                        n_rn = np.unique(n_rt)
                        nrn = len(n_rn)
                        h_rn = np.arange(nrn) + nh
                        nh += nrn

                        # Where to get the data from
                        r1_rt = r1_ct[thisr2_ct]
                        for r1 in np.unique(r1_rt):
                            thisr1_rt = r1_rt == r1
                            # What ns are the process getting?
                            n_reh = np.unique(n_rt[thisr1_rt])
                            # Where to put them
                            for n in n_reh:
                                h = h_rn[np.where(n_rn == n)[0][0]]
                                h_r1rh[r1].append(h)

                                # h to t mapping
                                thisn_myt = n_myt == n
                                thish_myt = np.logical_and(thisn_myt,
                                                           thiss_myt)
                                h_myt[thish_myt] = h

        return (data, myu_eu, myn_eueh, ik_r2, nrh_r2,
                eh_eur2reh, rh_eur2reh, h_r1rh, h_myt, myt_myt)

    def create_get_extraction_info(self):
        """Creator component of the extraction information factory."""
        if self.calc_parallel:
            return self.get_parallel_extraction_info
        else:
            return self.get_serial_extraction_info

    @staticmethod
    def get_serial_extraction_info(u, n_ct, r2_ct):
        """Figure out where to extract the data from in the gs calc"""
        # Let the process extract its own data
        myu = u  # The process has access to all data
        r1_ct = r2_ct
        myn_ct = n_ct

        return myu, r1_ct, myn_ct

    def get_parallel_extraction_info(self, u, n_ct, *unused):
        """Figure out where to extract the data from in the gs calc"""
        wfs = self.calc.wfs
        # Find out where data is in wfs
        k, s = divmod(u, wfs.nspins)
        kptrank, q = wfs.kd.who_has(k)
        myu = q * wfs.nspins + s
        r1_ct, myn_ct = [], []
        for n in n_ct:
            bandrank, myn = wfs.bd.who_has(n)
            # XXX this will fail when using non-standard nesting
            # of communicators.
            r1 = (kptrank * wfs.gd.comm.size * wfs.bd.comm.size
                  + bandrank * wfs.gd.comm.size)
            r1_ct.append(r1)
            myn_ct.append(myn)

        return myu, np.array(r1_ct), np.array(myn_ct)

    @timer('Allocate transfer arrays')
    def allocate_transfer_arrays(self, data, nrh_r2, ik_r2, h_r1rh):
        """Allocate arrays for intermediate storage of data."""
        wfs = self.calc.wfs
        kptex = wfs.kpt_u[0]
        Pshape = kptex.projections.array.shape
        Pdtype = kptex.projections.matrix.dtype
        psitdtype = kptex.psit.array.dtype

        # Number of h-indeces to receive
        nrh_r1 = [len(h_rh) for h_rh in h_r1rh]

        # if self.kptblockcomm.rank in range(len(ik_p)):
        if data[2] is not None:
            ik = data[2]
            ng = self.pd0.ng_q[ik]
            eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG = [], [], [], []
            for nrh in nrh_r1:
                if nrh >= 1:
                    eps_r1rh.append(np.empty(nrh))
                    f_r1rh.append(np.empty(nrh))
                    P_r1rhI.append(np.empty((nrh,) + Pshape[1:], dtype=Pdtype))
                    psit_r1rhG.append(np.empty((nrh, ng), dtype=psitdtype))
                else:
                    eps_r1rh.append(None)
                    f_r1rh.append(None)
                    P_r1rhI.append(None)
                    psit_r1rhG.append(None)
        else:
            eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG = None, None, None, None

        eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG = [], [], [], []
        for nrh, ik in zip(nrh_r2, ik_r2):
            if nrh:
                eps_r2rh.append(np.empty(nrh))
                f_r2rh.append(np.empty(nrh))
                P_r2rhI.append(np.empty((nrh,) + Pshape[1:], dtype=Pdtype))
                ng = self.pd0.ng_q[ik]
                psit_r2rhG.append(np.empty((nrh, ng), dtype=psitdtype))
            else:
                eps_r2rh.append(None)
                f_r2rh.append(None)
                P_r2rhI.append(None)
                psit_r2rhG.append(None)

        return (eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG,
                eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG)

    def map_who_has(self, p, t_t):
        """Convert k-point and transition index to global world rank
        and local transition index"""
        trank_t, myt_t = np.divmod(t_t, self.mynt)
        return p * self.transitionblockscomm.size + trank_t, myt_t

    @timer('Extracting eps, f and P_I from wfs')
    def extract_wfs_data(self, myu, myn_eh):
        wfs = self.calc.wfs
        kpt = wfs.kpt_u[myu]
        # Get eig and occ
        eps_eh, f_eh = kpt.eps_n[myn_eh], kpt.f_n[myn_eh] / kpt.weight

        # Get projections
        assert kpt.projections.atom_partition.comm.size == 1
        P_ehI = kpt.projections.array[myn_eh]

        return eps_eh, f_eh, P_ehI

    @timer('Extracting wave function from wfs')
    def add_wave_function(self, myu, myn_eh,
                          eh_r2reh, rh_r2reh, psit_r2rhG):
        """Add the plane wave coefficients of the smooth part of
        the wave function to the psit_r2rtG arrays."""
        wfs = self.calc.wfs
        kpt = wfs.kpt_u[myu]

        for eh_reh, rh_reh, psit_rhG in zip(eh_r2reh, rh_r2reh, psit_r2rhG):
            if eh_reh:
                for eh, rh in zip(eh_reh, rh_reh):
                    psit_rhG[rh] = kpt.psit_nG[myn_eh[eh]]

    @timer('Distributing kptdata')
    def distribute_extracted_data(self, eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG,
                                  eps_r2rh, f_r2rh, P_r2rhI, psit_r2rhG):
        """Send the extracted data to appropriate destinations"""
        # Store the data extracted by the process itself
        rank = self.world.rank
        # Check if there is actually some data to store
        if eps_r2rh[rank] is not None:
            eps_r1rh[rank] = eps_r2rh[rank]
            f_r1rh[rank] = f_r2rh[rank]
            P_r1rhI[rank] = P_r2rhI[rank]
            psit_r1rhG[rank] = psit_r2rhG[rank]

        # Receive data
        if eps_r1rh is not None:  # The process may not be receiving anything
            for r1, (eps_rh, f_rh,
                     P_rhI, psit_rhG) in enumerate(zip(eps_r1rh, f_r1rh,
                                                       P_r1rhI, psit_r1rhG)):
                # Check if there is any data to receive
                if r1 != rank and eps_rh is not None:
                    rreq1 = self.world.receive(eps_rh, r1,
                                               tag=201, block=False)
                    rreq2 = self.world.receive(f_rh, r1,
                                               tag=202, block=False)
                    rreq3 = self.world.receive(P_rhI, r1,
                                               tag=203, block=False)
                    rreq4 = self.world.receive(psit_rhG, r1,
                                               tag=204, block=False)
                    self.rrequests += [rreq1, rreq2, rreq3, rreq4]

        # Send data
        for r2, (eps_rh, f_rh,
                 P_rhI, psit_rhG) in enumerate(zip(eps_r2rh, f_r2rh,
                                                   P_r2rhI, psit_r2rhG)):
            # Check if there is any data to send
            if r2 != rank and eps_rh is not None:
                sreq1 = self.world.send(eps_rh, r2, tag=201, block=False)
                sreq2 = self.world.send(f_rh, r2, tag=202, block=False)
                sreq3 = self.world.send(P_rhI, r2, tag=203, block=False)
                sreq4 = self.world.send(psit_rhG, r2, tag=204, block=False)
                self.srequests += [sreq1, sreq2, sreq3, sreq4]

        with self.timer('Waiting to complete mpi.receive'):
            while self.rrequests:
                self.world.wait(self.rrequests.pop(0))

    @timer('Collecting kptdata')
    def collect_kptdata(self, data, h_r1rh,
                        eps_r1rh, f_r1rh, P_r1rhI, psit_r1rhG):
        """From the extracted data, collect the KohnShamKPoint data arrays"""

        # Some processes may not have to return a k-point
        if data[0] is None:
            return None
        K, k_c, ik = data

        # Allocate data arrays
        wfs = self.calc.wfs
        maxh_r1 = [max(h_rh) for h_rh in h_r1rh if h_rh]
        if maxh_r1:
            nh = max(maxh_r1) + 1
        else:  # Carry around empty array
            assert self.ta == self.tb
            nh = 1
        eps_h = np.empty(nh)
        f_h = np.empty(nh)
        Ph = wfs.kpt_u[0].projections.new(nbands=nh, bcomm=None)
        psit_hG = np.empty((nh, self.pd0.ng_q[ik]),
                           dtype=wfs.kpt_u[0].psit.array.dtype)

        # Store extracted data in the arrays
        for (h_rh, eps_rh,
             f_rh, P_rhI, psit_rhG) in zip(h_r1rh, eps_r1rh,
                                           f_r1rh, P_r1rhI, psit_r1rhG):
            if h_rh:
                eps_h[h_rh] = eps_rh
                f_h[h_rh] = f_rh
                Ph.array[h_rh] = P_rhI
                psit_hG[h_rh] = psit_rhG

        return (K, k_c, eps_h, f_h, Ph, psit_hG)

    @timer('Unfolding arrays')
    def unfold_arrays(self, eps_h, f_h, Ph, ut_hR, h_myt, myt_myt):
        """Create transition data arrays from the composite h = (n, s) index"""

        wfs = self.calc.wfs
        # Allocate data arrays for the k-point
        mynt = self.mynt
        eps_myt = np.empty(mynt)
        f_myt = np.empty(mynt)
        P = wfs.kpt_u[0].projections.new(nbands=mynt, bcomm=None)
        ut_mytR = wfs.gd.empty(self.mynt, wfs.dtype)

        # Unfold k-point data
        eps_myt[myt_myt] = eps_h[h_myt]
        f_myt[myt_myt] = f_h[h_myt]
        P.array[myt_myt] = Ph.array[h_myt]
        ut_mytR[myt_myt] = ut_hR[h_myt]

        return eps_myt, f_myt, P, ut_mytR

    @timer('Extracting data from the ground state calculator object')
    def serial_extract_kptdata(self, k_pc, n_t, s_t):
        # All processes can access all data. Each process extracts it own data.
        wfs = self.calc.wfs

        # Do data extraction for the processes, which have data to extract
        if self.kptblockcomm.rank in range(len(k_pc)):
            # Find k-point indeces
            k_c = k_pc[self.kptblockcomm.rank]
            K = self.find_kpoint(k_c)
            ik = wfs.kd.bz2ibz_k[K]
            # Construct symmetry operators
            (_, T, a_a, U_aii, shift_c,
             time_reversal) = self.construct_symmetry_operators(K, k_c=k_c)

            (myu_eu, myn_eurn, nh, h_eurn, h_myt,
             myt_myt) = self.get_serial_extraction_protocol(ik, n_t, s_t)

            # Allocate transfer arrays
            eps_h = np.empty(nh)
            f_h = np.empty(nh)
            Ph = wfs.kpt_u[0].projections.new(nbands=nh, bcomm=None)
            ut_hR = wfs.gd.empty(nh, wfs.dtype)

            # Extract data from the ground state
            for myu, myn_rn, h_rn in zip(myu_eu, myn_eurn, h_eurn):
                kpt = wfs.kpt_u[myu]
                with self.timer('Extracting eps, f and P_I from wfs'):
                    eps_h[h_rn] = kpt.eps_n[myn_rn]
                    f_h[h_rn] = kpt.f_n[myn_rn] / kpt.weight
                    Ph.array[h_rn] = kpt.projections.array[myn_rn]

                with self.timer('Extracting, fourier transforming and '
                                'symmetrizing wave function'):
                    for myn, h in zip(myn_rn, h_rn):
                        ut_hR[h] = T(wfs.pd.ifft(kpt.psit_nG[myn], kpt.q))

            # Symmetrize projections
            with self.timer('Apply symmetry operations'):
                P_ahi = []
                for a1, U_ii in zip(a_a, U_aii):
                    P_hi = np.ascontiguousarray(Ph[a1])
                    # Apply symmetry operations. This will map a1 onto a2
                    np.dot(P_hi, U_ii, out=P_hi)
                    if time_reversal:
                        np.conj(P_hi, out=P_hi)
                    P_ahi.append(P_hi)

                # Store symmetrized projectors
                for a2, P_hi in enumerate(P_ahi):
                    I1, I2 = Ph.map[a2]
                    Ph.array[..., I1:I2] = P_hi

            (eps_myt, f_myt,
             P, ut_mytR) = self.unfold_arrays(eps_h, f_h, Ph, ut_hR,
                                              h_myt, myt_myt)

            return (K, eps_myt, f_myt, ut_mytR, P, shift_c)

    @timer('Create data extraction protocol')
    def get_serial_extraction_protocol(self, ik, n_t, s_t):
        """Figure out how to extract data efficiently.
        For the serial communicator, all processes can access all data,
        and resultantly, there is no need to send any data.
        """
        wfs = self.calc.wfs

        # Only extract the transitions handled by the process itself
        myt_myt = np.arange(self.tb - self.ta)
        t_myt = range(self.ta, self.tb)
        n_myt = n_t[t_myt]
        s_myt = s_t[t_myt]

        # In the ground state, kpts are indexed by u=(s, k)
        myu_eu = []
        myn_eurn = []
        nh = 0
        h_eurn = []
        h_myt = np.empty(self.tb - self.ta, dtype=int)
        for s in set(s_myt):
            thiss_myt = s_myt == s
            n_ct = n_myt[thiss_myt]

            # Find unique composite h = (n, u) indeces
            n_rn = np.unique(n_ct)
            nrn = len(n_rn)
            h_eurn.append(np.arange(nrn) + nh)
            nh += nrn

            # Find mapping between h and the transition index
            for n, h in zip(n_rn, h_eurn[-1]):
                thisn_myt = n_myt == n
                thish_myt = np.logical_and(thisn_myt, thiss_myt)
                h_myt[thish_myt] = h

            # Find out where data is in wfs
            u = ik * wfs.nspins + s
            # The process has access to all data
            myu = u
            myn_rn = n_rn

            myu_eu.append(myu)
            myn_eurn.append(myn_rn)

        return myu_eu, myn_eurn, nh, h_eurn, h_myt, myt_myt

    @timer('Identifying k-points')
    def find_kpoint(self, k_c):
        return self.kdtree.query(np.mod(np.mod(k_c, 1).round(6), 1))[1]

    @timer('Apply symmetry operations')
    def transform_and_symmetrize(self, K, k_c, Ph, psit_hG):
        """Get wave function on a real space grid and symmetrize it
        along with the corresponding PAW projections."""
        (_, T, a_a, U_aii, shift_c,
         time_reversal) = self.construct_symmetry_operators(K, k_c=k_c)

        # Symmetrize wave functions
        wfs = self.calc.wfs
        ik = wfs.kd.bz2ibz_k[K]
        ut_hR = wfs.gd.empty(len(psit_hG), wfs.dtype)
        with self.timer('Fourier transform and symmetrize wave functions'):
            for h, psit_G in enumerate(psit_hG):
                ut_hR[h] = T(self.pd0.ifft(psit_G, ik))

        # Symmetrize projections
        P_ahi = []
        for a1, U_ii in zip(a_a, U_aii):
            P_hi = np.ascontiguousarray(Ph[a1])
            # Apply symmetry operations. This will map a1 onto a2
            np.dot(P_hi, U_ii, out=P_hi)
            if time_reversal:
                np.conj(P_hi, out=P_hi)
            P_ahi.append(P_hi)

        # Store symmetrized projectors
        for a2, P_hi in enumerate(P_ahi):
            I1, I2 = Ph.map[a2]
            Ph.array[..., I1:I2] = P_hi

        return Ph, ut_hR, shift_c

    @timer('Construct symmetry operators')
    def construct_symmetry_operators(self, K, k_c=None):
        """Construct symmetry operators for wave function and PAW projections.

        We want to transform a k-point in the irreducible part of the BZ to
        the corresponding k-point with index K.

        Returns U_cc, T, a_a, U_aii, shift_c and time_reversal, where:

        * U_cc is a rotation matrix.
        * T() is a function that transforms the periodic part of the wave
          function.
        * a_a is a list of symmetry related atom indices
        * U_aii is a list of rotation matrices for the PAW projections
        * shift_c is three integers: see code below.
        * time_reversal is a flag - if True, projections should be complex
          conjugated.

        See the extract_orbitals() method for how to use these tuples.
        """

        wfs = self.calc.wfs
        kd = wfs.kd

        s = kd.sym_k[K]
        U_cc = kd.symmetry.op_scc[s]
        time_reversal = kd.time_reversal_k[K]
        ik = kd.bz2ibz_k[K]
        if k_c is None:
            k_c = kd.bzk_kc[K]
        ik_c = kd.ibzk_kc[ik]

        sign = 1 - 2 * time_reversal
        shift_c = np.dot(U_cc, ik_c) - k_c * sign

        try:
            assert np.allclose(shift_c.round(), shift_c)
        except AssertionError:
            print('shift_c ' + str(shift_c), file=self.fd)
            print('k_c ' + str(k_c), file=self.fd)
            print('kd.bzk_kc[K] ' + str(kd.bzk_kc[K]), file=self.fd)
            print('ik_c ' + str(ik_c), file=self.fd)
            print('U_cc ' + str(U_cc), file=self.fd)
            print('sign ' + str(sign), file=self.fd)
            raise AssertionError

        shift_c = shift_c.round().astype(int)

        if (U_cc == np.eye(3)).all():
            def T(f_R):
                return f_R
        else:
            N_c = self.calc.wfs.gd.N_c
            i_cr = np.dot(U_cc.T, np.indices(N_c).reshape((3, -1)))
            i = np.ravel_multi_index(i_cr, N_c, 'wrap')

            def T(f_R):
                return f_R.ravel()[i].reshape(N_c)

        if time_reversal:
            T0 = T

            def T(f_R):
                return T0(f_R).conj()

            shift_c *= -1

        a_a = []
        U_aii = []
        for a, id in enumerate(self.calc.wfs.setups.id_a):
            b = kd.symmetry.a_sa[s, a]
            S_c = np.dot(self.calc.spos_ac[a], U_cc) - self.calc.spos_ac[b]
            x = np.exp(2j * np.pi * np.dot(ik_c, S_c))
            U_ii = wfs.setups[a].R_sii[s].T * x
            a_a.append(b)
            U_aii.append(U_ii)

        shift0_c = (kd.bzk_kc[K] - k_c).round().astype(int)
        shift_c += -shift0_c

        return U_cc, T, a_a, U_aii, shift_c, time_reversal


def get_calc(gs, fd=None, timer=None):
    """Get ground state calculation object."""
    if isinstance(gs, GPAW):
        return gs
    else:
        if timer is None:
            def timer(*unused):
                def __enter__(self):
                    pass

                def __exit__(self):
                    pass

        with timer('Read ground state'):
            assert Path(gs).is_file()
            if fd is not None:
                print('Reading ground state calculation:\n  %s' % gs,
                      file=fd)
            with disable_dry_run():
                return GPAW(gs, txt=None, communicator=mpi.serial_comm)


class PairMatrixElement:
    """Class for calculating matrix elements for transitions in Kohn-Sham
    linear response functions."""
    def __init__(self, kspair):
        """
        Parameters
        ----------
        kslrf : KohnShamLinearResponseFunction instance
        """
        self.calc = kspair.calc
        self.fd = kspair.fd
        self.timer = kspair.timer
        self.transitionblockscomm = kspair.transitionblockscomm

    def initialize(self, *args, **kwargs):
        """Initialize e.g. PAW corrections or other operations
        ahead in time of integration."""
        pass

    def __call__(self, kskptpair, *args, **kwargs):
        """Calculate the matrix element for all transitions in kskptpairs."""
        raise NotImplementedError('Define specific matrix element')


class PlaneWavePairDensity(PairMatrixElement):
    """Class for calculating pair densities:

    n_T(q+G) = <s'n'k'| e^(i (q + G) r) |snk>

    in the plane wave mode"""
    def __init__(self, kspair):
        PairMatrixElement.__init__(self, kspair)

        # Save PAW correction for all calls with same q_c
        self.Q_aGii = None
        self.currentq_c = None

    def initialize(self, pd):
        """Initialize PAW corrections ahead in time of integration."""
        self.initialize_paw_corrections(pd)

    @timer('Initialize PAW corrections')
    def initialize_paw_corrections(self, pd):
        """Initialize PAW corrections, if not done already, for the given q"""
        q_c = pd.kd.bzk_kc[0]
        if self.Q_aGii is None or not np.allclose(q_c - self.currentq_c, 0.):
            self.Q_aGii = self._initialize_paw_corrections(pd)
            self.currentq_c = q_c

    def _initialize_paw_corrections(self, pd):
        wfs = self.calc.wfs
        spos_ac = self.calc.spos_ac
        G_Gv = pd.get_reciprocal_vectors()

        pos_av = np.dot(spos_ac, pd.gd.cell_cv)

        # Collect integrals for all species:
        Q_xGii = {}
        for id, atomdata in wfs.setups.setups.items():
            Q_Gii = two_phi_planewave_integrals(G_Gv, atomdata)
            ni = atomdata.ni
            Q_Gii.shape = (-1, ni, ni)

            Q_xGii[id] = Q_Gii

        Q_aGii = []
        for a, atomdata in enumerate(wfs.setups):
            id = wfs.setups.id_a[a]
            Q_Gii = Q_xGii[id]
            x_G = np.exp(-1j * np.dot(G_Gv, pos_av[a]))
            Q_aGii.append(x_G[:, np.newaxis, np.newaxis] * Q_Gii)

        return Q_aGii

    @timer('Calculate pair density')
    def __call__(self, kskptpair, pd):
        """Calculate the pair densities for all transitions:
        n_t(q+G) = <s'n'k+q| e^(i (q + G) r) |snk>
                 = <snk| e^(-i (q + G) r) |s'n'k+q>
        """
        Q_aGii = self.get_paw_projectors(pd)
        Q_G = self.get_fft_indices(kskptpair, pd)
        mynt, nt, ta, tb = kskptpair.transition_distribution()

        n_mytG = pd.empty(mynt)

        # Calculate smooth part of the pair densities:
        with self.timer('Calculate smooth part'):
            ut1cc_mytR = kskptpair.kpt1.ut_tR.conj()
            n_mytR = ut1cc_mytR * kskptpair.kpt2.ut_tR
            # Unvectorized
            for myt in range(tb - ta):
                n_mytG[myt] = pd.fft(n_mytR[myt], 0, Q_G) * pd.gd.dv

        # Calculate PAW corrections with numpy
        with self.timer('PAW corrections'):
            P1 = kskptpair.kpt1.projections
            P2 = kskptpair.kpt2.projections
            for (Q_Gii, (a1, P1_myti),
                 (a2, P2_myti)) in zip(Q_aGii, P1.items(), P2.items()):
                P1cc_myti = P1_myti[:tb - ta].conj()
                C1_Gimyt = np.tensordot(Q_Gii, P1cc_myti, axes=([1, 1]))
                P2_imyt = P2_myti.T[:, :tb - ta]
                n_mytG[:tb - ta] += np.sum(C1_Gimyt * P2_imyt[np.newaxis,
                                                              :, :], axis=1).T

        # Attach the calculated pair density to the KohnShamKPointPair object
        kskptpair.attach('n_mytG', 'n_tG', n_mytG)

    def get_paw_projectors(self, pd):
        """Make sure PAW correction has been initialized properly
        and return projectors"""
        self.initialize_paw_corrections(pd)
        return self.Q_aGii

    @timer('Get G-vector indices')
    def get_fft_indices(self, kskptpair, pd):
        """Get indices for G-vectors inside cutoff sphere."""
        kpt1 = kskptpair.kpt1
        kpt2 = kskptpair.kpt2
        kd = self.calc.wfs.kd
        q_c = pd.kd.bzk_kc[0]

        N_G = pd.Q_qG[0]

        shift_c = kpt1.shift_c - kpt2.shift_c
        shift_c += (q_c - kd.bzk_kc[kpt2.K]
                    + kd.bzk_kc[kpt1.K]).round().astype(int)
        if shift_c.any():
            n_cG = np.unravel_index(N_G, pd.gd.N_c)
            n_cG = [n_G + shift for n_G, shift in zip(n_cG, shift_c)]
            N_G = np.ravel_multi_index(n_cG, pd.gd.N_c, 'wrap')
        return N_G
