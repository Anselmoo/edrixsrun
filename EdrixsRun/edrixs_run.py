#!/usr/bin/env python
__author__ = 'Anselm W. Hahn'
__credits__ = ['Anselm W. Hahn']
__license__ = "MIT"
__version__ = "0.5"
__email__ = "Anselm.Hahn@gmail.com"
__status__ = "Production"

from os.path import splitext

import numpy as np
from scipy.special import binom

import edrixs


# def math_exp(val):
#    # Take care that math-expression will be included like *, /, +, or -
#    return eval(val[0])
def compl_mat(X, dim=2, dtype=np.complex):
    """
    Creating an empty complex 2D-matrix

    Parameters
    ----------
    X: int
        Number for rows and columns.
    dim: int
        Number of the dimension


    Returns
    -------
    2D-matrix of complex-zeros with the given shape of X
    """
    if dim == 1:
        return np.zeros((X), dtype=dtype)
    elif dim == 2:
        return np.zeros((X, X), dtype=dtype)
    elif dim == 3:
        return np.zeros((X, X, X), dtype=dtype)
    elif dim == 4:
        return np.zeros((X, X, X, X), dtype=dtype)
    else:
        print("Dimension not available!")


def trans_mat(X, dtype=np.complex):
    """
    Creating an empty complex 2D-matrix

    Parameters
    ----------
    X: int-list
        Number for rows and columns depending on the size

    Returns
    -------
    nD-matrix of complex-zeros with the given shape of x and y
    """
    if len(X) == 1:
        return np.zeros((X[0]), dtype=dtype)
    elif len(X) == 2:
        return np.zeros((X[0], X[1]), dtype=dtype)
    elif len(X) == 3:
        return np.zeros((X[0], X[1], X[2]), dtype=dtype)

class TanabeSugano(object):
    def __init__(self, F2_dd=0., F4_dd=0., zeta_d=0., sym=['Oh', 0.,1.,100], d=9, ext_B=[0., 0., 0.], fname='RIXS.input'):
        # 1-10: 3d-transition-metal valence orbitals
        # 11-16: 2p-transition-metal core orbitals
        # Single particle basis: complex shperical Harmonics
        self.ndorb = 10  # 5 Alpha, 5 Beta of the 3d-shell


        # Defining the empty model-Hamiltonian-matrix
        self.emat_i = compl_mat(X=self.ndorb)  # Init-State

        # 4-index Coulomb interaction tensor, parameterized by
        # Slater integrals, which are obtained from Cowan's code
        # Averaged dd Coulomb interaction is set to be zero
        # Defining the 3d-repulsion integrals, which can also come
        # from the Racah-parameters A,B,C
        # Init-State
        self.F0_dd_i = edrixs.get_F0('d', F2_dd, F4_dd)
        self.F2_dd_i = F2_dd
        self.F4_dd_i = F4_dd


        # Atomic spin-orbit coupling and external B-field for XMCD
        # Init-State
        self.zeta_d_i = zeta_d
        self.ext_B = ext_B  # For XMCD

        # Crystal-field-tensor
        self.cryst = compl_mat(X=5)
        self.sym = sym  # Crystal-field symmetry as list ['Sp']
        self.d = d  # Number of d-electrons

        # Two-electron-Components
        self.umat_i = compl_mat(X=self.ndorb, dim=4)

        # Multi-reference basis
        init_states = int(binom(self.ndorb, self.d))
        self.basis_i = trans_mat(X=[init_states])  # Init-State
        self.ncfg_i = len(self.basis_i)  # Number of Inital-State CSF

        # Eigenvalues of the Manybody-Hamiltonian as float
        self.eval_i = compl_mat(X=init_states, dim=1, dtype=np.float64)  # Init-State
        self.evec_i = compl_mat(X=init_states, dtype=np.float64)  # Init-State

        # Input-Parameters from the terminal
        self.fname = fname


    def one_electron_part(self):
        """
        Solving the one- and pseudo-one-electron problems
        1. SOC
        2. Crystalfield

        Parameters
        ----------
        self.zeta_d: float-list
            SOC of the d-shell for init- and final-state

        self.ext_B: float list
            external B-field with B_x, B_y, B_z

        Returns
        -------
        self.emat_i: 2d complex array
            The model-Hamiltonian-matrix of the init-state

        self.emat_n: 2d complex array
            The model-Hamiltonian-matrix of the final-state

        """
        if self.zeta_d_i > 0:
            # Atomic spin-orbit coupling of the inital-state 2p6-3dn and the final-state 2p5-3dn+1
            self.emat_i[0:self.ndorb, 0:self.ndorb] += edrixs.atom_hsoc('d', self.zeta_d_i)  # Init-State

        # Real cubic Harmonics basis will be created for symmetry or real Atomic-Orbital energy for init-state
        # Real cubic Harmonics basis will be transformed to complex shperical Harmonics basis for init-state
        self.cryst[:, :] = edrixs.cb_op(self.cryst, edrixs.tmat_r2c('d'))  # Setting-up the crystal-field tensor for init-state
        self.emat_i[0:self.ndorb:2, 0:self.ndorb:2] += self.cryst
        self.emat_i[1:self.ndorb:2, 1:self.ndorb:2] += self.cryst
        # Real cubic Harmonics basis will be created for symmetry or real Atomic-Orbital energy for init-state

        # external magnetic field for XMCD and RIXS-XMCD
        if np.sum(self.ext_B) != 0.:
            # Setting up the angular-momentum and spin
            # This is copied from solvers.py, probably wrong for final-state
            v_orbl = 2  # for 3d-shell
            lx, ly, lz = edrixs.get_lx(v_orbl, True), edrixs.get_ly(v_orbl, True), edrixs.get_lz(v_orbl, True)
            sx, sy, sz = edrixs.get_sx(v_orbl), edrixs.get_sy(v_orbl), edrixs.get_sz(v_orbl)
            zeeman = self.ext_B[0] * (lx + 2 * sx) + self.ext_B[1] * (ly + 2 * sy) + self.ext_B[2] * (lz + 2 * sz)
            self.emat_i[0:self.ndorb, 0:self.ndorb] += zeeman

    def two_electron_part(self):
        """
        Creating an empty complex 2D-matrix

        Parameters
        ----------
        Slater-Condon-parameters for the electronic repulsion between the 3d-shell and 2p3d-shell:

        self.F0_d: float-list
            for init- and final-state
        self.F2_d: float-list
            for init- and final-state
        self.F4_d: float-list
            for init- and final-state


        self.d: int
            Number of d-electrons

        Returns
        -------
        self.eval_i: 2d float array
            The calculated energies of the initial-state

         self.basis_i: 1d float array
            The Fock basis of the initial-state
        """
        # Part-1
        # Calculate the Coulomb interaction tensor which is parameterized by
        # Slater integrals F and G for the 3d- and 2p3d-shell repulsion
        # Init-State meaing 3dn
        self.umat_i = edrixs.get_umat_slater('d', self.F0_dd_i, self.F2_dd_i, self.F4_dd_i)  # dd

        # Part-2
        # Build Fock basis in its binary form
        # Converting the list into a real numpy-array as designed
        # Ground-State wavefunction
        self.basis_i = np.asarray(edrixs.get_fock_bin_by_N(self.ndorb, self.d))
        # Substracting an e- from the p-shell (2p5) and add it to the d-shell (dn+1)

        # Part-3
        # Build many-body Hamiltonian in Fock basis
        hmat_i = compl_mat(X=self.ncfg_i)
        hmat_i[:, :] += edrixs.two_fermion(self.emat_i, self.basis_i)
        hmat_i[:, :] += edrixs.four_fermion(self.umat_i, self.basis_i)

        # Part-4

        # Do exact-diagonalization to get eigenvalues and eigenvectors
        # With eigh an ascending ordering is introduced, but values are given in real space as float
        self.eval_i, self.evec_i = np.linalg.eigh(hmat_i)

    def run_tanabe(self):
        self.cryst = edrixs.crystalfield_symmetry(sym=['Oh',1.])
        self.one_electron_part()
        self.two_electron_part()
        print(self.eval_i)





class RIXS_XAS_XES_2p3d(object):
    """
    This class includes all routine to run a single 2p3d-RIXS-calculation.
    For running a batch of different 2p3d-RIXS-calculation, the __init__ has to
    be iteratively set-up.
    """

    def __init__(self, F2_dd=[0., 0.], F4_dd=[0., 0.], F2_dp=0., G1_dp=0., G3_dp=0., zeta_d=[0., 0.], zeta_p=0.,
                 sym=[['Oh', 0.], ['Oh', 0.]], d=9, ext_B=[0., 0., 0.], omega=[-10, 20, 1000],
                 eloss=[-10, 20, 1000], pol_type=[('isotropic', 0)],
                 gamma=[0.1, 0.1], temperature=2., thin=45., phi=45., pwfc=False, fname='RIXS.input'):
        # 1-10: 3d-transition-metal valence orbitals
        # 11-16: 2p-transition-metal core orbitals
        # Single particle basis: complex shperical Harmonics
        self.ndorb = 10  # 5 Alpha, 5 Beta of the 3d-shell
        self.nporb = 6  # 3 Alpha, 3 Beta of the 2p-shell
        self.ntot = 16  # Total number of Alpha, Beta electrons

        # Defining the empty model-Hamiltonian-matrix
        self.emat_i = compl_mat(X=self.ntot)  # Init-State
        self.emat_n = compl_mat(X=self.ntot)  # Final-State

        # 4-index Coulomb interaction tensor, parameterized by
        # Slater integrals, which are obtained from Cowan's code
        # Averaged dd Coulomb interaction is set to be zero
        # Defining the 3d-repulsion integrals, which can also come
        # from the Racah-parameters A,B,C
        # Init-State
        self.F0_dd_i = edrixs.get_F0('d', F2_dd[0], F4_dd[0])
        self.F2_dd_i = F2_dd[0]
        self.F4_dd_i = F4_dd[0]
        # Final-State
        self.F0_dd_n = edrixs.get_F0('d', F2_dd[1], F4_dd[1])
        self.F2_dd_n = F2_dd[1]
        self.F4_dd_n = F4_dd[1]

        # Averaged dd Coulomb interaction is set to be zero
        # Defining the 2p3d-repulsion integrals
        self.G1_dp = G1_dp
        self.G3_dp = G3_dp
        self.F0_dp = edrixs.get_F0('dp', G1_dp, G3_dp)
        self.F2_dp = F2_dp

        # Atomic spin-orbit coupling and external B-field for XMCD
        # Init-State
        self.zeta_d_i = zeta_d[0]
        # Final-State
        self.zeta_d_n = zeta_d[1]
        self.zeta_p = zeta_p
        self.ext_B = ext_B  # For XMCD

        # Crystal-field-tensor
        self.cryst = compl_mat(X=5)
        self.sym = sym  # Crystal-field symmetry as list ['Sp']
        self.d = d  # Number of d-electrons

        # Two-electron-Components
        self.umat_i = compl_mat(X=self.ntot, dim=4)
        self.umat_n = compl_mat(X=self.ntot, dim=4)

        # Multi-reference basis
        init_states = int(binom(self.ndorb, self.d))
        final_states = int(binom(self.ndorb, self.d + 1) * 6)
        self.basis_i = trans_mat(X=[init_states])  # Init-State
        self.basis_n = trans_mat(X=[final_states])  # Final-State
        self.ncfg_i = len(self.basis_i)  # Number of Inital-State CSF
        self.ncfg_n = len(self.basis_n)  # Number of Final-State CSF

        # Eigenvalues of the Manybody-Hamiltonian as float
        self.eval_i = compl_mat(X=init_states, dim=1, dtype=np.float64)  # Init-State
        self.evec_i = compl_mat(X=init_states, dtype=np.float64)  # Init-State
        self.eval_n = compl_mat(X=final_states, dim=1, dtype=np.float64)  # Final-State
        self.evec_n = compl_mat(X=final_states, dtype=np.float64)  # Final-State

        # Dipole-moments
        # Build dipolar transition operators -> n=3
        self.dipole = trans_mat([3, self.ntot, self.ntot])
        self.T_abs = trans_mat([3, self.ncfg_n, self.ncfg_i])
        self.T_emi = trans_mat([3, self.ncfg_i, self.ncfg_n])

        # XAS- & XES-section

        self.omega = omega  # Min, Max excitation band, and it's increment
        self.eloss = eloss  # Min, Max emission, and it's increment
        self.xasen = np.linspace(self.omega[0], self.omega[1], self.omega[2])  # Energy-band for XAS
        self.xesen = np.linspace(self.eloss[0], self.eloss[1], self.eloss[2])  # Energy-band for XES

        # Empty-Scattering-Matrices
        self.pol_type = pol_type  # Type of polarization for XMCD
        self.xas = trans_mat([self.omega[2], len(self.pol_type)], dtype=np.float)  # Spc-band for XAS depending on pol
        self.xes = trans_mat([self.eloss[2], len(self.pol_type)], dtype=np.float)  # Spc-band for XES depending on pol
        # self.xas_stk = trans_mat([2,self.ncfg_i, len(self.pol_type)], dtype=np.float)  # Stk-band for XAS depending on pol
        # self.xes_stk = trans_mat([2,self.ncfg_n, len(self.pol_type)], dtype=np.float)  # Stk-band for XES depending on pol
        self.rixs = trans_mat([self.omega[2], self.eloss[2], len(self.pol_type)], dtype=np.float)  # RIXS-plance

        # Core-Hole-Lifetime and parameters
        self.gamma_c = gamma[0]  # Life-time broadening of the intermediate-states
        self.gamma_f = gamma[1]  # Life-time broadening of the final-states
        self.temperature = temperature  # Temperature of the Boltzmann-distribution (default 2K)
        self.thin = np.radians(thin)  # The incident angle of photon (in radian)
        self.phi = np.radians(phi)  # Azimuthal angle (in radian)

        # Input-Parameters from the terminal
        self.print = pwfc
        self.fname = fname

    def crystalfield(self, mode=0):
        """
        Defining the crystal-field-splitting from
        1. sperical symmetry to Oh (-Oh = Tg) to D4h
        2. with real orbital energy in eV

        See reference: Crispy
        http://www.esrf.eu/computing/scientific/crispy/tutorials/ti_l23_xas.html

        Parameters
        ----------
        self.sym: list in list
            contains the crystal-field parameters for init- and final-state
        mode : int
            select init- and final-state
        dt, ds, dq : float
        Crystal field splitting paramater for ds, dt =0 -> Oh

        or dxy, dxz, dyz, dx2y2, dz2 : float
        Real orbital-energy of the 3d-shell


        Returns
        -------
        2D-complex-matrix of crystal-field but only expressed in real-basis meaning has to transform later
        """
        # Tetragonal crystal field splitting terms,
        # which are first defined in the real cubic Harmonics basis
        if self.sym[mode][0] == 'Sp' and len(self.sym[mode]) == 1:
            # Spherical symmetry
            dq, dt, ds = 0., 0., 0.
            self.cryst[0, 0] = - 4 * dq + 2 * ds - 1 * dt  # dxy
            self.cryst[1, 1] = - 4 * dq - 1 * ds + 4 * dt  # dxz
            self.cryst[2, 2] = - 4 * dq - 1 * ds + 4 * dt  # dyz
            self.cryst[3, 3] = + 6 * dq + 2 * ds - 1 * dt  # dx2y2
            self.cryst[4, 4] = + 6 * dq - 2 * ds - 6 * dt  # dz2
            #self.cryst[:, :] = edrixs.cb_op(self.cryst, edrixs.tmat_r2c('d'))
        elif self.sym[mode][0] == 'Oh' and len(self.sym[mode]) == 2:
            # Octahedral symmetry
            dq, dt, ds = self.sym[mode][1] / 10., 0., 0.
            self.cryst[0, 0] = - 4 * dq + 2 * ds - 1 * dt  # dxy
            self.cryst[1, 1] = - 4 * dq - 1 * ds + 4 * dt  # dxz
            self.cryst[2, 2] = - 4 * dq - 1 * ds + 4 * dt  # dyz
            self.cryst[3, 3] = + 6 * dq + 2 * ds - 1 * dt  # dx2y2
            self.cryst[4, 4] = + 6 * dq - 2 * ds - 6 * dt  # dz2
        elif self.sym[mode][0] == 'Td' and len(self.sym[mode]) == 2:
            # Tetrahedal symmetry
            dq, dt, ds = -self.sym[mode][1] / 10., 0., 0.
            self.cryst[0, 0] = - 4 * dq + 2 * ds - 1 * dt  # dxy
            self.cryst[1, 1] = - 4 * dq - 1 * ds + 4 * dt  # dxz
            self.cryst[2, 2] = - 4 * dq - 1 * ds + 4 * dt  # dyz
            self.cryst[3, 3] = + 6 * dq + 2 * ds - 1 * dt  # dx2y2
            self.cryst[4, 4] = + 6 * dq - 2 * ds - 6 * dt  # dz2
        elif self.sym[mode][0] == 'D4h' and len(self.sym[mode]) == 4:
            # D4h symmetry
            # doi:10.1088/1742-6596/190/1/012143 by Frank deGroot
            dq, dt, ds = self.sym[mode][1] / 10., self.sym[mode][2] / 10., self.sym[mode][3] / 10.
            self.cryst[0, 0] = - 4 * dq + 2 * ds - 1 * dt  # dxy
            self.cryst[1, 1] = - 4 * dq - 1 * ds + 4 * dt  # dxz
            self.cryst[2, 2] = - 4 * dq - 1 * ds + 4 * dt  # dyz
            self.cryst[3, 3] = + 6 * dq + 2 * ds - 1 * dt  # dx2y2
            self.cryst[4, 4] = + 6 * dq - 2 * ds - 6 * dt  # dz2
        elif self.sym[mode][0] == 'real' and len(self.sym[mode]) == 6:
            # Real orbital energy, however that has to be validated more carefully
            self.cryst[0, 0] = self.sym[mode][1]  # dxy
            self.cryst[1, 1] = self.sym[mode][2]  # dxz
            self.cryst[2, 2] = self.sym[mode][3]  # dyz
            self.cryst[3, 3] = self.sym[mode][4]  # dx2y2
            self.cryst[4, 4] = self.sym[mode][5]  # dz2
        else:
            print("Error in the definition of the crystal-field!")

    def one_electron_part(self):
        """
        Solving the one- and pseudo-one-electron problems
        1. SOC
        2. Crystalfield

        Parameters
        ----------
        self.zeta_d: float-list
            SOC of the d-shell for init- and final-state
        self.zeta_p: float
            SOC of the p-shell

        self.ext_B: float list
            external B-field with B_x, B_y, B_z

        Returns
        -------
        self.emat_i: 2d complex array
            The model-Hamiltonian-matrix of the init-state

        self.emat_n: 2d complex array
            The model-Hamiltonian-matrix of the final-state

        """

        # Atomic spin-orbit coupling of the inital-state 2p6-3dn and the final-state 2p5-3dn+1
        self.emat_i[0:self.ndorb, 0:self.ndorb] += edrixs.atom_hsoc('d', self.zeta_d_i)  # Init-State
        self.emat_n[0:self.ndorb, 0:self.ndorb] += edrixs.atom_hsoc('d', self.zeta_d_n)  # Final-State
        self.emat_n[self.ndorb:self.ntot, self.ndorb:self.ntot] += edrixs.atom_hsoc('p', self.zeta_p)  # Final-State

        # Real cubic Harmonics basis will be created for symmetry or real Atomic-Orbital energy for init-state
        self.crystalfield(mode=0)  # Set-up the crystal-field tensor for init-state
        # Real cubic Harmonics basis will be transformed to complex shperical Harmonics basis for init-state
        self.cryst[:, :] = edrixs.cb_op(self.cryst, edrixs.tmat_r2c('d'))  # Setting-up the crystal-field tensor for init-state
        self.emat_i[0:self.ndorb:2, 0:self.ndorb:2] += self.cryst
        self.emat_i[1:self.ndorb:2, 1:self.ndorb:2] += self.cryst
        # Real cubic Harmonics basis will be created for symmetry or real Atomic-Orbital energy for init-state
        self.crystalfield(mode=1)  # Set-up the crystal-field tensor for init-state
        # Real cubic Harmonics basis will be transformed to complex shperical Harmonics basis for init-state
        self.cryst[:, :] = edrixs.cb_op(self.cryst, edrixs.tmat_r2c('d'))  # Setting-up the crystal-field tensor for init-state
        self.emat_n[0:self.ndorb:2, 0:self.ndorb:2] += self.cryst
        self.emat_n[1:self.ndorb:2, 1:self.ndorb:2] += self.cryst

        # external magnetic field for XMCD and RIXS-XMCD
        if np.sum(self.ext_B) != 0.:
            # Setting up the angular-momentum and spin
            # This is copied from solvers.py, probably wrong for final-state
            v_orbl = 2  # for 3d-shell
            lx, ly, lz = edrixs.get_lx(v_orbl, True), edrixs.get_ly(v_orbl, True), edrixs.get_lz(v_orbl, True)
            sx, sy, sz = edrixs.get_sx(v_orbl), edrixs.get_sy(v_orbl), edrixs.get_sz(v_orbl)
            zeeman = self.ext_B[0] * (lx + 2 * sx) + self.ext_B[1] * (ly + 2 * sy) + self.ext_B[2] * (lz + 2 * sz)
            self.emat_i[0:self.ndorb, 0:self.ndorb] += zeeman
            self.emat_n[0:self.ndorb, 0:self.ndorb] += zeeman

    def two_electron_part(self):
        """
        Creating an empty complex 2D-matrix

        Parameters
        ----------
        Slater-Condon-parameters for the electronic repulsion between the 3d-shell and 2p3d-shell:

        self.F0_d: float-list
            for init- and final-state
        self.F2_d: float-list
            for init- and final-state
        self.F4_d: float-list
            for init- and final-state

        self.F0_dp: float
        self.F2_dp: float
        self.G1_dp: float
        self.G3_dp: float

        self.d: int
            Number of d-electrons

        Returns
        -------
        self.eval_i: 2d float array
            The calculated energies of the initial-state
         self.eval_n: 2d float array
            The calculated energies of the final-state

         self.basis_i: 1d float array
            The Fock basis of the initial-state
         self.eval_n: 1d float array
            The Fock basis of the final-state
        """
        # Part-1
        # Calculate the Coulomb interaction tensor which is parameterized by
        # Slater integrals F and G for the 3d- and 2p3d-shell repulsion
        # Init-State meaing 2p6-3dn
        self.umat_i = edrixs.get_umat_slater('dp', self.F0_dd_i, self.F2_dd_i, self.F4_dd_i,  # dd
                                             0, 0, 0, 0,  # dp
                                             0, 0)  # pp
        # Final-State meaing 2p5-3dn+1
        self.umat_n = edrixs.get_umat_slater('dp', self.F0_dd_n, self.F2_dd_n, self.F4_dd_n,  # dd
                                             self.F0_dp, self.F2_dp, self.G1_dp, self.G3_dp,  # dp
                                             0, 0)  # pp

        # Part-2
        # Build Fock basis in its binary form
        # Converting the list into a real numpy-array as designed
        # Ground-State wavefunction
        self.basis_i = np.asarray(edrixs.get_fock_bin_by_N(self.ndorb, self.d, self.nporb, self.nporb))
        # Substracting an e- from the p-shell (2p5) and add it to the d-shell (dn+1)
        self.basis_n = np.asarray(edrixs.get_fock_bin_by_N(self.ndorb, self.d + 1, self.nporb, self.nporb - 1))

        # Part-3
        # Build many-body Hamiltonian in Fock basis
        hmat_i = compl_mat(X=self.ncfg_i)
        hmat_n = compl_mat(X=self.ncfg_n)
        hmat_i[:, :] += edrixs.two_fermion(self.emat_i, self.basis_i)
        hmat_i[:, :] += edrixs.four_fermion(self.umat_i, self.basis_i)
        hmat_n[:, :] += edrixs.two_fermion(self.emat_n, self.basis_n)
        hmat_n[:, :] += edrixs.four_fermion(self.umat_n, self.basis_n)

        # Part-4

        if self.print:
            # Non-ascending exact-diagonalization to get eigenvalues and eigenvectors
            self.eval_i, self.evec_i = np.linalg.eig(hmat_i)
            self.eval_n, self.evec_n = np.linalg.eig(hmat_n)

            # Index sorting from smallest to highest eigen-values
            e_sort_i = np.argsort(self.eval_i)
            e_sort_n = np.argsort(self.eval_n)

            # Rearranging the wavefunction
            self.basis_i = self.basis_i[e_sort_i]
            self.basis_n = self.basis_n[e_sort_n]

            # Rearranging the eigen-values
            self.eval_i = self.eval_i[e_sort_i]
            self.eval_n = self.eval_n[e_sort_n]
            # Rearranging the eigen-vectors
            self.evec_i = self.evec_i[e_sort_i, :]
            self.evec_n = self.evec_n[e_sort_n, :]
            self.evec_i = self.evec_i[:, e_sort_i]
            self.evec_n = self.evec_n[:, e_sort_n]

            spin_i, lang_i, ls_i = edrixs.get_s_l_ls_values_i(l=2, basis=self.basis_i, evec=self.evec_i)
            spin_x, lang_x, ls_x = edrixs.get_s_l_ls_values_i(l=2, basis=self.basis_n, evec=self.evec_n)
            spin_n, lang_n, ls_n = edrixs.get_s_l_ls_values_n(l=2, basis=self.basis_n, evec=self.evec_n,
                                                              ref=self.emat_n, ndorb=self.ndorb, ntot=self.ntot)

            # Removing the empty imaginary-part of the energies for the Green-Function-Ansatz later
            self.eval_i, self.eval_n = self.eval_i.real, self.eval_n.real
            # For the correct printing the wavefunction has to be sorted
            print("edrixs >>> Ground-State Wavefunction ...\n")
            edrixs.print_wavefunction(eng=self.eval_i, wfc=self.basis_i, spin=spin_i, lang=lang_i, LS=ls_i,
                                      fname=self.fname + '_GS_Wavefunction.out')
            print("\nedrixs >>> Ground-State Wavefunction Done!")
            print("edrixs >>> Excited-State Wavefunction without 2p-coupling ...\n")
            edrixs.print_wavefunction(eng=self.eval_n, wfc=self.basis_n, spin=spin_x, lang=lang_x, LS=ls_x,
                                      fname=self.fname + '_EC_Wavefunction_no2p.out')
            print("\nedrixs >>> Excited-State Wavefunction without 2p-coupling Done!")
            print("edrixs >>> Excited-State Wavefunction with 2p-coupling ...\n")
            edrixs.print_wavefunction(eng=self.eval_n, wfc=self.basis_n, spin=spin_n, lang=lang_n, LS=ls_n,
                                      fname=self.fname + '_EC_Wavefunction_w2p.out')
            print("\nedrixs >>> Excited-State Wavefunction with 2p-coupling Done!")
        else:
            # Do exact-diagonalization to get eigenvalues and eigenvectors
            # With eigh an ascending ordering is introduced, but values are given in real space as float
            self.eval_i, self.evec_i = np.linalg.eigh(hmat_i)
            self.eval_n, self.evec_n = np.linalg.eigh(hmat_n)

    def dipole_moments(self):
        """
        Calculate the dipole-moments for XAS and XES for a given Fock basis

        Parameters
        ----------
         self.basis_i: 1d float array
            The Fock basis of the initial-state
         self.eval_n: 1d float array
            The Fock basis of the final-state

        self.evec_i: 2D-complex array
            Eigenvectors of the inital-states
        self.evec_n: 2D-complex array
            Eigenvectors of the final-states

        Returns
        -------
        self.T_abs: 3d-complex-array

        self.T_emi

        """

        tmp = edrixs.get_trans_oper('dp')
        for i in range(3):
            self.dipole[i, 0:self.ndorb, self.ndorb:self.ntot] = tmp[i]
            # First, in the Fock basis
            self.T_abs[i] = edrixs.two_fermion(self.dipole[i], self.basis_n, self.basis_i)
            # Then, transfrom to the eigenvector basis
            self.T_abs[i] = edrixs.cb_op2(self.T_abs[i], self.evec_n, self.evec_i)
            # The complext-conjugated will be built the emission
            self.T_emi[i] = np.conj(np.transpose(self.T_abs[i]))

    def xas_spectra(self):
        """
        Calculate XAS for the case of one valence shell plus one core shell with Python solve
        This solver is only suitable for small size of Hamiltonian, typically the dimension
        of both initial and intermediate Hamiltonian are smaller than 10,00
        Parameters
        ----------
        self.eval_i: 1d float array
            The eigenvalues of the initial Hamiltonian.
        self.eval_n: 1d float array
            The eigenvalues of the intermediate Hamiltonian.
        self.T_abs: 3d complex array
            The transition operators in the eigenstates basis.
        self.xasen: 1d float array
            Incident energy of photon.
        self.gamma_c: a float number or a 1d float array with the same shape as ominc.
            The core-hole life-time broadening factor. It can be a constant value
            or incident energy dependent.
        self.thin: float number
            The incident angle of photon (in radian).
        self.phi: float number
            Azimuthal angle (in radian), defined with respect to the
            :math:`x`-axis of the scattering axis: scatter_axis[:,0].
        self.pol_type: list of tuples
            Type of polarization, options can b
            - ('linear', alpha), linear polarization, where alpha is the angle between the
              polarization vector and the scattering plan

            - ('left', 0), left circular polarizatio

            - ('right', 0), right circular polarization

            - ('isotropic', 0). isotropic polarization

            It will set pol_type=[('isotropic', 0)] if not provided.

        self.gs_list: 1d list of ints
            The indices of initial states which will be used in XAS calculations.

            It will set gs_list=[0] if not provided.
        self.temperature: float number
            Temperature (in K) for boltzmann distribution.

        Returns
        -------
        self.xas: 2d float array
            The calculated XAS spectra. The 1st dimension is for the incident energy, and the
            2nd dimension is for different polarizations.
        """
        n_om = self.omega[2]
        npol = 3  # because of pure dipole-interaction
        if self.pol_type is None:
            self.pol_type = [('isotropic', 0)]
        gs_list = list(range(0, 3))  # Three lowest eigenstates are used
        scatter_axis = np.eye(3)  # Scattering along the x-,y-,z-axis

        gamma_core = compl_mat(X=n_om, dim=1, dtype=np.float)
        prob = edrixs.boltz_dist([self.eval_i[i] for i in gs_list], self.temperature)

        if np.isscalar(self.gamma_c):
            gamma_core[:] = np.ones(n_om) * self.gamma_c
        else:
            if len(self.gamma_c) < n_om:
                gamma_core[:] = np.ones(n_om) * self.gamma_c[0]
            else:
                gamma_core[:] = self.gamma_c

        for i, om in enumerate(self.xasen):
            for it, (pt, alpha) in enumerate(self.pol_type):
                if pt.strip() not in ['left', 'right', 'linear', 'isotropic']:
                    raise Exception("Unknown polarization type: ", pt)
                polvec = np.zeros(npol, dtype=np.complex)
                if pt.strip() == 'left' or pt.strip() == 'right' or pt.strip() == 'linear':
                    polvec[:] = edrixs.dipole_polvec_xas(self.thin, self.phi, alpha, scatter_axis, pt)

                # loop over all the initial states
                for j, igs in enumerate(gs_list):
                    if pt.strip() == 'isotropic':
                        for k in range(npol):
                            self.xas[i, it] += (
                                    prob[j] * np.sum(np.abs(self.T_abs[k, :, igs]) ** 2 * gamma_core[i] /
                                                     np.pi / ((om - (self.eval_n[:] - self.eval_i[igs])) ** 2 +
                                                              gamma_core[i] ** 2))
                            )
                        self.xas[i, it] /= npol
                    else:
                        F_mag = np.zeros(self.ncfg_n, dtype=np.complex)
                        for k in range(npol):
                            F_mag += self.T_abs[k, :, igs] * polvec[k]
                        self.xas[i, it] += (
                                prob[j] * np.sum(np.abs(F_mag) ** 2 * gamma_core[i] / np.pi /
                                                 ((om - (self.eval_n[:] - self.eval_i[igs])) ** 2 +
                                                  gamma_core[i] ** 2))
                        )

    def xes_spectra(self):
        """
        Calculate XES for the case of one valence shell plus one core shell with Python solver.
        This solver is only suitable for small size of Hamiltonian, typically the dimension
        of both initial and intermediate Hamiltonian are smaller than 10,00
        In contrast to the XAS, the three highest indices of the eigenvalues have to be choosen
        for the correct XES-spectra.

        Parameters
        ----------
        self.eval_i: 1d float array
            The eigenvalues of the initial Hamiltonian.
        self.eval_n: 1d float array
            The eigenvalues of the intermediate Hamiltonian.
        self.T_abs: 3d complex array
            The transition operators in the eigenstates basis.
        self.xasen: 1d float array
            Incident energy of photon.
        self.gamma_c: a float number or a 1d float array with the same shape as ominc.
            The core-hole life-time broadening factor. It can be a constant value
            or incident energy dependent.
        self.thin: float number
            The incident angle of photon (in radian).
        self.phi: float number
            Azimuthal angle (in radian), defined with respect to the
            :math:`x`-axis of the scattering axis: scatter_axis[:,0].
        self.pol_type: list of tuples
            Type of polarization, options can be:

            - ('linear', alpha), linear polarization, where alpha is the angle between the polarization vector and the scattering plane.

            - ('left', 0), left circular polarization.

            - ('right', 0), right circular polarization.

            - ('isotropic', 0). isotropic polarization.

            It will set pol_type=[('isotropic', 0)] if not provided.
        self.gs_list: 1d list of ints
            The indices of initial states which will be used in XAS calculations.

            It will set gs_list=[0] if not provided.
        self.temperature: float number
            Temperature (in K) for boltzmann distribution.

        Returns
        -------
        xes: 2d float array
            The calculated XAS spectra. The 1st dimension is for the incident energy, and the
            2nd dimension is for different polarizations.
        """
        n_om = self.eloss[2]
        npol = 3  # because of pure dipole-interaction
        if self.pol_type is None:
            self.pol_type = [('isotropic', 0)]

        max_eig = self.T_emi.shape[1]
        gs_list = list(range(max_eig - 3, max_eig))  # Three highest eigenstates are used
        scatter_axis = np.eye(3)  # Scattering along the x-,y-,z-axis

        gamma_core = compl_mat(X=n_om, dim=1, dtype=np.float)
        prob = edrixs.boltz_dist([self.eval_i[i] for i in gs_list], self.temperature)

        if np.isscalar(self.gamma_f):
            gamma_core[:] = np.ones(n_om) * self.gamma_f
        else:
            if len(self.gamma_f) < n_om:
                gamma_core[:] = np.ones(n_om) * self.gamma_f[0]
            else:
                gamma_core[:] = self.gamma_f

        for i, om in enumerate(self.xesen):
            for it, (pt, alpha) in enumerate(self.pol_type):
                if pt.strip() not in ['left', 'right', 'linear', 'isotropic']:
                    raise Exception("Unknown polarization type: ", pt)
                polvec = np.zeros(npol, dtype=np.complex)
                if pt.strip() == 'left' or pt.strip() == 'right' or pt.strip() == 'linear':
                    polvec[:] = edrixs.dipole_polvec_xas(self.thin, self.phi, alpha, scatter_axis, pt)

                # loop over all the initial states
                for j, igs in enumerate(gs_list):
                    if pt.strip() == 'isotropic':
                        for k in range(npol):
                            self.xes[i, it] += (
                                    prob[j] * np.sum(np.abs(self.T_emi[k, igs, :]) ** 2 * gamma_core[i] /
                                                     np.pi / ((om - (-self.eval_n[:] + self.eval_n[igs])) ** 2 +
                                                              gamma_core[i] ** 2))
                            )
                    else:
                        F_mag = np.zeros(self.ncfg_n, dtype=np.complex)
                        for k in range(npol):
                            F_mag += self.T_emi[k, igs, :] * polvec[k]
                        self.xes[i, it] += (
                                prob[j] * np.sum(np.abs(F_mag) ** 2 * gamma_core[i] / np.pi /
                                                 ((om - (-self.eval_n[:] + self.eval_n[igs])) ** 2 +
                                                  gamma_core[i] ** 2))
                        )

    def rixs_spectra(self):
        """
        Calculate RIXS for the case of one valence shell plus one core shell with Python solver.

        This solver is only suitable for small size of Hamiltonian, typically the dimension
        of both initial and intermediate Hamiltonian are smaller than 10,000.

        Parameters
        ----------
        self.eval_i: 1d float array
            The eigenvalues of the initial Hamiltonian.
        self.eval_n: 1d float array
           The eigenvalues of the intermediate Hamiltonian.
        self.trans_op: 3d complex array
           The transition operators in the eigenstates basis.
        self.ominc: 1d float array
            Incident energy of photon.
        self.eloss: 1d float array
            Energy loss.
        self.gamma_c: a float number or a 1d float array with same shape as ominc.
            The core-hole life-time broadening factor. It can be a constant value
            or incident energy dependent.
        self.gamma_f: a float number or a 1d float array with same shape as eloss.
            The final states life-time broadening factor. It can be a constant value
            or energy loss dependent.
        self.thin: float number
            The incident angle of photon (in radian).
        self.thout: float number
            The scattered angle of photon (in radian).
        self.phi: float number
            Azimuthal angle (in radian), defined with respect to the
            :math:`x`-axis of scattering axis: scatter_axis[:,0].
        self.pol_type: list of 4-elements-tuples
            Type of polarizations. It has the following form:

            (str1, alpha, str2, beta)

            where, str1 (str2) can be 'linear', 'left', 'right', and alpha (beta) is
            the angle (in radian) between the linear polarization vector and the scattering plane.

            It will set pol_type=[('linear', 0, 'linear', 0)] if not provided.
        self.gs_list: 1d list of ints
            The indices of initial states which will be used in RIXS calculations.

            It will set gs_list=[0] if not provided.
        self.temperature: float number
            Temperature (in K) for boltzmann distribution.
        self.scatter_axis: 3*3 float array
            The local axis defining the scattering plane. The scattering plane is defined in
            the local :math:`zx`-plane.

            - local :math:`x`-axis: scatter_axis[:,0]

            - local :math:`y`-axis: scatter_axis[:,1]

            - local :math:`z`-axis: scatter_axis[:,2]

            It will be an identity matrix if not provided.

        Returns
        -------
        self.rixs: 3d float array
            The calculated RIXS spectra. The 1st dimension is for the incident energy,
            the 2nd dimension is for the energy loss and the 3rd dimension is for
            different polarizations.
        """
        n_ominc = self.omega[2]
        n_eloss = self.eloss[2]
        gamma_core = compl_mat(X=n_ominc, dim=1, dtype=np.float)
        gamma_final = compl_mat(X=n_eloss, dim=1, dtype=np.float)
        if np.isscalar(self.gamma_c):
            gamma_core[:] = np.ones(n_ominc) * self.gamma_c
        else:
            if len(self.gamma_c) < n_ominc:
                gamma_core[:] = np.ones(n_ominc) * self.gamma_c[0]
            else:
                gamma_core[:] = self.gamma_c
        if np.isscalar(self.gamma_f):
            gamma_final[:] = np.ones(n_eloss) * self.gamma_f
        else:
            if len(self.gamma_f) < n_eloss:
                gamma_final[:] = np.ones(n_eloss) * self.gamma_f[0]
            else:
                gamma_final[:] = self.gamma_f

        if self.pol_type is None:
            self.pol_type = [('linear', 0, 'linear', 0)]

        gs_list = list(range(3))  # Three lowest eigenstates are used
        scatter_axis = np.eye(3)  # Scattering along the x-,y-,z-axis

        prob = edrixs.boltz_dist([self.eval_i[i] for i in gs_list], self.temperature)

        npol, n, m = self.T_emi.shape
        trans_emi = np.zeros((npol, m, n), dtype=np.complex128)
        for i in range(npol):
            trans_emi[i] = np.conj(np.transpose(self.T_emi[i]))
        polvec_i = compl_mat(X=3, dim=1)
        polvec_f = compl_mat(X=3, dim=1)

        # Calculate RIXS
        for i, om in enumerate(self.xasen):
            F_fi = edrixs.scattering_mat(self.eval_i, self.eval_n, self.T_abs[:, :, 0:max(gs_list) + 1],
                                         self.T_emi, om, gamma_core[i])
            # print(F_fi)
            for j, (it, alpha, jt, beta) in enumerate(self.pol_type):
                # dipolar transition
                polvec_i[:], polvec_f[:] = edrixs.dipole_polvec_rixs(thin=self.thin, thout=-self.thin,
                                                                     phi=self.phi, alpha=alpha, beta=beta,
                                                                     local_axis=scatter_axis, pol_type=(it, jt))
                # scattering magnitude with polarization vectors
                F_mag = np.zeros((len(self.eval_i), len(gs_list)), dtype=np.complex)
                for m in range(npol):
                    for n in range(npol):
                        F_mag[:, :] += np.conj(polvec_f[m]) * F_fi[m, n] * polvec_i[n]

                for m, igs in enumerate(gs_list):
                    for n in range(len(self.eval_i)):
                        self.rixs[i, :, j] += (
                                prob[m] * np.abs(F_mag[n, igs]) ** 2 * gamma_final / np.pi /
                                ((self.xesen - (self.eval_i[n] - self.eval_i[igs])) ** 2 + gamma_final ** 2)
                        )

    def run_rixs(self):
        # Pseudo run function, has to be adapted for real cases
        print("edrixs >>> Running One-Electron-Part ...")
        self.one_electron_part()
        print("edrixs >>> One-Electron-Part Done!")
        print("edrixs >>> Running Two-Electron-Part ...")
        self.two_electron_part()
        print("edrixs >>> Two-Electron-Part Done!")
        print("edrixs >>> Generating the Dipole-Moments ...")
        self.dipole_moments()
        print("edrixs >>> Dipole-moments Done!")


class Read_Run_Plot(RIXS_XAS_XES_2p3d,TanabeSugano):
    def __init__(self):
        super().__init__()
        self.inputs = {}
        self.args = object.__new__
        self.mode = []
        self.shift = [0., 0.]
        self.xas_pol_type = []
        self.xes_pol_type = []
        self.rixs_pol_type = []
        self.gamma = [0., 0.]

    def input_read(self):
        """
        Input-file reader for single calculation, has to be modified for interval-style
        :return:
        """
        self.inputs, self.args = edrixs.arg_input()  # Getting the dict from the ascii-file

        # print(self.inputs.keys())
        print("\nedrixs >>> Reading Input ...")
        try:
            for key, vals in self.inputs.items():
                if key.lower() == 'electrons':
                    # Number of d-electrons
                    self.d = int(vals[0])
                    print("Number of electrons >>> {}\n".format(self.d))
                if key.lower() == 'symmetry':
                    """
                    This evaluator is incomplete because it's only optimized for Oh
                    """

                    # Point-group-symmetry and crystal-field
                    gs = eval(vals[0])
                    it = eval(vals[1])

                    if gs == 'Sp' and it == 'Sp':
                        self.sym = [[gs], [it]]
                        print("The crystal-field >>> {}-symmetry\n".format(
                            self.sym[0][0]))
                    elif gs[0] == 'Oh' and it[0] == 'Oh':
                        self.sym = [[gs[0], float(gs[1])], [it[0], float(it[1])]]
                        print("The crystal-field >>> {}-symmetry\n"
                              "\twith Dq-{} eV for ground-state\n"
                              "\twith Dq-{} eV for excited-state\n".format(
                            self.sym[0][0], self.sym[0][1], self.sym[1][1])
                        )
                    elif gs[0] == 'Td' and it[0] == 'Td':
                        self.sym = [[gs[0], float(gs[1])], [it[0], float(it[1])]]
                        print("The crystal-field >>> {}-symmetry\n"
                              "\twith D-{} eV for ground-state\n"
                              "\twith D-{} eV for excited-state\n".format(
                            self.sym[0][0], self.sym[0][1], self.sym[1][1])
                        )
                    elif gs[0] == 'D4h' and it[0] == 'D4h':
                        self.sym = [[gs[0], float(gs[1]), float(gs[2]), float(gs[3])],
                                    [it[0], float(it[1]), float(it[2]), float(it[3])]]
                        print("The crystal-field >>> {}-symmetry\n"
                              "\twith Dq-{}, Dt-{}, and Ds-{} eV for ground-state\n"
                              "\twith Dq-{}, Dt-{}, and Ds-{} eV for excited-state\n".format(
                            self.sym[0][0], self.sym[0][1], self.sym[0][2], self.sym[0][3],
                            self.sym[1][1], self.sym[1][2], self.sym[1][3])
                        )
                    elif gs[0] == 'real' and it[0] == 'real':
                        self.sym = [[gs[0], float(gs[1]), float(gs[2]), float(gs[3]), float(gs[4]), float(gs[5])],
                                    [it[0], float(it[1]), float(it[2]), float(it[3]), float(it[4]), float(it[5])]]
                        print("The crystal-field >>> {}-symmetry\n"
                              "\twith dxy-{}, dxz-{}, dyz-{}, dx2y2-{}, and dz2={} eV for ground-state\n"
                              "\twith dxy-{}, dxz-{}, dyz-{}, dx2y2-{}, and dz2={} eV for excited-state\n".format(
                            self.sym[0][0], self.sym[0][1], self.sym[0][2], self.sym[0][3],self.sym[0][4],self.sym[0][5],
                            self.sym[1][1], self.sym[1][2], self.sym[1][3],self.sym[1][4],self.sym[1][5]))
                    else:
                        print("Wrong Definition of the Crystal-Field! ")


                if key.lower() == 'f2_dd':
                    # Slater-Condon
                    self.F2_dd_i, self.F2_dd_n = eval(vals[0]), eval(vals[1])
                    print("The Slater-Condon-Parameter of "
                          "F2_dd\n"
                          "\tinit-state >>> {:2.5f} eV\n"
                          "\texcited-state >>> {:2.5f} eV\n".format(self.F2_dd_i, self.F2_dd_n))
                if key.lower() == 'f4_dd':
                    # Slater-Condon
                    self.F4_dd_i, self.F4_dd_n = eval(vals[0]), eval(vals[1])
                    print("The Slater-Condon-Parameter of "
                          "F4_dd\n"
                          "\tinit-state >>> {:2.5f} eV\n"
                          "\texcited-state >>> {:2.5f} eV\n".format(self.F4_dd_i, self.F4_dd_n))
                if key.lower() == 'f2_dp':
                    # Slater-Condon
                    self.F2_dp = eval(vals[0])
                    print("The Slater-Condon-Parameter of "
                          "F2_dp >>> {:2.5f} eV\n".format(self.F2_dp))
                if key.lower() == 'g3_dp':
                    # Slater-Condon
                    self.G3_dp = eval(vals[0])
                    print("The Slater-Condon-Parameter of "
                          "G3_dp >>> {:2.5f} eV\n".format(self.G3_dp))
                if key.lower() == 'g1_dp':
                    # Slater-Condon
                    self.G1_dp = eval(vals[0])
                    print("The Slater-Condon-Parameter of "
                          "G1_dp >>> {:2.5f} eV\n".format(self.G1_dp))
                if key.lower() == 'soc_d':
                    # SOC
                    self.zeta_d_i, self.zeta_d_n = eval(vals[0]), eval(vals[1])
                    print("The nd-SOC-Parameter\n"
                          "\tinit-state >>> {:2.5f} eV\n"
                          "\texcited-state >>> {:2.5f} eV\n".format(self.zeta_d_i, self.zeta_d_n))
                if key.lower() == 'soc_p':
                    # SOC
                    self.zeta_p = eval(vals[0])
                    print("The mp-SOC-Parameter >>> {:2.5f} eV\n".format(self.zeta_p))
                if key.lower() == 'b':
                    # B-field for XMCD and RIXSXMCD
                    self.ext_B = [eval(vals[0]), eval(vals[1]), eval(vals[2])]
                    print("The magnetic B-field\n"
                          "\tx-axis >>> {:2.3f} T\n"
                          "\ty-axis >>> {:2.3f} T\n"
                          "\tz-axis >>> {:2.3f} T\n".format(self.ext_B[0], self.ext_B[1], self.ext_B[2]))
                if key.lower() == 'k':
                    # Temperature for Boltzmann-distribution
                    self.temperature = eval(vals[0])
                    print("Temperature >>> {:4.2f} K\n".format(self.temperature))
                if key.lower() == 'abs':
                    """
                    Parameter for the Absorption(XAS)-Plot
                    First three inputs for the energy-band
                    x0, x1, points in between, plus energy-shift
                    """
                    self.omega = [eval(vals[0]), eval(vals[1]), int(eval(vals[2]))]
                    self.shift[0] = eval(vals[3])
                    print("The Absorption-Band\n"
                          "\tmin(abs) >>> {:4.3f} eV\n"
                          "\tmax(abs) >>> {:4.3f} eV\n"
                          "\tsteps >>> {} \n"
                          "\tAbsorption-Shift >>> {:4.3f} eV\n".format(np.min(self.xasen), np.max(self.xasen),
                                                                       self.xasen.size, self.shift[0]))
                if key.lower() == 'ems':
                    """
                    Parameter for the Emission(XES)-Plot
                    First three inputs for the energy-band
                    x0, x1, points in between, plus energy-shift
                    """
                    self.eloss = [eval(vals[0]), eval(vals[1]), int(eval(vals[2]))]
                    self.shift[1] = eval(vals[3])
                    print("The Emission-Band\n"
                          "\tmin(ems) >>> {:4.3f} eV\n"
                          "\tmax(ems) >>> {:4.3f} eV\n"
                          "\tsteps >>> {} \n"
                          "\tEmission-Shift >>> {:4.3f} eV\n".format(np.min(self.xesen), np.max(self.xesen),
                                                                     self.xesen.size, self.shift[1]))
                if key.lower() == 'fhmw':
                    # Defining the FHMW-broadering
                    self.gamma = [eval(vals[0]), eval(vals[1])]
                    print("The FHMW-Broadering\n"
                          "\tXAS-Broadering >>> {:2.3f} eV\n"
                          "\tXES-Broadering >>> {:2.3f} eV\n".format(self.gamma_c, self.gamma_f))
                if key.lower() == 'alpha':
                    self.thin = np.radians(eval(vals[0]))
                    print("The incoming-angle "
                          "Alpha >>> {:3.2f}\n".format(np.degrees(self.thin)))
                if key.lower() == 'phi':
                    self.phi = np.radians(eval(vals[0]))
                    print("The outgoing-angle "
                          "Phi >>> {:3.2f}\n".format(np.degrees(self.phi)))
                if key.lower() == 'xas' and bool(vals[0]):
                    self.mode.append(key.lower())
                    print('XAS-modul is >>> on')
                    if vals[1:] != []:
                        print('\t!Polarization-effect are turned on!')
                        for key in vals[1:]:
                            self.xas_pol_type.append(eval(key))
                        print('\t', self.xas_pol_type)
                    else:
                        self.xas_pol_type.append(('isotropic', 0))
                        print('\t', self.xas_pol_type)
                if key.lower() == 'xes' and bool(vals[0]):
                    self.mode.append(key.lower())
                    print('XES-modul is >>> on')
                    if vals[1:] != []:
                        print('\t!Polarization-effect are turned on!')
                        for key in vals[1:]:
                            self.xes_pol_type.append(eval(key))
                        print('\t', self.xes_pol_type)
                    else:
                        self.xes_pol_type.append(('isotropic', 0))
                        print('\t', self.xes_pol_type)
                if key.lower() == 'rixs' and bool(vals[0]):
                    self.mode.append(key.lower())
                    print('RIXS-modul is >>> on')
                    if vals[1:] != []:
                        print('\t!Polarization-effect are turned on!')
                        for key in vals[1:]:
                            self.rixs_pol_type.append(eval(key))
                        print('\t', self.rixs_pol_type)
                    else:
                        self.rixs_pol_type.append(('linear', 0, 'linear', 0))
                        print('\t', self.rixs_pol_type)
        except SyntaxError:
            print("Ups! Your input-file has a Syntax-Error ... \n"
                  "Please, check carfully your input,\notherwise results"
                  "might be wrong!")

        print("\nedrixs >>> Input Done!")
        print("\n\n")

        fname = splitext(self.args.fname)[0]
        if 'rixs' in self.mode or 'xas' in self.mode or 'xes' in self.mode:
            print("edrixs >>> Setting-Up the Variables ...")
            classmethod(RIXS_XAS_XES_2p3d.__init__(self, F2_dd=[self.F2_dd_i, self.F2_dd_n],
                                                   F4_dd=[self.F4_dd_i, self.F4_dd_n], F2_dp=self.F2_dp,
                                                   G1_dp=self.G1_dp, G3_dp=self.G3_dp, sym=self.sym, d=self.d,
                                                   zeta_d=[self.zeta_d_i, self.zeta_d_n], zeta_p=self.zeta_p,
                                                   omega=self.omega, eloss=self.eloss, gamma=self.gamma,
                                                   temperature=self.temperature, thin=self.thin, phi=self.phi,
                                                   pwfc=self.args.wave, fname=fname))
            print("edrixs >>> Setting-Up the Variables Done!")
            self.run_rixs()  # Run the one-, two-, and Dipole-part
            # Checking which transition-integrals has to be calculated
            for key in self.mode:
                if key == 'xas':
                    print("edrixs >>> Starting XAS ...")
                    # Updating the XAS-Plot-Parameters
                    self.pol_type = self.xas_pol_type
                    self.xas = trans_mat([self.omega[2], len(self.xas_pol_type)], dtype=np.float)
                    self.xas_spectra()
                    print("edrixs >>> XAS Done!")
                    if self.args.plot:
                        print("edrixs >>> Starting XAS-Plotting ...")
                        edrixs.xas_spectra(en=self.xasen, spc=self.xas, shift=self.shift[0], fname=fname)
                        print("edrixs >>> XAS-Plotting Done!")
                    if self.args.txtfile:
                        print("edrixs >>> Starting XAS-Saving ...")
                        edrixs.save_spc(en=self.xasen, spc=self.xas, fname=fname, shift=self.shift[0])
                        print("edrixs >>> Starting XAS Done!")
                if key == 'xes':
                    print("edrixs >>> Starting XES ...")
                    # Updating the XES-Plot-Parameters
                    self.pol_type = self.xes_pol_type
                    self.xes = trans_mat([self.eloss[2], len(self.xes_pol_type)], dtype=np.float)
                    # Run XES
                    self.xes_spectra()
                    print("edrixs >>> XES Done!")
                    if self.args.plot:
                        print("edrixs >>> XES-Plotting Done!")
                        edrixs.xes_spectra(en=self.xesen, spc=self.xes, fname=fname)
                        print("edrixs >>> XES-Plotting Done!")
                    if self.args.txtfile:
                        print("edrixs >>> Starting XES-Saving ...")
                        edrixs.save_spc(en=self.xesen, spc=self.xes, fname=fname, shift=self.shift[1], type='XES')
                        print("edrixs >>> Starting XES-Saving Done!")
                if key == 'rixs':
                    print("edrixs >>> Starting RIXS ...")
                    # Updating the XAS-Plot-Parameters
                    self.pol_type = self.rixs_pol_type
                    self.xas = trans_mat([self.omega[2], len(self.xas_pol_type)], dtype=np.float)
                    self.xes = trans_mat([self.eloss[2], len(self.xes_pol_type)], dtype=np.float)
                    self.rixs = trans_mat([self.omega[2], self.eloss[2], len(self.rixs_pol_type)], dtype=np.float)
                    self.rixs_spectra()
                    print("edrixs >>> RIXS Done!")
                    if self.args.plot:
                        print("edrixs >>> RIXS-Plotting ...")
                        edrixs.rixs_spectra(incident=self.xasen, emission=self.xesen,
                                            shift=self.shift[0], rixs=self.rixs, fname=fname)
                        edrixs.rxes_spectra(en=self.xesen, spc=self.rixs, fname=fname)
                        edrixs.rxas_spectra(en=self.xasen, spc=self.rixs,shift=self.shift[0], fname=fname)
                        print("edrixs >>> RIXS-Plotting Done!")
                    if self.args.txtfile:
                        print("edrixs >>> Starting RIXS-Saving ...")
                        edrixs.save_rixs(inc=self.xasen, emi=self.xesen, rixs=self.rixs, shift=self.shift[0],
                                         fname=fname)
                        print("edrixs >>> Starting RIXS-Saving Done!")

        if self.args.show:
            edrixs.plot_show()


if __name__ == '__main__':
    TanabeSugano().run_tanabe()
    #Read_Run_Plot().input_read()
    # RIXS = RIXS_XAS_XES_2p3d(F2_dd=[5.67,5.87],F4_dd=[4.6,4.8],F2_dp=7.446, G1_dp=5.566,
    #                         G3_dp=3.16,zeta_d=[.054,0.06], zeta_p=8.199
    #                         ,sym=[['Oh',1.47/10],['Oh',1.67/10]],d=9,ext_B=[0.5,0,0]
    #                         ,pol_type = [('left', 90,'right', 90),('right', 90,'right', 90)])
    # RIXS.run()
