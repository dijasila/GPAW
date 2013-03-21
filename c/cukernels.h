#ifndef CUKERNELS_H
#define CUKERNELS_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void cudaAdd( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N );

void cudaMul( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N );

void cudaMulc( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N );

void cudaMap_G2Q( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, int N, int nG0, int nmultix );

void cudaMap_Q2G( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, int N, int nG0, int nmultix );

void cudaDensity_matrix_R( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N, int nmultix );

void cudaOpt_phase( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N, int nmultix, int conjugate );

void cudaOpt_rhoG0_copy( cuDoubleComplex* rho_u, cuDoubleComplex* rho_uG, int nG0, int nmultix);

void cudaOpt_dE( cuDoubleComplex* rho_uG, int nG0, int nmultix, double* e_skn, int s, int k, int n, int* m_m, int nkpt, int nband);

void cudaC_wu( double* e_skn, double* f_skn, double* w_w, double* C_wu, cuDoubleComplex* alpha_wu, 
               int s, int k1, int k2, int n, int* m_u, int nu, int nmultix, int nkpt, int nband, int nw);

void cudaTransform_wfs( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, int N, int nmultix );

void cudaTransform_wfs_noindex( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, double* dk, int N0, int N1, int N2 );

void cudaConj( cuDoubleComplex* dev_a, int N );

void cudaCopy( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int N );

void cudaP_ani( double* dev_spos_ac, double* dev_ibzk_kc, int* dev_op_scc, int* dev_a_sa, 
                double **dev_R_asii, cuDoubleComplex **P_ani, cuDoubleComplex **Pout_ani, int* Ni_a,
                bool time_rev, int Na, int s, int ik, int* n_n, int n, int* offset, int NN);

void cudaP_aup( cuDoubleComplex **P1_ai, cuDoubleComplex **P2_aui, cuDoubleComplex **P_aup, 
                int* Ni_a, int Na, int n, int* offset, int NN);

void cudaQ_anL( cuDoubleComplex **P1_ami, cuDoubleComplex **P2_ai,  
                double **Delta_apL, cuDoubleComplex **Q_amL, int mband, int Na, 
                int* Ni_a, int* nL_a);

#ifdef __cplusplus
}
#endif

#endif
