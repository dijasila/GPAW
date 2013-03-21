#include <complex.h>
#include <stdio.h>
#include <cublas_v2.h> 
#include <math_constants.h>
#include "cukernels.h"

__global__ void add( cuDoubleComplex* a, cuDoubleComplex* b, cuDoubleComplex* c, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N)
    c[tid] = cuCadd(a[tid], b[tid]);
}

__global__ void mul( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) c[tid] = cuCmul(a[tid], b[tid]);
}

__global__ void mulc( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) c[tid] = cuCmul(cuConj(a[tid]), b[tid]);
}

__global__ void map_G2Q( cuDoubleComplex *a, cuDoubleComplex *b, int *c, int n, int nG0, int nmultix ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n*nmultix) b[c[tid%n]+tid/n*nG0] = a[tid];
}

__global__ void map_Q2G( cuDoubleComplex *a, cuDoubleComplex *b, int *c, int n, int nG0, int nmultix ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n*nmultix) b[tid] = a[c[tid%n]+tid/n*nG0];
}

__global__ void density_matrix_R( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int n, int nmultix ){
  /* perform psit1_R.conj() * expqr_R * psit2_uR */
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ii = tid%n;
  if (tid < n*nmultix) {
    /*    c[tid] = cuCmul(cuCmul(cuConj(a[ii]), b[ii]), c[tid]);*/
    c[tid] = cuCmul(cuCmul(a[ii], b[ii]), c[tid]);
  }
}

__global__ void opt_phase( cuDoubleComplex *a, cuDoubleComplex *b, cuDoubleComplex *c, int n, int nmultix, int conjugate ){
  /* perform optexp_R * optpsit2_uR */
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ii = tid%n;
  if (tid < n*nmultix) {
    if (conjugate > 0){
      c[tid] = cuCmul(cuConj(a[ii]), b[tid]);
    }
    else{
      c[tid] = cuCmul(a[ii], b[tid]);
    }
  }
}

__global__ void opt_rhoG0_copy( cuDoubleComplex *rho_u, cuDoubleComplex *rho_uG, int nG0, int nmultix){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < nmultix) {
    rho_uG[tid*nG0] = rho_u[tid]; 
  }
}


__global__ void opt_dE( cuDoubleComplex *rho_uG, int nG0, int nmultix, double *e_skn, int s, int k, int n, int *m_m, int nkpt, int nband){
  /* perform optexp_R * optpsit2_uR */
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double e1 = e_skn[s*nkpt*nband + k*nband + n];

  if (tid < nmultix) {
    double e2 = e_skn[s*nkpt*nband + k*nband + m_m[tid]] - e1;
    if (e2 > 0.0036749309467970742){
      rho_uG[tid*nG0] = make_cuDoubleComplex( cuCreal(rho_uG[tid*nG0]) / e2, cuCimag(rho_uG[tid*nG0]) / e2 ); 
    }
    else{
      rho_uG[tid*nG0] = make_cuDoubleComplex(0., 0.);
    }
  }
}


__global__ void GetC_wu( double *e_skn, double *f_skn, double *w_w, double *C_wu, cuDoubleComplex *alpha_wu, 
		      int s, int k1, int k2, int n, int *m_u, int nu, int nmultix, int nkpt, int nband, int nw){
  /* perform optexp_R * optpsit2_uR */
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double e1 = e_skn[s*nkpt*nband + k1*nband + n];
  double f1 = f_skn[s*nkpt*nband + k1*nband + n];
  int nn = nw * nu;

  if (tid < nn) {
    int iw = tid/nu;
    int iu = tid%nu;
    double e2 = e_skn[s*nkpt*nband + k2*nband + m_u[iu]] - e1;
    double f2 = f1 - f_skn[s*nkpt*nband + k2*nband + m_u[iu]];

    C_wu[iw*nmultix+iu] = 2*e2*f2 / (w_w[iw] - e2*e2);
  }
  __syncthreads();

  if (tid < nn) {
    int iw = tid/nu;
    int iu = tid%nu;
    if (iw > 0){
      alpha_wu[iw*nmultix+iu] = make_cuDoubleComplex( sqrt( C_wu[iw*nmultix+iu] / C_wu[(iw-1)*nmultix+iu]), 0.) ;
    }
    else{
      alpha_wu[iw*nmultix+iu] = make_cuDoubleComplex( sqrt(-C_wu[iw*nmultix+iu]), 0.) ;      
    }
  }
}


/*
__global__ void Apply_alpha_u( double *alpha_wu, cuDoubleComplex *rho_uG, int iw, int nu, int nmultix, int npw){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int nn = nu * npw;
  if (tid < nn) {
    int iu = tid/npw;
    rho_uG[tid] = make_cuDoubleComplex( cuCreal(rho_uG[tid]) * alpha_wu[iw*nmultix+iu], cuCimag(rho_uG[tid]) * alpha_wu[iw*nmultix+iu] );
  }

}
*/

__global__ void trans_wfs( cuDoubleComplex *a, cuDoubleComplex *b, int *index, int n, int nmultix ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n*nmultix) b[index[tid%n]+tid/n*n] = a[tid]; /*cuCmul(a[tid], phase[tid]);*/
}

__global__ void trans_wfs_noindex( cuDoubleComplex *a, cuDoubleComplex *b, int *C, double *dk, int ng0, int ng1, int ng2 ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int n = ng0 * ng1 * ng2;
  if (tid < n) {
    int g2 = tid % ng2;
    int g1 = (tid / ng2) % ng1;
    int g0 = (tid / ng2) / ng1;

    int p0 = ((C[0] * g0 + C[3] * g1 + C[6] * g2) % ng0 + ng0) % ng0;
    int p1 = ((C[1] * g0 + C[4] * g1 + C[7] * g2) % ng1 + ng1) % ng1;
    int p2 = ((C[2] * g0 + C[5] * g1 + C[8] * g2) % ng2 + ng2) % ng2;

    int index = (p0 * ng1 + p1) * ng2 + p2;
    double tmp = dk[0]/ng0*p0 + dk[1]/ng1*p1 + dk[2]/ng2*p2; 
    cuDoubleComplex phase  = make_cuDoubleComplex(cos(2*CUDART_PI*tmp), sin(2*CUDART_PI*tmp));

    b[index] = cuCmul(a[tid], phase); 
  }

}


__global__ void conj( cuDoubleComplex *a, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) a[tid] = cuConj(a[tid]);
}

__global__ void copy( cuDoubleComplex *a, cuDoubleComplex *b, int N ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < N) b[tid] = a[tid];
}


/*
__global__ void Pold_ai( double *spos_ac, double *ibzk_kc, int *op_scc, int *a_sa,
		      double **R_asii, cuDoubleComplex **P_ani, cuDoubleComplex **Pout_ani, 
		      int *Ni_a, bool time_rev, 
		      int Na, int s, int ik, int *n_n, int n){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  __shared__ cuDoubleComplex x;
  __shared__ double S_c[3];
  __shared__ double tmp;

  x=make_cuDoubleComplex(0., 0.);
  tmp = 0. ;

  for (int ia=0; ia<Na; ia++){
      int ib=a_sa[s*Na+ia];
      int Ni=Ni_a[ia];
    
      if (tid < 3){
        S_c[tid] = 0.;
        for (int dim=0; dim<3; dim++){
          S_c[tid] += spos_ac[ia*3+dim] * op_scc[s*9+dim*3+tid] ;
         }
        S_c[tid] -= spos_ac[ib*3+tid];
      }
    
      __syncthreads();

      tmp = S_c[0] * ibzk_kc[ik*3+0] + S_c[1] * ibzk_kc[ik*3+1] + S_c[2] * ibzk_kc[ik*3+2];

      x = make_cuDoubleComplex(cos(2*CUDART_PI*tmp), sin(2*CUDART_PI*tmp));

      int m = tid / Ni;
      int i = tid % Ni;

      if (tid < Ni*n){
        for (int j=0; j<Ni; j++){
	  Pout_ani[ia][m*Ni+i] = cuCadd(cuCmul(make_cuDoubleComplex(R_asii[ia][s*Ni*Ni+i*Ni+j],0), P_ani[ib][n_n(m)*Ni+j]), 
  				Pout_ani[ia][m*Ni+i]);
        }
  
        Pout_ani[ia][m*Ni+i] = cuCmul(x, Pout_ai[ia][m*Ni+i]);
  
        if (time_rev > 0){
          Pout_ani[ia][m*Ni+i] = cuConj(Pout_ai[ia][m*Ni+i]);
        }
      }
      __syncthreads();

  }

}
*/

__global__ void P_ai( double *spos_ac, double *ibzk_kc, int *op_scc, int *a_sa,
		      double **R_asii, cuDoubleComplex **P_ani, cuDoubleComplex **Pout_ani, 
		      int *Ni_a, bool time_rev, 
		      int Na, int s, int ik, int *n_n, int n, int *offset, int NN){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  cuDoubleComplex x;
  double S_c[3];
  double tmp;
  int ia;
  x=make_cuDoubleComplex(0., 0.);
  tmp = 0. ;

  if (tid < NN*n){
    for (int i=0; i<Na; i++){
      if (tid < offset[i+1]*n && tid >= offset[i]*n){
        ia = i;
        break;
      }
    }
  
    /*    printf("%d,%d,\n", tid,ia);*/

    int ib=a_sa[s*Na+ia];
    int Ni=Ni_a[ia];
    int Nj=Ni_a[ib];
    
    for (int i=0; i<3; i++){
      S_c[i] = 0.;
      for (int dim=0; dim<3; dim++){
        S_c[i] += spos_ac[ia*3+dim] * op_scc[s*9+dim*3+i] ;
       }
      S_c[i] -= spos_ac[ib*3+i];
    }
    
    tmp = S_c[0] * ibzk_kc[ik*3+0] + S_c[1] * ibzk_kc[ik*3+1] + S_c[2] * ibzk_kc[ik*3+2];
  
    x = make_cuDoubleComplex(cos(2*CUDART_PI*tmp), sin(2*CUDART_PI*tmp));
  
    int m = (tid-offset[ia]*n) / Ni;
    int i = (tid-offset[ia]*n) % Ni;
    /*    printf("%d,%d,%d\n", tid,m,i);*/
  
    for (int j=0; j<Nj; j++){
    	  Pout_ani[ia][m*Ni+i] = cuCadd(cuCmul(make_cuDoubleComplex(R_asii[ia][s*Ni*Nj+i*Nj+j],0), P_ani[ib][n_n[m]*Nj+j]), 
    				Pout_ani[ia][m*Ni+i]);
	  /*
	  printf("Pout_ani, %d,%f,%f\n", j, cuCreal(Pout_ani[ia][m*Ni+i]), cuCimag(Pout_ani[ia][m*Ni+i]));
	  printf("P_ani, %d,%f,%f\n", j, cuCreal(P_ani[ib][n_n[m]*Nj+j]), cuCimag(P_ani[ia][n_n[m]*Nj+j]));
	  */
    }
    
    Pout_ani[ia][m*Ni+i] = cuCmul(x, Pout_ani[ia][m*Ni+i]);
    
    if (time_rev > 0){
      Pout_ani[ia][m*Ni+i] = cuConj(Pout_ani[ia][m*Ni+i]);
    }
  }
}



__global__ void P_ai_outer(cuDoubleComplex **P1_ai, cuDoubleComplex **P2_aui,  
			   cuDoubleComplex **P_aup, 
			   int *Ni_a, int Na, int n, int *offset, int NN){
  /* NN is Ni_a.sum(), n_n is a list of bands, n is len(n_n)*/
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int ia;

  if (tid < NN*n){
    for (int i=0; i<Na; i++){
      if (tid < offset[i+1]*n && tid >= offset[i]*n){
        ia = i;
        break;
      }
    }
  
    int Ni=Ni_a[ia];

    for (int i=0; i<Ni; i++){
      int iu = (tid-offset[ia]*n) / Ni;
      int j = (tid-offset[ia]*n) % Ni;
      int p = Ni*i + j;
      P_aup[ia][iu*Ni*Ni+p] = cuCmul(cuConj(P1_ai[ia][i]),  P2_aui[ia][iu*Ni+j]);
    }
  }
}



__global__ void Q_anL(cuDoubleComplex **P1_ami, cuDoubleComplex **P2_ai, 
		      double **Delta_apL, cuDoubleComplex **Q_amL, int mband, int Na, 
		      int *Ni_a, int *nL_a){
  int tidx = threadIdx.x;
  int tidy = threadIdx.y;  
  int ia = blockIdx.x;

  int Ni=Ni_a[ia];
  int nL=nL_a[ia];

  if (blockDim.x < mband) printf("Q_anL calculation is wrong !! ");
  if (blockDim.y < nL) printf("Q_anL calculation is wrong !! ");

  if (tidx < mband && tidy < nL){
    for (int ix=0; ix<Ni; ix++){
      for (int iy=0; iy<Ni; iy++){
        int ip = ix*Ni+iy;
	cuDoubleComplex tmp = cuCmul(P1_ami[ia][tidx*Ni+ix], P2_ai[ia][iy]);
	cuDoubleComplex tmp2 = make_cuDoubleComplex(Delta_apL[ia][ip*nL+tidy] * cuCreal(tmp), 
			     Delta_apL[ia][ip*nL+tidy] * cuCimag(tmp));
        Q_amL[ia][tidx*nL+tidy] = cuCadd(Q_amL[ia][tidx*nL+tidy], tmp2);
      }
    }
  }

  /*  __shared__ cuDoubleComplex Q_mL;
      Q_mL = Q_amL[ia];*/

  /*  
  if (tidx < Ni && tidy < Ni){
    for (int im=0; im<mband; im++){
      cuDoubleComplex tmp = cuCmul(P1_ami[ia][im*Ni+tidx], P2_ai[ia][tidy]);
      for (int i=0; i<nL; i++){
        int ip=tidx+Ni+tidy;
	cuDoubleComplex tmp2 = make_cuDoubleComplex(Delta_apL[ia][ip*nL+i] * cuCreal(tmp), 
			     Delta_apL[ia][ip*nL+i] * cuCimag(tmp));
	atomicAdd( &(Q_amL[ia][im*nL+i]), tmp2);
      }
    }
  }

  __syncthreads();
*/
}


extern "C" {
void cudaAdd( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  add<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMul( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  mul<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMulc( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  mulc<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
  void cudaMap_G2Q( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, int N, int nG0, int nmultix ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  map_G2Q<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N, nG0, nmultix);
}
}

extern "C" {
  void cudaMap_Q2G( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, int N, int nG0, int nmultix ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  map_Q2G<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N, nG0, nmultix);
}
}

extern "C" {
  void cudaDensity_matrix_R( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N, int nmultix ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  density_matrix_R<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N, nmultix);
}
}

extern "C" {
  void cudaOpt_phase( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, cuDoubleComplex* dev_c, int N, int nmultix, int conjugate ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  opt_phase<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N, nmultix, conjugate);
}
}


extern "C" {
  void cudaOpt_rhoG0_copy( cuDoubleComplex* rho_u, cuDoubleComplex* rho_uG, int nG0, int nmultix){
  int threads = 64;
  int nn = nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  opt_rhoG0_copy<<<blocks, threads>>>( (cuDoubleComplex*)rho_u, (cuDoubleComplex*)rho_uG, nG0, nmultix);
}
}


extern "C" {
  void cudaOpt_dE( cuDoubleComplex* rho_uG, int nG0, int nmultix, double* e_skn, int s, int k, int n, int* m_m, int nkpt, int nband){
  int threads = 64;
  int nn = nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  opt_dE<<<blocks, threads>>>( (cuDoubleComplex*)rho_uG, nG0, nmultix, (double*)e_skn, s, k, n, (int*)m_m, nkpt, nband);
}
}


extern "C" {
  void cudaC_wu( double* e_skn, double* f_skn, double* w_w, double* C_wu, cuDoubleComplex* alpha_wu, 
		 int s, int k1, int k2, int n, int* m_u, int nu, int nmultix, int nkpt, int nband, int nw){
  int threads = 128;
  int nn = nu * nw;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  GetC_wu<<<blocks, threads>>>( (double*)e_skn, (double*)f_skn, (double*)w_w, (double*)C_wu, (cuDoubleComplex*)alpha_wu,
			       s, k1, k2, n, (int*)m_u, nu, nmultix, nkpt, nband, nw);
}
}

/*
extern "C" {
  void cudaalpha_u( double* alpha_wu, cuDoubleComplex* rho_uG, int iw, int nu, int nmultix, int npw){
  int threads = 128;
  int nn = nu * npw;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  Apply_alpha_u<<<blocks, threads>>>( (double*)alpha_wu, (cuDoubleComplex*)rho_uG, iw, nu, nmultix, npw);
}
}
*/

extern "C" {
  void cudaTransform_wfs( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, int N, int nmultix ) {
    int threads = 128;
    int nn = N * nmultix;
    int blocks = nn/threads + (nn%threads == 0 ? 0:1);
    trans_wfs<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N, nmultix );
  }
}

extern "C" {
  void cudaTransform_wfs_noindex( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int* dev_c, double* dk, int N0, int N1, int N2 ) {
    int threads = 128;
    int N = N0 * N1 * N2;
    int blocks = N/threads + (N%threads == 0 ? 0:1);
    trans_wfs_noindex<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, (double*)dk, N0, N1, N2 );
  }
}

extern "C" {
  void cudaConj( cuDoubleComplex* dev_a, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  conj<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, N);
}
}

extern "C" {
  void cudaCopy( cuDoubleComplex* dev_a, cuDoubleComplex* dev_b, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  copy<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, N);
}
}


extern "C" {
  void cudaP_ani( double* dev_spos_ac, double* dev_ibzk_kc, int* dev_op_scc, int* dev_a_sa, 
		 double **dev_R_asii, cuDoubleComplex **P_ani, cuDoubleComplex **Pout_ani, int* Ni_a,
		 bool time_rev, int Na, int s, int ik, int* n_n, int n, int* offset, int NN){

  int threads = 128;
  int blocks = (NN*n)/threads + ((NN*n)%threads == 0 ? 0:1);

  P_ai<<<blocks, threads>>>( (double*)dev_spos_ac, (double*)dev_ibzk_kc, 
			     (int*)dev_op_scc, (int*)dev_a_sa, 
			     (double**)dev_R_asii,(cuDoubleComplex**)P_ani, 
			     (cuDoubleComplex**)Pout_ani, (int*)Ni_a,
			     time_rev, Na, s, ik, (int*)n_n, n, (int*)offset, NN);
}
}


extern "C" {
  void cudaP_aup( cuDoubleComplex **P1_ai, cuDoubleComplex **P2_aui, cuDoubleComplex **P_aup, 
		  int* Ni_a, int Na, int n, int* offset, int NN){

  int threads = 128;
  int blocks = (NN*n)/threads + ((NN*n)%threads == 0 ? 0:1);

  P_ai_outer<<<blocks, threads>>>( (cuDoubleComplex**)P1_ai, (cuDoubleComplex**)P2_aui, 
			     (cuDoubleComplex**)P_aup, (int*)Ni_a,
			     Na, n, (int*)offset, NN);
}
}

extern "C" {
  void cudaQ_anL( cuDoubleComplex **P1_ami, cuDoubleComplex **P2_ai,  
		  double **Delta_apL, cuDoubleComplex **Q_amL, int mband, int Na, 
		  int* Ni_a, int* nL_a){

  dim3 threads(16,16);
  int blocks = Na;

  Q_anL<<<blocks, threads>>>( (cuDoubleComplex**)P1_ami, (cuDoubleComplex**)P2_ai,
			     (double**)Delta_apL, (cuDoubleComplex**)Q_amL,
			     mband, Na, (int*)Ni_a, (int*)nL_a);

}
}
