#include <complex.h>
#include <stdio.h>
#include <cublas_v2.h> 
#include <math_constants.h>

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
    c[tid] = cuCmul(cuCmul(cuConj(a[ii]), b[ii]), c[tid]);
  }
}



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



__global__ void P_ai( double *spos_ac, double *ibzk_kc, int *op_scc, int *a_sa,
		      double **R_asii, cuDoubleComplex **P_ani, cuDoubleComplex **Pout_ai, 
		      int *Ni_a, bool time_rev, 
		      int Na, int s, int ik, int n){
  int tid = threadIdx.x;
  int ia = blockIdx.x;
  __shared__ cuDoubleComplex x;
  __shared__ double S_c[3];
  __shared__ double tmp;

  x=make_cuDoubleComplex(0., 0.);
  tmp = 0. ;

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

  if (tid < Ni){
    for (int j=0; j<Ni; j++){
      Pout_ai[ia][tid] = cuCadd(cuCmul(make_cuDoubleComplex(R_asii[ia][s*Ni*Ni+tid*Ni+j],0), P_ani[ib][n*Ni+j]), Pout_ai[ia][tid]);
      __syncthreads();
    }
    Pout_ai[ia][tid] = cuCmul(x, Pout_ai[ia][tid]);

    if (time_rev > 0){
      Pout_ai[ia][tid] = cuConj(Pout_ai[ia][tid]);
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
void cudaAdd( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  add<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMul( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  mul<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
void cudaMulc( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  mulc<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N);
}
}

extern "C" {
  void cudaMap_G2Q( double complex* dev_a, double complex* dev_b, int* dev_c, int N, int nG0, int nmultix ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  map_G2Q<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N, nG0, nmultix);
}
}

extern "C" {
  void cudaMap_Q2G( double complex* dev_a, double complex* dev_b, int* dev_c, int N, int nG0, int nmultix ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  map_Q2G<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N, nG0, nmultix);
}
}

extern "C" {
  void cudaDensity_matrix_R( double complex* dev_a, double complex* dev_b, double complex* dev_c, int N, int nmultix ) {
  int threads = 128;
  int nn = N * nmultix;
  int blocks = nn/threads + (nn%threads == 0 ? 0:1);
  density_matrix_R<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (cuDoubleComplex*)dev_c, N, nmultix);
}
}


extern "C" {
  void cudaTransform_wfs( double complex* dev_a, double complex* dev_b, int* dev_c, int N, int nmultix ) {
    int threads = 128;
    int nn = N * nmultix;
    int blocks = nn/threads + (nn%threads == 0 ? 0:1);
    trans_wfs<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N, nmultix );
  }
}

extern "C" {
  void cudaTransform_wfs_noindex( double complex* dev_a, double complex* dev_b, int* dev_c, double* dk, int N0, int N1, int N2 ) {
    int threads = 128;
    int N = N0 * N1 * N2;
    int blocks = N/threads + (N%threads == 0 ? 0:1);
    trans_wfs_noindex<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, (double*)dk, N0, N1, N2 );
  }
}

extern "C" {
  void cudaConj( double complex* dev_a, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  conj<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, N);
}
}

extern "C" {
  void cudaCopy( double complex* dev_a, double complex* dev_b, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  copy<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, N);
}
}


extern "C" {
  void cudaP_ai( double* dev_spos_ac, double* dev_ibzk_kc, int* dev_op_scc, int* dev_a_sa, 
		 double **dev_R_asii, double complex **P_ani, double complex **Pout_ai, int* Ni_a,
		 bool time_rev, int Na, int s, int ik, int n){

  int threads = 128;
  int blocks = Na;

  P_ai<<<blocks, threads>>>( (double*)dev_spos_ac, (double*)dev_ibzk_kc, 
			     (int*)dev_op_scc, (int*)dev_a_sa, 
			     (double**)dev_R_asii,(cuDoubleComplex**)P_ani, 
			     (cuDoubleComplex**)Pout_ai, (int*)Ni_a,
			     time_rev, Na, s, ik, n);
}
}



extern "C" {
  void cudaQ_anL( double complex **P1_ami, double complex **P2_ai,  
		  double **Delta_apL, double complex **Q_amL, int mband, int Na, 
		  int* Ni_a, int* nL_a){

  dim3 threads(16,16);
  int blocks = Na;

  Q_anL<<<blocks, threads>>>( (cuDoubleComplex**)P1_ami, (cuDoubleComplex**)P2_ai,
			     (double**)Delta_apL, (cuDoubleComplex**)Q_amL,
			     mband, Na, (int*)Ni_a, (int*)nL_a);

}
}

