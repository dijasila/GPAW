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

__global__ void map_G2Q( cuDoubleComplex *a, cuDoubleComplex *b, int *c, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[c[tid]] = a[tid];
}

__global__ void map_Q2G( cuDoubleComplex *a, cuDoubleComplex *b, int *c, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[tid] = a[c[tid]];
}

__global__ void trans_wfs( cuDoubleComplex *a, cuDoubleComplex *b, int *index, int n ){
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) b[index[tid]] = a[tid]; /*cuCmul(a[tid], phase[tid]);*/
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
		      double **R_asii, cuDoubleComplex **P_ani, cuDoubleComplex **Pout_ai, int *Ni_a, bool time_rev, 
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
void cudaMap_G2Q( double complex* dev_a, double complex* dev_b, int* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  map_G2Q<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N);
}
}

extern "C" {
void cudaMap_Q2G( double complex* dev_a, double complex* dev_b, int* dev_c, int N ) {
  int threads = 128;
  int blocks = N/threads + (N%threads == 0 ? 0:1);
  map_Q2G<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N);
}
}

extern "C" {
  void cudaTransform_wfs( double complex* dev_a, double complex* dev_b, int* dev_c, int N ) {
    int threads = 128;
    int blocks = N/threads + (N%threads == 0 ? 0:1);
    trans_wfs<<<blocks, threads>>>( (cuDoubleComplex*)dev_a, (cuDoubleComplex*)dev_b, (int*)dev_c, N );
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

  P_ai<<<blocks, threads>>>( (double*)dev_spos_ac, (double*)dev_ibzk_kc, (int*)dev_op_scc, (int*)dev_a_sa, 
			     (double**)dev_R_asii,(cuDoubleComplex**)P_ani, (cuDoubleComplex**)Pout_ai, (int*)Ni_a,
			     time_rev, Na, s, ik, n);
}
}


