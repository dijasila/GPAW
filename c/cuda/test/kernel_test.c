
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>


#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuComplex.h"

#ifdef __cplusplus
extern "C" {
#endif 
#include "../../bmgs/bmgs.h"
#include "../../bmgs/bmgs.c"


  double ddot(int *n, void *dx, int *incx, void *dy, int *incy);
  double ddot_(int *n, void *dx, int *incx, void *dy, int *incy);

#ifdef __cplusplus
}
#endif 


#include "../gpaw-cuda.h"  

int check_result(int n0,int n1, int n2,int blocks,double *b,double *b_cuda)
{
  int e_i=0;
  int bsize=n0*n1*n2;
  double error=0;
  for (int k=0;k<blocks;k++) {
    for (int i0=0;i0<n0;i0++){
      for (int i1=0;i1<n1;i1++){
	for (int i2=0;i2<n2;i2++){
	  long i=i0*n1*n2+i1*n2+i2; 
	  double ee=sqrt((b[i]-b_cuda[i])*(b[i]-b_cuda[i]));
	  if  (ee>1e-7) {
	    if (e_i<64)	    fprintf(stdout,"error [[%d,%d,%d],%d] er:%f cpu:%f gpu:%f\n",i0,i1,i2,k,ee,b[i],b_cuda[i]);
	    e_i++;
	  }
	  
	  error+=ee;
	}
      }
    }
    b+=n0*n1*n2;
    b_cuda+=n0*n1*n2;
  }
  if  (e_i>0)
    fprintf(stdout,"sum error %g count %d/%d\n",error,e_i,blocks*bsize);
  return e_i;

}


int check_resultz(int n0,int n1, int n2,int blocks,cuDoubleComplex *b,cuDoubleComplex *b_cuda)
{
  int e_i=0;
  int bsize=n0*n1*n2;
  double error=0;
  
  for (int k=0;k<blocks;k++) {
    for (int i0=0;i0<n0;i0++){
      for (int i1=0;i1<n1;i1++){
	for (int i2=0;i2<n2;i2++){
	  long i=i0*n1*n2+i1*n2+i2; 
	  double ee=sqrt((b[i].x-b_cuda[i].x)*(b[i].x-b_cuda[i].x)+(b[i].y-b_cuda[i].y)*(b[i].y-b_cuda[i].y));
	if  (ee>1e-14) {
	  //	  fprintf(stdout,"error %d %d %d %f %f %f\n",i0,i1,i2,ee,b[i],b_cuda[i]);
	  e_i++;
	}
	
	error+=ee;
	}
      }
    }
    b+=n0*n1*n2;
    b_cuda+=n0*n1*n2;
  }
  if  (e_i>0)
    fprintf(stdout,"sum error %g count %d/%d\n",error,e_i,bsize*blocks);
  return e_i;
}

int calc_test_fd(int n1,int n2,int n3,int blocks)
{
  bmgsstencil s;
  double h[3]={1.0, 1.0, 1.0};
  long   n[3]={n1,n2,n3};

  int ntimes=1;
  double *a,*b,*b_cuda;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3;

  s=bmgs_laplace(7,1.0,h,n);
  /* double coefs[19]={10.82556207,-1.98836854, 0.19883685,-0.01472866,-1.98836854,
		    0.19883685,-0.01472866,-1.98836854, 0.19883685,-0.01472866,
		    -1.98836854, 0.19883685,-0.01472866,-1.98836854, 0.19883685,
		    -0.01472866,  -1.98836854 ,  0.19883685,  -0.01472866};
  s.coefs=coefs;

  
  long offsets[]={0,1 ,   2 ,   3 ,  -1 ,  -2 ,  -3 ,  14 ,  28 ,  42 , -14,  -28,  -42 , 196 , 392,  588, -196, -392, -588};

  s.offsets=offsets;*/

  asize=s.j[0]+s.n[0]*(s.j[1]+s.n[1]*(s.n[2]+s.j[2]));
  bsize=s.n[0]*s.n[1]*s.n[2];
  
  a=malloc(blocks*asize*sizeof(double));
  b=malloc(blocks*bsize*sizeof(double));  
  b_cuda=malloc(blocks*bsize*sizeof(double));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<blocks*asize;i++){
    a[i]=rand()/(double)RAND_MAX;
  }
  
  bmgs_fd_cuda_cpu(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,1);

  memset(b,0,blocks*bsize*sizeof(double));  
  memset(b_cuda,0,blocks*bsize*sizeof(double));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++){
    for (int k=0;k<blocks;k++)
    bmgs_fd(&s,a+k*asize,b+k*bsize);
  }
  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_fd_cuda_cpu(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,blocks);
  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d,%3d) \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");

  free(a);
  free(b);
  free(b_cuda);
  return 0;

}
//extern "C" {
  int calc_test_dotu(int n1,int n2,int n3,int blocks)
  {
  long   n[3]={n1,n2,n3};

  int ntimes=5;
  int inc=1;
  double *a,*b,*c,*c_cuda,*c_cuda2,*a_gpu,*b_gpu;
  int asize,bsize;

  struct timeval  t0, t1, t2, t3; 
  double flops,flops2,flops3;

  // s=bmgs_laplace(7,1.0,h,n);


  asize=n[0]*n[1]*n[2];
  //asize=s.j[0]+s.n[0]*(s.j[1]+s.n[1]*(s.n[2]+s.j[2]));
  bsize=n[0]*n[1]*n[2];
  
  a=malloc(blocks*asize*sizeof(double));
  b=malloc(blocks*bsize*sizeof(double));  
  c=malloc(blocks*sizeof(double));  
  c_cuda=malloc(blocks*sizeof(double));  
  c_cuda2=malloc(blocks*sizeof(double));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<blocks*asize;i++){
    a[i]=rand()/(double)RAND_MAX;
  }
  for (int i=0;i<blocks*bsize;i++){
    b[i]=rand()/(double)RAND_MAX;
  }

  gpaw_cudaSafeCall(cudaMalloc((void**)&a_gpu,sizeof(double)*asize*blocks));
  
  gpaw_cudaSafeCall(cudaMalloc((void**)&b_gpu,sizeof(double)*bsize*blocks));
  
  
  gpaw_cudaSafeCall(cudaMemcpy(a_gpu,a,sizeof(double)*asize*blocks,
			       cudaMemcpyHostToDevice));
  gpaw_cudaSafeCall(cudaMemcpy(b_gpu,b,sizeof(double)*bsize*blocks,
			       cudaMemcpyHostToDevice));
  gpaw_cudaSafeCall(cudaGetLastError());
  
  mdotu_cuda_gpu(a_gpu,b_gpu,c_cuda,(int)asize,blocks);
  c_cuda[0] = cublasDdot(asize, (double*)a_gpu,
			 inc, (double*)b_gpu, inc);
  
  //bmgs_fd_cuda_cpu(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,1);
  c[0]=ddot_(&asize, a, &inc, b, &inc);
  memset(c_cuda,0,blocks*sizeof(double));  
  memset(c,0,blocks*sizeof(double));  
  cudaThreadSynchronize(); 
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++){
    for (int k=0;k<blocks;k++)
      c[k]=ddot_(&asize, a+k*asize, &inc, b+k*bsize, &inc);
  }
  /*
#ifdef __cplusplus
}

#endif     */
  gettimeofday(&t1,NULL);
  for (int i=0;i<ntimes;i++){
    for (int k=0;k<blocks;k++)
      c_cuda[k] = cublasDdot(asize, (double*)a_gpu+k*asize,
			     inc, (double*)b_gpu+k*bsize, inc);
  }
  cudaThreadSynchronize(); 
  //for (int i=0;i<ntimes;i++) flops3+=bmgs_fd_cuda_cpu(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,blocks);
  gettimeofday(&t2,NULL);
  for (int i=0;i<ntimes;i++)
    mdotu_cuda_gpu(a_gpu,b_gpu,c_cuda2,(int)asize,blocks);
  cudaThreadSynchronize(); 
  gettimeofday(&t3,NULL);
  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  flops3=((t3.tv_sec+t3.tv_usec/1000000.0-t2.tv_sec-t2.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d,%3d) \t %f\t %f\t %f\t %3.1f\t %3.1f\t %3.1f",n[0],n[1],n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops2),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops2,flops/flops3,flops2/flops3);

  for (int k=0;k<blocks;k++){
    c[k]/=asize;
    c_cuda[k]/=asize;
    c_cuda2[k]/=asize;

  }
  if (!check_result(1,1,1,blocks,c,c_cuda)) {
    printf("\tOK");
  }
  if (!check_result(1,1,1,blocks,c,c_cuda2)) {
    printf(" OK");
  }
  printf("\n");

  free(a);
  free(b);
  free(c);
  free(c_cuda);
  free(c_cuda2);
  gpaw_cudaSafeCall(cudaFree(a_gpu));
  gpaw_cudaSafeCall(cudaFree(b_gpu));
  return 0;

}
//  }

int calc_test_interpolate(int n1,int n2,int n3,int blocks)
{

  double h[3]={1.0, 1.0, 1.0};
  int   n[3]={n1,n2,n3};
  int skip[3][2]={{1,0},{1,0},{1,0}};

  //int skip[3][2]={{0,0},{0,0},{0,0}};
  int   b_n[3]={2*n1-2-skip[0][0]+skip[0][1],
		2*n2-2-skip[1][0]+skip[1][1],
		2*n3-2-skip[2][0]+skip[2][1]};

  int ntimes=1;
  double *a,*b,*b_cuda,*w;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3;

  
  asize=n[0]*n[1]*n[2];
  bsize=b_n[0]*b_n[1]*b_n[2];
  
  a=malloc(blocks*asize*sizeof(double));

  b=malloc(blocks*bsize*sizeof(double));  
  w=malloc(blocks*bsize*sizeof(double));  
  b_cuda=malloc(blocks*bsize*sizeof(double));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<blocks*asize;i++){
    a[i]=rand()/(double)RAND_MAX;
  }

  bmgs_interpolate_cuda_cpu(2,skip,a,n,b_cuda,blocks);

  memset(b,0,blocks*bsize*sizeof(double));  
  memset(w,0,blocks*bsize*sizeof(double));  
  memset(b_cuda,0,blocks*bsize*sizeof(double));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++){
    for (int k=0;k<blocks;k++)
      bmgs_interpolate(2,skip,a+k*asize,n,b+k*bsize,w);
  }

  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_interpolate_cuda_cpu(2,skip,a,n,b_cuda,blocks);
  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d,%3d) -> (%3d,%3d,%3d,%3d) \t %f\t %f\t %3.1f",n[0],n[1],n[2],blocks,b_n[0],b_n[1],b_n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_result(b_n[0],b_n[1],b_n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");

  free(a);

  free(b);

  free(w);

  free(b_cuda);

  return 0;

}

int calc_test_restrict(int n1,int n2,int n3,int blocks)
{

  double h[3]={1.0, 1.0, 1.0};
  int   n[3]={n1,n2,n3};
  //  int   b_n[3]={n1,n2,n3};
  //  int   b_n[3]={(n1-1)/2,(n2-1)/2,(n3-1)/2};
  int   b_n[3]={(n1-1)/2,(n2-1)/2,(n3-1)/2};


  int ntimes=1;
  double *a,*b,*b_cuda,*w,*a_cuda;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3;

  
  asize=n[0]*n[1]*n[2];
  bsize=(b_n[0])*(b_n[1])*(b_n[2]);
  
  a=malloc(blocks*asize*sizeof(double));
  a_cuda=malloc(blocks*asize*sizeof(double));

  b=malloc(blocks*bsize*sizeof(double));  
  w=malloc(blocks*asize*sizeof(double));  
  b_cuda=malloc(blocks*bsize*sizeof(double));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<blocks*asize;i++){
    a[i]=rand()/(double)RAND_MAX;
    a_cuda[i]=a[i];
  }

  bmgs_restrict_cuda_cpu(2,a_cuda,n,b_cuda,blocks);

  for (int i=0;i<blocks*asize;i++){
    a_cuda[i]=a[i];
  }
  memset(b,0,blocks*bsize*sizeof(double));  
  memset(w,0,blocks*asize*sizeof(double));  
  memset(b_cuda,0,blocks*bsize*sizeof(double));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++){
    for (int k=0;k<blocks;k++)
      bmgs_restrict(2,a+k*asize,n,b+k*bsize,w);
  }

  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_restrict_cuda_cpu(2,a_cuda,n,b_cuda,blocks);
  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d,%3d) -> (%3d,%3d,%3d,%3d) \t %f\t %f\t %3.1f",n[0],n[1],n[2],blocks,b_n[0],b_n[1],b_n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_result(b_n[0],b_n[1],b_n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");
  free(a);
  free(a_cuda);
  free(b);

  free(w);
  free(b_cuda);

  return 0;

}




int calc_test_paste(int n1,int n2,int n3,int blocks)
{
  bmgsstencil s;
  double h[3]={1.0, 1.0, 1.0};
  long   n[3]={n1,n2,n3};

  int ntimes=10;
  double *a,*b,*b_cuda,*b_cuda2;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3,flops4;

  s=bmgs_laplace(7,1.0,h,n);

  asize=s.j[0]+s.n[0]*(s.j[1]+s.n[1]*(s.n[2]+s.j[2]));
  bsize=s.n[0]*s.n[1]*s.n[2];
  

  int c[3]={0,0,0};
  int nia[3]={1,1,s.n[0]*s.n[1]*s.n[2]};
  int ni[3]={s.n[0],s.n[1],s.n[2]};

  a=malloc(blocks*asize*sizeof(double));
  b=malloc(blocks*bsize*sizeof(double));  
  b_cuda=malloc(blocks*bsize*sizeof(double));  
  b_cuda2=malloc(blocks*bsize*sizeof(double));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<blocks*asize;i++){
    a[i]=rand()/(double)RAND_MAX;
  }
  
  bmgs_paste_cuda_cpu(a,ni,b_cuda,ni,c);
  memset(b,0,blocks*bsize*sizeof(double));  
  memset(b_cuda,0,blocks*bsize*sizeof(double));  
  memset(b_cuda2,0,blocks*bsize*sizeof(double));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++){
    for (int k=0;k<blocks;k++)
      bmgs_paste(a+k*asize,ni,b+k*bsize,ni,c);
  }
  gettimeofday(&t1,NULL);
  flops3=0;
  flops4=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_paste_cuda_cpu(a,ni,b_cuda,ni,c);


  //  for (int i=0;i<ntimes;i++) flops4+=bmgs_paste_cuda_cpu2(a,ni,b_cuda2,ni,c);

  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d,%3d) \t %f\t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  /*  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda2)) {
    printf("\tOK");
    }*/
  printf("\n");

  free(a);
  free(b);
  free(b_cuda);
  return 0;

}


int calc_test_relax(int n1,int n2,int n3)
{
  bmgsstencil s;
  double h[3]={1.0, 1.0, 1.0};
  long   n[3]={n1,n2,n3};
  int blocks=1;
  int ntimes=10;
  double *a,*b,*b_cuda;
  double *src;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3;
  double w=2/3;
  int nrelax=10;
  int relax_method=2;

  s=bmgs_laplace(7,1.0,h,n);

  asize=s.j[0]+s.n[0]*(s.j[1]+s.n[1]*(s.n[2]+s.j[2]));
  bsize=s.n[0]*s.n[1]*s.n[2];
  
  a=malloc(asize*sizeof(double));
  b=malloc(bsize*sizeof(double));  
  b_cuda=malloc(bsize*sizeof(double));  
  src=malloc(bsize*sizeof(double));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<asize;i++){
    a[i]=rand()/(double)RAND_MAX;
  }
  for (int i=0;i<bsize;i++){
    src[i]=rand()/(double)RAND_MAX;
  }
  
  for(int i=0;i<s.ncoefs;i++){
    if (s.offsets[i]==0){
      long temp=s.offsets[0];
      s.offsets[0]=s.offsets[i];
      s.offsets[i]=temp;
      double ctemp=s.coefs[0];
      s.coefs[0]=s.coefs[i];
      s.coefs[i]=ctemp;
    }

  }

  bmgs_relax_cuda_cpu(relax_method,&s,a,b_cuda,src,w);
  memset(b,0,bsize*sizeof(double));  
  memset(b_cuda,0,bsize*sizeof(double));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<nrelax;i++) bmgs_relax(relax_method,&s,a,b,src,w);
  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<nrelax;i++) flops3+=  bmgs_relax_cuda_cpu(relax_method,&s,a,b_cuda,src,w);
  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d) \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],nrelax*bsize/(1000000.0*flops),nrelax*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");

  free(a);
  free(b);
  free(src);
  free(b_cuda);
  return 0;

}

int calc_test_fdz(int n1,int n2,int n3,int blocks)
{
  bmgsstencil s;
  double h[3]={1.0, 1.0, 1.0};
  long   n[3]={n1,n2,n3};

  int ntimes=10;
  cuDoubleComplex *a,*b,*b_cuda;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3;

  s=bmgs_laplace(7,1.0,h,n);

  asize=s.j[0]+s.n[0]*(s.j[1]+s.n[1]*(s.n[2]+s.j[2]));
  bsize=s.n[0]*s.n[1]*s.n[2];
  
  a=malloc(blocks*asize*sizeof(cuDoubleComplex));
  b=malloc(blocks*bsize*sizeof(cuDoubleComplex));  
  b_cuda=malloc(blocks*bsize*sizeof(cuDoubleComplex));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<blocks*asize;i++){
    a[i].x=rand()/(double)RAND_MAX;
    a[i].y=rand()/(double)RAND_MAX;
  }
  
  bmgs_fd_cuda_cpuz(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,1);

  memset(b,0,blocks*bsize*sizeof(cuDoubleComplex));  
  memset(b_cuda,0,blocks*bsize*sizeof(cuDoubleComplex));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++) 
    for (int k=0;k<blocks;k++)
      bmgs_fdz(&s,(double complex*)a+k*asize,(double complex*)b+k*bsize);

  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_fd_cuda_cpuz(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,blocks);
  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d,%3d) \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_resultz(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");
  /*
  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("(%3d,%3d,%3d) \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],ntimes*bsize/(1000000.0*flops),ntimes*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_resultz(s.n[0],s.n[1],s.n[2],b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");*/
  free(a);
  free(b);
  free(b_cuda);
  return 0;

}

int main(void)
{
  struct cudaDeviceProp prop;
  int device=1;
  cudaSetDevice(device);  
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  cudaGetDeviceProperties(&prop,device);
  
  printf("Device: %s\n",prop.name);
  printf("Device compute capability: %d.%d\n",prop.major,prop.minor);
  printf("Device multiprocessors: %d\n",prop.multiProcessorCount);
  printf("Device clock rate: %d MHz\n",prop.clockRate/1000);
  printf("Device mem: %d MB\n",prop.totalGlobalMem/1024/1024);
  printf("Device mem clock rate: %d MHz\n",prop.memoryClockRate/1000);
  printf("Device mem bus width: %d bits\n",prop.memoryBusWidth);
  printf("Device ECC: %d\n",prop.ECCEnabled);
  printf("Device L2 cache: %d kB\n",prop.l2CacheSize/1024);

  cudaSetDevice(1);
    cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  //for (int n=16;n<449;n+=6)
  /*  printf("# bmgs_paste  \n");  
      printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
      for (int n=16;n<280;n+=24)
      calc_test_paste(n,n,n,1);*/
  /*printf("# bmgs_fdz  \n");  
    printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  calc_test_fdz(8,8,8,2);  

  exit(0);
  */
  /*  printf("# bmgs_fdz  \n");  
  printf("# N \t\t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  calc_test_fdz(8,8,8,2);
  exit(0);
  */

    printf("# dotu\n");  
  printf("# N \t\t\t CPU Mpoint/s \t GPU Mpoint/s \t GPU2 Mpoint/s \t Speed-up\n");    
  calc_test_dotu(8,8,8,1);
  for (int n=16;n<300;n+=6){
    calc_test_dotu(n,n,n,1);
  }
  for (int n=16;n<140;n+=24)
    calc_test_dotu(n,n,n,16);
  for (int n=1;n<=512;n*=2)
    calc_test_dotu(30,30,30,n);
  for (int n=1;n<=512;n*=2)
    calc_test_dotu(60,60,60,n);
  //  exit(0);
  
  printf("# bmgs_restrict\n");  
  printf("# N \t\t\t\t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");    //
  //calc_test_restrict(28,28,28,1);
  //
  calc_test_restrict(8,8,8,1);
  for (int n=16;n<300;n+=6){
    calc_test_restrict(n,n,n,1);
  }
  for (int n=16;n<140;n+=24)
    calc_test_restrict(n,n,n,16);
  for (int n=1;n<=128;n*=2)
    calc_test_restrict(30,30,30,n);
  for (int n=1;n<=128;n*=2)
    calc_test_restrict(60,60,60,n);
  //    exit(0);
  printf("# bmgs_interpolate  \n");  
  printf("# N \t\t\t\t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  // calc_test_interpolate(2,2,2,1);
  //calc_test_interpolate(4,4,4,1);
  calc_test_interpolate(8,8,8,1);
  for (int n=16;n<300/2;n+=6){
    calc_test_interpolate(n,n,n,1);
  }
  for (int n=8;n<70;n+=140)
   calc_test_interpolate(n,n,n,16);
  for (int n=1;n<=128;n*=2)
    calc_test_interpolate(30,30,30,n);
  for (int n=1;n<=128;n*=2)
    calc_test_interpolate(60,60,60,n);
  
  
  printf("# bmgs_fd  \n");  
  printf("# N \t\t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  //calc_test_fd(2,2,2,1);
  //calc_test_fd(4,4,4,1);
  calc_test_fd(8,8,8,1);
  for (int n=16;n<300;n+=6){
    calc_test_fd(n,n,n,1);
  }
  for (int n=16;n<140;n+=24)
    calc_test_fd(n,n,n,16);
  for (int n=1;n<=128;n*=2)
    calc_test_fd(30,30,30,n);
  for (int n=1;n<=128;n*=2)
    calc_test_fd(60,60,60,n);



  printf("# bmgs_fdz  \n");  
  printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  //calc_test_fdz(2,2,2,1);
  //calc_test_fdz(4,4,4,1);
  calc_test_fdz(8,8,8,1);
  for (int n=16;n<300;n+=6)
    calc_test_fdz(n,n,n,1);
  printf("# bmgs_relax  \n");  
  printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  calc_test_relax(2,2,2);
  calc_test_relax(4,4,4);
  calc_test_relax(8,8,8);
  for (int n=16;n<300;n+=6)
    calc_test_relax(n,n,n);
  
}


