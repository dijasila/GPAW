
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
    fprintf(stdout,"sum error %g count %d/%d\n",error,e_i,bsize);
  return e_i;

}


int check_resultz(int n0,int n1, int n2,cuDoubleComplex *b,cuDoubleComplex *b_cuda)
{
  int e_i=0;
  int bsize=n0*n1*n2;
  double error=0;
  
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
  if  (e_i>0)
    fprintf(stdout,"sum error %g count %d/%d\n",error,e_i,bsize);
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
  printf ("%dx%dx%dx%d \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");

  free(a);
  free(b);
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


  for (int i=0;i<ntimes;i++) flops4+=bmgs_paste_cuda_cpu2(a,ni,b_cuda2,ni,c);

  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("%dx%dx%dx%d \t %f\t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],blocks,ntimes*blocks*bsize/(1000000.0*flops),ntimes*blocks*bsize/(1000000.0*flops3),ntimes*blocks*bsize/(1000000.0*flops4),flops/flops3);

  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda)) {
    printf("\tOK");
  }
  if (!check_result(s.n[0],s.n[1],s.n[2],blocks,b,b_cuda2)) {
    printf("\tOK");
  }
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
  printf ("%dx%dx%d \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],nrelax*bsize/(1000000.0*flops),nrelax*bsize/(1000000.0*flops3),flops/flops3);

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
  
  a=malloc(asize*sizeof(cuDoubleComplex));
  b=malloc(bsize*sizeof(cuDoubleComplex));  
  b_cuda=malloc(bsize*sizeof(cuDoubleComplex));  

  srand ( time(NULL) );
  //srand (0 );

  for (int i=0;i<asize;i++){
    a[i].x=rand()/(double)RAND_MAX;
    a[i].y=rand()/(double)RAND_MAX;
  }
  
  bmgs_fd_cuda_cpuz(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,blocks);

  memset(b,0,bsize*sizeof(cuDoubleComplex));  
  memset(b_cuda,0,bsize*sizeof(cuDoubleComplex));  
  gettimeofday(&t0,NULL);

  for (int i=0;i<ntimes;i++) bmgs_fdz(&s,(double complex*)a,(double complex*)b);
  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_fd_cuda_cpuz(&s,a,b_cuda,GPAW_BOUNDARY_NORMAL,blocks);
  gettimeofday(&t2,NULL);

  flops=((t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0)); 
  flops2=((t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0));
  printf ("%dx%dx%d \t %f\t %f\t %3.1f",s.n[0],s.n[1],s.n[2],ntimes*bsize/(1000000.0*flops),ntimes*bsize/(1000000.0*flops3),flops/flops3);

  if (!check_resultz(s.n[0],s.n[1],s.n[2],b,b_cuda)) {
    printf("\tOK");
  }
  printf("\n");
  free(a);
  free(b);
  free(b_cuda);
  return 0;

}

int main(void)
{
  cudaSetDevice(1);
  cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  srand((unsigned int) time(NULL));
  //srand(0);
    //for (int n=16;n<449;n+=6)
  /*  printf("# bmgs_paste  \n");  
  printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  for (int n=16;n<280;n+=24)
  calc_test_paste(n,n,n,1);*/
  printf("# bmgs_fd  \n");  
  printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  for (int n=16;n<280;n+=24){
    calc_test_fd(n,n,n,1);
  }
  for (int n=16;n<140;n+=24)
    calc_test_fd(n,n,n,16);
  for (int n=1;n<=128;n*=2){
    calc_test_fd(40,40,40,n);
    calc_test_fd(80,80,80,n);
  }
  printf("# bmgs_fdz  \n");  
  printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  for (int n=16;n<280;n+=24)
    calc_test_fdz(n,n,n,1);
  printf("# bmgs_relax  \n");  
  printf("# N \t\t CPU Mpoint/s \t GPU Mpoint/s \t Speed-up\n");  
  for (int n=16;n<280;n+=24)
    calc_test_relax(n,n,n);
  
}


