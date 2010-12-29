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

#include "../../bmgs/bmgs.h"
#include "../gpaw-cuda.h"



int main(void)
{
  bmgsstencil s,s2;
   double h[3]={1.0, 1.0, 1.0};
   //long   n[3]={6,6,6};
   // long   n[3]={512,512,512};
  //   long   n[3]={71,75,75};
  //long   n[3]={143,151,151};
  int ntimes=1;
  long   n[3]={256,256,256};
    double *a,*b,*a_cuda,*b_cuda,error=0,*a_cuda2,*b_cuda2,*src,*src_cuda;
  size_t asize,bsize;

  struct timeval  t0, t1, t2; 
  double flops,flops2,flops3;

  printf("double complex %d  cuDoubleComplex %d\n",sizeof(double complex),sizeof(cuDoubleComplex));

  s=bmgs_laplace(7,1.0,h,n);
  s2=bmgs_laplace(7,1.0,h,n);

  asize=s.j[0]+s.n[0]*(s.j[1]+s.n[1]*(s.n[2]+s.j[2]));
  bsize=s.n[0]*s.n[1]*s.n[2];

  
  a=malloc(asize*sizeof(double));
  a_cuda=malloc(asize*sizeof(double));
  a_cuda2=malloc(asize*sizeof(double));
  b=malloc(bsize*sizeof(double));  
  b_cuda=malloc(bsize*sizeof(double));  
  b_cuda2=malloc(bsize*sizeof(double)); 
  src=malloc(bsize*sizeof(double));  
  src_cuda=malloc(bsize*sizeof(double));   


  /*  
  cudaMallocHost((void**)&a,sizeof(*a)*asize);
  cudaMallocHost((void**)&a_cuda,sizeof(*a)*asize);
  cudaMallocHost((void**)&a_cuda2,sizeof(*a)*asize);
  cudaMallocHost((void**)&b,sizeof(*b)*bsize);
  cudaMallocHost((void**)&b_cuda,sizeof(*b_cuda)*bsize);
  cudaMallocHost((void**)&b_cuda2,sizeof(*b_cuda)*bsize);
  */  
  
  fprintf(stdout,"%ld\t", s.ncoefs);
  for(int i = 0; i < s.ncoefs; ++i)
    fprintf(stdout,"(%lf %ld)\t", s.coefs[i], s.offsets[i]);
  fprintf(stdout,"\n%ld %ld %ld %ld %ld %ld\n",s.j[0],s.j[1],s.j[2],s.n[0],s.n[1],s.n[2]);
  
    cudaSetDevice(2); 
    srand((unsigned int) time(NULL));
    //srand(0);

  for (int i=0;i<asize;i++){
    a[i]=rand()/(double)RAND_MAX;
    a_cuda[i]=a[i];
  }
  bmgs_fd_cuda_cpu(&s2,a_cuda2,b_cuda2);

  gettimeofday(&t0,NULL);
  
  for (int i=0;i<ntimes;i++) bmgs_fd(&s,a,b);
  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_fd_cuda_cpu(&s,a_cuda,b_cuda);
  gettimeofday(&t2,NULL);
  //flops=2*s.ncoefs*bsize/(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  //flops2=2*s.ncoefs*bsize/(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);

  flops=(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  flops2=(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);
  //  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f\n",s.n[0],s.n[1],s.n[2], flops/1000000000.0,flops2/1000000000.0,flops3/1000000000.0);
  printf("bmgs_fd %d times:\n",ntimes);
  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f ms\n",s.n[0],s.n[1],s.n[2],1000*flops,1000*flops2,1000*flops3);
  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f Gflops\n",s.n[0],s.n[1],s.n[2],2*s.ncoefs*bsize/(1000000000.0*flops),s.n[2],2*s.ncoefs*bsize/(1000000000.0*flops2),s.n[2],2*s.ncoefs*bsize/(1000000000.0*flops3));
  /*  for (int n0=0;n0<n[0];n0++){
    for (int n1=0;n1<n[1];n1++){
      fprintf(stdout,"%lf ",b[n[2]*n[1]*n0+n[2]*n1]);
    }    
    fprintf(stdout,"\n");
  }
  */
  /*  for (int i=0;i<bsize;i++){
    error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
    }*/
  for (int i0=0;i0<s.n[0];i0++){
    for (int i1=0;i1<s.n[1]-0;i1++){
      for (int i2=0;i2<s.n[2]-0;i2++){
	long i=i0*s.n[1]*s.n[2]+i1*s.n[2]+i2; 
	error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
      }
    }
  }
  
  error=error/(double)bsize;
  fprintf(stdout,"mean sqr error %lf\n",error);

  for (int i=0;i<asize;i++){
    a[i]=rand()/(double)RAND_MAX;
    a_cuda[i]=a[i];
  }
  int ni[3]={n[0],n[1],n[2]};
  int c[3]={0,0,0};
  
  gettimeofday(&t0,NULL);
  
  for (int i=0;i<ntimes;i++) bmgs_paste(a,ni,b,ni,c);
  gettimeofday(&t1,NULL);
  flops3=0;
  for (int i=0;i<ntimes;i++) flops3+=bmgs_paste_cuda_cpu(a,ni,b,ni,c);
  gettimeofday(&t2,NULL);
  //flops=2*s.ncoefs*bsize/(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  //flops2=2*s.ncoefs*bsize/(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);

  flops=(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  flops2=(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);
  //  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f\n",s.n[0],s.n[1],s.n[2], flops/1000000000.0,flops2/1000000000.0,flops3/1000000000.0);
  printf("bmgs_paste %d times:\n",ntimes);
  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f ms\n",s.n[0],s.n[1],s.n[2],1000*flops,1000*flops2,1000*flops3);

  /*  for (int n0=0;n0<n[0];n0++){
    for (int n1=0;n1<n[1];n1++){
      fprintf(stdout,"%lf ",b[n[2]*n[1]*n0+n[2]*n1]);
    }    
    fprintf(stdout,"\n");
  }
  */
  /*  for (int i=0;i<bsize;i++){
    error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
    }*/
  for (int i0=0;i0<s.n[0];i0++){
    for (int i1=0;i1<s.n[1]-0;i1++){
      for (int i2=0;i2<s.n[2]-0;i2++){
	long i=i0*s.n[1]*s.n[2]+i1*s.n[2]+i2; 
	error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
      }
    }
  }
  
  error=error/(double)bsize;
  fprintf(stdout,"mean sqr error %lf\n",error);






  for (int i=0;i<asize;i++){
    a[i]=rand()/(double)RAND_MAX;
    a_cuda[i]=a[i];
  }  
  for (int i=0;i<bsize;i++){
    src[i]=rand()/(double)RAND_MAX;
    b[i]=rand()/(double)RAND_MAX;
    b_cuda[i]=b[i];
    src_cuda[i]=src[i];
  }

  int relax_method=2;
  s.offsets[0]=0;
  
  gettimeofday(&t0,NULL);
  bmgs_relax(relax_method,&s,a,b,src, 0.67);
  gettimeofday(&t1,NULL);
  //bmgs_relax(2,&s,a_cuda,b_cuda,src_cuda, 0.67);
  flops3=bmgs_relax_cuda_cpu(relax_method,&s,a_cuda,b_cuda,src_cuda, 0.67);
  gettimeofday(&t2,NULL);

  flops=(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  flops2=(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);
  printf("bmgs_relax:\n");
  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f ms\n",s.n[0],s.n[1],s.n[2],1000*flops,1000*flops2,1000*flops3);

  error=0;
  for (int i0=0;i0<s.n[0];i0++){
    for (int i1=0;i1<s.n[1]-0;i1++){
      for (int i2=0;i2<s.n[2]-0;i2++){
	long i=i0*s.n[1]*s.n[2]+i1*s.n[2]+i2; 
	error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
      }
    }
  }
  error=error/(double)bsize;
  fprintf(stdout,"mean sqr error %lf\n",error);


  for (int i=0;i<bsize;i++){
    a[i]=rand()/(double)RAND_MAX;
    src[i]=rand()/(double)RAND_MAX;
    b[i]=rand()/(double)RAND_MAX;
    a_cuda[i]=a[i];    
    b_cuda[i]=b[i];
    src_cuda[i]=src[i];
  }

  int k=2;
  int nre[3]={s.n[0],s.n[1],s.n[2]};

  gettimeofday(&t0,NULL);
  bmgs_restrict(k, a, nre, b, src);
  gettimeofday(&t1,NULL);
  //bmgs_relax(2,&s,a_cuda,b_cuda,src_cuda, 0.67);
  flops3=bmgs_restrict_cuda_cpu(k, a_cuda, nre, b_cuda, src_cuda);
  //bmgs_restrict(k, a_cuda, nre, b_cuda, src_cuda);
  gettimeofday(&t2,NULL);

  flops=(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  flops2=(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);
  printf("bmgs_restrict:\n");
  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f ms\n",s.n[0],s.n[1],s.n[2],1000*flops,1000*flops2,1000*flops3);

  error=0;
  for (int i0=0;i0<s.n[0];i0++){
    for (int i1=0;i1<s.n[1]-0;i1++){
      for (int i2=0;i2<s.n[2]-0;i2++){
	long i=i0*s.n[1]*s.n[2]+i1*s.n[2]+i2; 
	error+=(a[i]-a_cuda[i])*(a[i]-a_cuda[i]);
	error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
	error+=(src[i]-src_cuda[i])*(src[i]-src_cuda[i]);
      }
    }
  }
  error=error/3*(double)bsize;
  fprintf(stdout,"mean sqr error %lf\n",error);
  /*  
  for (int i=0;i<bsize;i++){
    a[i]=rand()/(double)RAND_MAX;
    src[i]=rand()/(double)RAND_MAX;
    b[i]=rand()/(double)RAND_MAX;
    a_cuda[i]=a[i];    
    b_cuda[i]=b[i];
    src_cuda[i]=src[i];
  }
  int nre2[3]={n[0]/4,n[1]/4,n[2]/4};
  int skip[3][2]={{0%2,n[0]%2},{0%2,n[1]%2},{0%2,n[2]%2}};
  fprintf(stdout,"mdd\n");
  gettimeofday(&t0,NULL);
  bmgs_interpolate(k, skip,a, nre2, b, src);
  gettimeofday(&t1,NULL);
  fprintf(stdout,"mdd 2\n");
  //bmgs_relax(2,&s,a_cuda,b_cuda,src_cuda, 0.67);
  flops3=bmgs_interpolate_cuda_cpu(k, skip,a_cuda, nre2, b_cuda, src_cuda);
  //bmgs_restrict(k, a_cuda, nre, b_cuda, src_cuda);
  fprintf(stdout,"mdd 3\n");
  gettimeofday(&t2,NULL);

  flops=(t1.tv_sec+t1.tv_usec/1000000.0-t0.tv_sec-t0.tv_usec/1000000.0); 
  flops2=(t2.tv_sec+t2.tv_usec/1000000.0-t1.tv_sec-t1.tv_usec/1000000.0);
  printf("bmgs_interpolate:\n");
  printf ("%dx%dx%d  \t CPU: %f\t GPU: %f\t GPU NOMEMTR: %f ms\n",s.n[0],s.n[1],s.n[2],1000*flops,1000*flops2,1000*flops3);

  error=0;
  for (int i0=0;i0<s.n[0];i0++){
    for (int i1=0;i1<s.n[1]-0;i1++){
      for (int i2=0;i2<s.n[2]-0;i2++){
	long i=i0*s.n[1]*s.n[2]+i1*s.n[2]+i2; 
	error+=(a[i]-a_cuda[i])*(a[i]-a_cuda[i]);
	error+=(b[i]-b_cuda[i])*(b[i]-b_cuda[i]);
	error+=(src[i]-src_cuda[i])*(src[i]-src_cuda[i]);
      }
    }
  }
  error=error/3*(double)bsize;
  fprintf(stdout,"mean sqr error %lf\n",error);
  */

  /*
  cudaFree(a);
  cudaFree(a_cuda2);
  cudaFree(a_cuda);
  cudaFree(b);
  cudaFree(b_cuda);
  cudaFree(b_cuda2);
  */
    
  free(a);
  free(b);
  free(a_cuda);
  free(b_cuda);
  free(a_cuda2);
  free(b_cuda2);
  return 0;
}
