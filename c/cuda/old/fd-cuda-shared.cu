// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <sys/types.h>
#include <sys/time.h>


//extern "C" {
  typedef struct
  {
    int ncoefs;
    double* coefs;
    long* offsets;
    long n[3];
    long j[3];
  } bmgsstencil;
  

//}

extern "C" {
bmgsstencil bmgs_laplace(int k, double scale, const double h[3], const long n[3]);
}

// includes, project
//#include <cutil_inline.h>


#define BLOCK_SIZEX 16
#define BLOCK_SIZEY 1
#define GRID_SIZE_MAX  128
#define GRID_SIZE_MIN  16
#define MAXCOEFS  32
#define MAX_J     6
#define AS_X     (MAX_J+1)
#define AS_Y     (MAX_J+1)
#define AS_Z     (BLOCK_SIZEX+MAX_J)

#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

__global__ void bmgs_fd_cuda_kernel(const bmgsstencil* s,const double* a,double* b)
{
  __shared__ long offsets[MAXCOEFS];
  __shared__ double coefs[MAXCOEFS];
  
  __shared__ bmgsstencil ss;


  __shared__ double as[AS_X][AS_Y][AS_Z];

  
  double *asp;
  
  asp=&as[0][0][0];
  
  if (threadIdx.x==0) {
  // && (threadIdx.y==0)) {
    ss.ncoefs=s->ncoefs;
    ss.offsets=offsets;
    ss.coefs=coefs;
    ss.n[0]=s->n[0]; ss.n[1]=s->n[1]; ss.n[2]=s->n[2];
    ss.j[0]=s->j[0]; ss.j[1]=s->j[1]; ss.j[2]=s->j[2];

  }
  __syncthreads();  
  for (int c = threadIdx.x+threadIdx.y*blockDim.x; c < ss.ncoefs; c+=blockDim.x*blockDim.y)
    //for (int c = 0; c < ss.ncoefs; c++)
    coefs[c]=s->coefs[c];
  for (int c = threadIdx.x+threadIdx.y*blockDim.x; c < ss.ncoefs; c+=blockDim.x*blockDim.y)
    //for (int c = 0; c < ss.ncoefs; c++)
    offsets[c]=s->offsets[c];
  __syncthreads();  
  

  /*  char *bptr=(char*)b.ptr;
  size_t bpitch=b.pitch;
  */
  
  /*
  int i0s=blockIdx.x;
  int i1s=blockIdx.y;

  int t0=gridDim.x;
  int t1=gridDim.y;
  */
  /*
  int i0s=blockIdx.x * blockDim.x + threadIdx.x;
  int i1s=blockIdx.y * blockDim.y + threadIdx.y;

  int t0=blockDim.x*gridDim.x;
  int t1=blockDim.y*gridDim.y;
  */

  double x;
  const double *aa;
  double *bb;
  int i0,i1,i2,c;
  int ias0,ias1,ias2;
  /*
  int s0=ss.n[0]/t0;
  


  (ss.n[0]%t0) ? s0++ : 0;

  int s1=ss.n[1]/t1;

  (ss.n[1]%t1) ? s1++ : 0;
  
  


  i0s=i0s*s0;
  i1s=i1s*s1;

  int i0e=i0s+s0;
  int i1e=i1s+s1;

  i0e=i0e>ss.n[0] ? ss.n[0] : i0e;
  i1e=i1e>ss.n[1] ? ss.n[1] : i1e;


  */
  //  a+=threadIdx.x+blockIdx.x*(ss.j[1]+ss.n[1]*(ss.j[2]+ss.n[2]));
  //b+=threadIdx.x+blockIdx.x*ss.n[1]*ss.n[2];
  //  a+=threadIdx.x+0*(ss.j[1]+ss.n[1]*(ss.j[2]+ss.n[2]));
  //b+=threadIdx.x+0*ss.n[1]*ss.n[2];
  //  for (i0 = blockIdx.x; i0 < ss.n[0]; i0+=gridDim.x) {
  for (i0 = 0; i0 < ss.n[0]; i0++) {
    //  for (int i0 = i0s; i0 < i0e; i0++) {
    // aa=a+blockIdx.y*(ss.j[2]+ss.n[2]);
    //bb=b+blockIdx.y*ss.n[2];
    //    for (i1 = blockIdx.y; i1 < ss.n[1]; i1+=gridDim.y) {
    for (i1=blockIdx.y*blockDim.y+threadIdx.y; i1 < ss.n[1]; i1+=blockDim.y*gridDim.y) {
      //    for (int i1 = i1s; i1 < i1e; i1++) {
      aa=a+i0*(ss.j[1]+ss.n[1]*(ss.j[2]+ss.n[2]))+i1*(ss.j[2]+ss.n[2]);
      bb=b+i0*ss.n[1]*ss.n[2]+i1*ss.n[2];
      //     aa=a+i1*(ss.j[2]+ss.n[2]);
      //bb=b+i1*ss.n[2];
      for (i2=blockIdx.x*blockDim.x+threadIdx.x; i2 < ss.n[2]; i2+=blockDim.x*gridDim.x) {
	//      for (i2 =threadIdx.x; i2 < ss.n[2]; i2+=blockDim.x) {

	ias0=MAX_J;
	for (ias1=0;ias1<AS_Y;ias1++){
	  for (ias2=threadIdx.x;ias2<AS_Z;ias2+=blockDim.x) {
	    asp[ias0*AS_Z*AS_Y+ias1*AS_Z+ias2]=*(a+(i0+ias0-MAX_J/2)*(ss.j[1]+ss.n[1]*(ss.j[2]+ss.n[2]))+(i1+ias1-MAX_J/2)*(ss.j[2]+ss.n[2])+i2-threadIdx.x+ias2-MAX_J/2);
	    
	  }
	}
	
	//	__syncthreads();  
	/*	x = aa[offsets[0]+i2] * coefs[0];
	for (c = 1; c < ss.ncoefs; c++){	  
	  x += aa[offsets[c]+i2] * coefs[c];
	  
	}	
	*(bb+i2) = x;*/
	//	bb+=blockDim.x; 
	//aa+=blockDim.x; 
      }
      
    }
    //__syncthreads();  
    //    a+=(ss.j[1]+ss.n[1]*(ss.j[2]+ss.n[2]));
    //b+=ss.n[1]*ss.n[2];
    //    a+=(ss.j[1]+ss.n[1]*(ss.j[2]+ss.n[2]))*gridDim.x;
    //b+=ss.n[1]*ss.n[2]*gridDim.x;
  }  
}

extern "C" {

double bmgs_fd_cuda(const bmgsstencil* s, const double* a, double* b)
{
  bmgsstencil stemp=*s,*sdev,stemp2,*ssmalldev;
  
  double *adev,*bdev;
  cudaPitchedPtr bdevp;

  size_t asize,bsize;
  struct timeval  t0, t1; 
  double flops;
  cudaExtent bext;
  long nsmall[3]={1,1,BLOCK_SIZEX};
  double h[3]={1.0, 1.0, 1.0};

  bmgsstencil ssmall=bmgs_laplace(7,1.0,h,nsmall);

  fprintf(stdout,"%ld\t", ssmall.ncoefs);
  for(int i = 0; i < ssmall.ncoefs; ++i)
    fprintf(stdout,"(%lf %ld)\t", ssmall.coefs[i], ssmall.offsets[i]);
  fprintf(stdout,"\n%ld %ld %ld %ld %ld %ld\n",ssmall.j[0],ssmall.j[1],ssmall.j[2],ssmall.n[0],ssmall.n[1],ssmall.n[2]);

  fprintf(stdout,"%ld\n",sizeof(double)*(ssmall.j[0]+ssmall.n[0]*(ssmall.j[1]+ssmall.n[1]*(ssmall.n[2]+ssmall.j[2]))));


  asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
  bsize=s->n[0]*s->n[1]*s->n[2];
  
  

  cudaMalloc(&(stemp.coefs),sizeof(double)*(s->ncoefs));
  cudaMemcpy(stemp.coefs,s->coefs,sizeof(double)*(s->ncoefs),cudaMemcpyHostToDevice);
  cudaMalloc(&(stemp.offsets),sizeof(double)*(s->ncoefs));
  cudaMemcpy(stemp.offsets,s->offsets,sizeof(double)*(s->ncoefs),cudaMemcpyHostToDevice);
  

  //fprintf(stdout,"0\n");

  cudaMalloc(&sdev,sizeof(bmgsstencil));
  cudaMemcpy(sdev,&stemp,sizeof(bmgsstencil),cudaMemcpyHostToDevice);

  //fprintf(stdout,"1\n");

  cudaMalloc(&adev,sizeof(double)*asize);
  cudaMemcpy(adev,a,sizeof(double)*asize,cudaMemcpyHostToDevice);

  //  fprintf(stdout,"3\n");

  cudaMalloc(&bdev,sizeof(double)*bsize);

  /*
  bext=make_cudaExtent(s->n[2]*sizeof(double),s->n[1],s->n[0]);
  cudaMalloc3D(&bdev,bext);
  */
  //cudaMemset(bdev,0,sizeof(double)*bsize);
  /*  fprintf(stdout,"1111\n");

fprintf(stdout,"pitch %zd\n",bdev.pitch);
  fprintf(stdout,"2222\n");
  */
  adev += (s->j[0] + s->j[1] + s->j[2]) / 2;  
  
  //  fprintf(stdout,"4 \n");

  //  dim3 dimBlock(MIN(BLOCK_SIZE,s->n[2]),1); 
  //dim3 dimGrid(MAX(MIN(s->n[0]/4,GRID_SIZE_MAX),GRID_SIZE_MIN),
  //	       MAX(MIN(s->n[1]/4,GRID_SIZE_MAX),GRID_SIZE_MIN)); 

  dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
  dim3 dimGrid(MAX(s->n[2]/BLOCK_SIZEX,1),MAX(s->n[1]/BLOCK_SIZEY,1));

  //  dim3 dimGrid(GRID_SIZE, GRID_SIZE); 
  gettimeofday(&t0,NULL);  
  bmgs_fd_cuda_kernel<<<dimGrid, dimBlock>>>(sdev,adev,bdev);
  cudaThreadSynchronize();
  gettimeofday(&t1,NULL);
  //  fprintf(stdout,"5 \n");

  cudaMemcpy(b,bdev,sizeof(double)*bsize,cudaMemcpyDeviceToHost);
  /*
  cudaMemcpy3DParms bParms = {0};
  bParms.srcPtr=bdev;
  bParms.extent=bext;
  bParms.srcPos=make_cudaPos(0,0,0);
  bParms.dstPtr=make_cudaPitchedPtr((char*)b,s->n[2]*sizeof(double),s->n[2]*sizeof(double),s->n[1]);
  bParms.dstPos=make_cudaPos(0,0,0);
  bParms.kind=cudaMemcpyDeviceToHost;

  cudaMemcpy3D(&bParms);
  */
  //  fprintf(stdout,"6\n");
  cudaFree(stemp.coefs);
  cudaFree(stemp.offsets);
  cudaFree(sdev);
  cudaFree(adev);
  cudaFree(bdev);
  //fprintf(stdout,"7\n");

  flops=2*s->ncoefs*bsize/(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 

  return flops;
  
  
}
  
}
