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
#define BLOCK_SIZEY 4
#define GRID_SIZE_MAX  128
#define GRID_SIZE_MIN  16
#define MAXCOEFS  32
#define MAXJ      8


#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

#define CUDA_PAD(lim,Npad)						\
  (  (((lim)<=((Npad))) ) ? (0) : (((Npad)-((lim)%(Npad)))%(Npad)))

__constant__ long c_n[3];
__constant__ long c_j[3];
__constant__ long c_offsets[MAXCOEFS];
__constant__ double c_coefs[MAXCOEFS];
//__constant__ long c_offsets1[MAXCOEFS];
//__constant__ double c_coefs1[MAXCOEFS];
//__constant__ int  c_ncoefs;

#define r_n c_n
#define r_j c_j
#define r_offsets c_offsets
#define r_coefs c_coefs


__global__ void bmgs_fd_cuda_kernel(int ncoefs,const double* a,double* b)
//__global__ void bmgs_fd_cuda_kernel(const bmgsstencil* s,const double* a,double* b)
{
  
  //  __shared__ double acache[BLOCK_SIZEY][BLOCK_SIZEX+MAXJ];

  //__shared__ int r_offsets[MAXCOEFS];
  //__shared__ double r_coefs[MAXCOEFS];
  
  //  __shared__ bmgsstencil ss;

  //int r_n[3]={c_n[0],c_n[1],c_n[2]};
  //int r_j[3]={c_j[0],c_j[1],c_j[2]};
  //__shared__ int r_n[3];
  //__shared__ int r_j[3];
  

  // __shared__ double bcache[BLOCK_SIZEY][BLOCK_SIZEX][4];

  //  double bcache[4];

  //  int ncoefs=s->ncoefs;
  /*
    for (int c = threadIdx.x+threadIdx.y*blockDim.x; c < 3; c+=blockDim.x*blockDim.y) {
    r_n[c]=c_n[c];
    r_j[c]=c_j[c];
    }*/
  //__syncthreads();  
   /*  if ((threadIdx.x==0) && (threadIdx.y==0)) {
      r_n[0]=s->n[0]; r_n[1]=s->n[1]; n[2]=s->n[2];
      r_j[0]=s->j[0]; r_rj[1]=s->j[1]; j[2]=s->j[2];
      }*/
  //__syncthreads();  
    /*  for (int c = threadIdx.x+threadIdx.y*blockDim.x; c < ncoefs; c+=blockDim.x*blockDim.y)
    //for (int c = 0; c < ncoefs; c++)
    r_coefs[c]=c_coefs[c];
  for (int c = threadIdx.x+threadIdx.y*blockDim.x; c < ncoefs; c+=blockDim.x*blockDim.y)
    //for (int c = 0; c < ncoefs; c++)
    r_offsets[c]=c_offsets[c];
  
  __syncthreads();  
    */
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
  const double *aa/*,*aaa*/;
  double *bb/*,*bbb*/;
  int i0,i1,i2,c/*,xind,yind,xsize,ysize,i0c*/;
  /*    int ai0s=c_j[1]+c_n[1]*(c_j[2]+c_n[2]);
  int ai1s=c_j[2]+c_n[2];
  int bi0s=c_n[1]*c_n[2];
  int bi1s=c_n[2];*/
  /*
    int s0=c_n[0]/t0;
  


    (c_n[0]%t0) ? s0++ : 0;

    int s1=c_n[1]/t1;

    (c_n[1]%t1) ? s1++ : 0;
  
  


    i0s=i0s*s0;
    i1s=i1s*s1;

    int i0e=i0s+s0;
    int i1e=i1s+s1;

    i0e=i0e>c_n[0] ? c_n[0] : i0e;
    i1e=i1e>c_n[1] ? c_n[1] : i1e;


  */
  //  a+=threadIdx.x+blockIdx.x*(c_j[1]+c_n[1]*(c_j[2]+c_n[2]));
  //b+=threadIdx.x+blockIdx.x*c_n[1]*c_n[2];
  //  a+=threadIdx.x+0*(c_j[1]+c_n[1]*(c_j[2]+c_n[2]));
  //b+=threadIdx.x+0*c_n[1]*c_n[2];
  //  for (i0 = blockIdx.x; i0 < c_n[0]; i0+=gridDim.x) {

 

  //  for (int i0 = i0s; i0 < i0e; i0++) {
  // aa=a+blockIdx.y*(r_j[2]+c_n[2]);
  //bb=b+blockIdx.y*c_n[2];
  //    for (i1 = blockIdx.y; i1 < c_n[1]; i1+=gridDim.y) {

   for (i0 = blockIdx.x; i0 < r_n[0]; i0+=gridDim.x) {
    for (i1=blockIdx.y*blockDim.y+threadIdx.y; i1 < r_n[1]; i1+=blockDim.y*gridDim.y) {
      
      //    for (int i1 = i1s; i1 < i1e; i1++) {
      
      //     aa=a+i1*(r_j[2]+r_n[2]);
      //bb=b+i1*r_n[2];
      for (i2=threadIdx.x; i2 < r_n[2]; i2+=blockDim.x) {
	    
	//bb=b+i1*r_n[2]+i2;	  
	//aa=a+i1*(r_j[2]+r_n[2])+i2;
	    
	//	for(i0c=0;i0c<4;i0c++){
	//      for (i2 =threadIdx.x; i2 < r_n[2]; i2+=blockDim.x) {
	//	bb=b+i0*nb[1]*nb[2]+i1*nb[2]+i2;	  
	//aa=a+i0*na[1]*na[2]+i1*na[2]+i2;
	bb=b+i0*r_n[1]*r_n[2]+i1*r_n[2]+i2;
	aa=a+i0*(r_j[1]+r_n[1]*(r_j[2]+r_n[2]))+i1*(r_j[2]+r_n[2])+i2;

	//	aa=a+i0*ai0s+i1*ai1s+i2;
	//bb=b+i0*bi0s+i1*bi1s+i2;
	//x = r_offsets[0] * r_coefs[0];
	//	x = aa[r_offsets[0]] * r_coefs[0];
	x = 0.0;
	//	for (c = 1; c < ncoefs; c++){
#pragma unroll 19
	for (c = 0; c < ncoefs; c++){	  
	  x += aa[r_offsets[c]] * r_coefs[c];
	  __syncthreads();  	  
	}	


	*(bb) = x;
	//	  bb+=c_n[1]*c_n[2];
	//aa+=(c_j[1]+c_n[1]*(c_j[2]+c_n[2]));
	//	  bcache[i0c]=x;
	//}
	/*	for(i0c=0;i0c<4;i0c++){
	    
	  
	}*/
	//	bb+=blockDim.x; 
	//aa+=blockDim.x; 
      }
      
      
    }
    //__syncthreads();  
    //    a+=(c_j[1]+c_n[1]*(c_j[2]+c_n[2]));
    //b+=c_n[1]*c_n[2];
    //    a+=(c_j[1]+c_n[1]*(c_j[2]+c_n[2]))*gridDim.x;
    //b+=c_n[1]*c_n[2]*gridDim.x;
  }  

  /*
    yind=blockIdx.y*blockDim.y+threadIdx.y;
    ysize=blockDim.y*gridDim.y;

    xind=blockIdx.x*blockDim.x+threadIdx.x;
    xsize=blockDim.x*gridDim.x;
  
    a+=xind+yind*(c_j[2]+c_n[2]);
    b+=xind+yind*c_n[2];
  

    for (i0 = 0; i0 < c_n[0]; i0++) {
    for (bb=b,aa=a; bb < b+c_n[1]; bb+=ysize,aa+=ysize) {
    for (bbb=bb,aaa=aa; bbb < bb+c_n[2];bbb+=xsize,aaa+=xsize) {
    x = aaa[c_offsets[0]] * c_coefs[0];
    for (c = 1; c < c_ncoefs; c++){	  
    x += aaa[c_offsets[c]] * c_coefs[c];
	  
    }		
    *(bbb) = x;
    }

    }
    b+=c_n[1]*c_n[2];
    a+=(c_j[1]+c_n[1]*(c_j[2]+c_n[2]));
    } 
  */ 
}

extern "C" {


  
  double bmgs_fd_cuda(const bmgsstencil* s, const double* a, double* b)
  {
    //    bmgsstencil stemp=*s,*sdev;
  
    double *adev,*bdev,*adevp,*bdevp,*bp;
    const double *ap;
    //cudaPitchedPtr bdevp;

    size_t asize,bsize,asizep,bsizep;
    struct timeval  t0, t1; 
    double flops;

    //cudaExtent bext;
    //  long nsmall[3]={1,1,BLOCK_SIZEX};
    //double h[3]={1.0, 1.0, 1.0};


    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];
  



    /*
    offsets=(long *)malloc(s->ncoefs*sizeof(long));
    coefs=(double *)malloc(s->ncoefs*sizeof(double));
    memcpy(offsets,s->offsets,s->ncoefs*sizeof(long));
    memcpy(coefs,s->coefs,s->ncoefs*sizeof(double));
    
    for(int x = 0; x < s->ncoefs; x++){
	for(int y = 0; y < s->ncoefs-1; y++)
	  if(abs(offsets[y]) > abs(offsets[y+1])) {
	    long offset= offsets[y+1];
	    offsets[y+1] = offsets[y];
	    offsets[y] = offset;
	    double coef= coefs[y+1];
	    coefs[y+1] = coefs[y];
	    coefs[y] = coef;
	  }
    }
    */
    
    /*    cudaMalloc(&(stemp.coefs),sizeof(double)*(s->ncoefs));
    cudaMemcpy(stemp.coefs,s->coefs,sizeof(double)*(s->ncoefs),cudaMemcpyHostToDevice);
    cudaMalloc(&(stemp.offsets),sizeof(double)*(s->ncoefs));
    cudaMemcpy(stemp.offsets,s->offsets,sizeof(double)*(s->ncoefs),cudaMemcpyHostToDevice);
    */
    //fprintf(stdout,"0\n");

    /*cudaMalloc(&sdev,sizeof(bmgsstencil));
    cudaMemcpy(sdev,&stemp,sizeof(bmgsstencil),cudaMemcpyHostToDevice);
    */
    //fprintf(stdout,"1\n");
    
    //    cudaMemcpyToSymbol(c_ncoefs,&(s->ncoefs),sizeof(int))



    cudaMemcpyToSymbol(c_n,s->n,sizeof(long)*3);
    cudaMemcpyToSymbol(c_j,s->j,sizeof(long)*3);
    cudaMemcpyToSymbol(c_offsets,s->offsets,sizeof(long)*s->ncoefs);
    cudaMemcpyToSymbol(c_coefs,s->coefs,sizeof(double)*s->ncoefs);


    cudaMalloc(&adev,sizeof(double)*asize);


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

  
    //  fprintf(stdout,"4 \n");

    //  dim3 dimBlock(MIN(BLOCK_SIZE,s->n[2]),1); 
    //dim3 dimGrid(MAX(MIN(s->n[0]/4,GRID_SIZE_MAX),GRID_SIZE_MIN),
    //	       MAX(MIN(s->n[1]/4,GRID_SIZE_MAX),GRID_SIZE_MIN)); 

    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(MAX(s->n[0]/2,1),MAX(s->n[1]/BLOCK_SIZEY,1));
    //  dim3 dimGrid(GRID_SIZE, GRID_SIZE); 
    fprintf(stdout,"j: %d x %d x %d\t",s->j[0],s->j[1],s->j[2]);
    fprintf(stdout,"array: %d x %d x %d\t",s->n[0],s->n[1],s->n[2]);
    fprintf(stdout,"block: %d x %d\t grid: %d x %d\n",BLOCK_SIZEX,BLOCK_SIZEY,MAX(s->n[0]/(2),1),MAX(s->n[1]/BLOCK_SIZEY,1));
  
    ap=a;
    bp=b;
    adevp=adev;
    bdevp=bdev;
    asizep=asize;
    bsizep=bsize;
  
   
    cudaMemcpy(adevp,ap,sizeof(double)*asizep,cudaMemcpyHostToDevice);
    gettimeofday(&t0,NULL);  
    
    //cudaMemcpy(adevp,ap,sizeof(double)*asizep,cudaMemcpyHostToDevice);
    
    
    //    bmgs_fd_cuda_kernel<<<dimGrid, dimBlock, 0>>>(sdev,adevp+(s->j[0]+s->j[1]+s->j[2])/2,bdevp,MIN(nstep0,s->n[0]-i0),s->n[1],s->n[2]);
    
    bmgs_fd_cuda_kernel<<<dimGrid, dimBlock, 0>>>(s->ncoefs,adevp+(s->j[0]+s->j[1]+s->j[2])/2,bdevp);
    
    cudaThreadSynchronize();  
    gettimeofday(&t1,NULL);
    //  fprintf(stdout,"5 \n");
    cudaMemcpy(bp,bdevp,sizeof(double)*bsizep,cudaMemcpyDeviceToHost);
    
    //cudaMemcpy(bp,bdevp,sizeof(double)*bsizep,cudaMemcpyDeviceToHost);
  

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

    //free(offsets);
    //free(coefs);
    /*    cudaFree(stemp.coefs);
    cudaFree(stemp.offsets);
    cudaFree(sdev);*/
    cudaFree(adev);
    cudaFree(bdev);
    //fprintf(stdout,"7\n");

    //    flops=2*s->ncoefs*bsize/(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 
    flops=(t1.tv_sec*1.0+t1.tv_usec/1000000.0-t0.tv_sec*1.0-t0.tv_usec/1000000.0); 

    return flops;
  
  
  }
  
}
