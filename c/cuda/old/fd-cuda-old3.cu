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
#define MAXCOEFS  24
#define MAXJ      8

#define ACACHE_SIZEX  (BLOCK_SIZEX+MAXJ)    
#define ACACHE_SIZEY  (BLOCK_SIZEY+MAXJ)
#define ACACHE_SIZEZ  (1+MAXJ)


#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

__constant__ long c_n[3];
__constant__ long c_j[3];
__constant__ long c_offsets[MAXCOEFS];
__constant__ double c_coefs[MAXCOEFS];
__constant__ long c_offsetsxy[MAXCOEFS];
__constant__ double c_coefsxy[MAXCOEFS];
__constant__ long c_offsetsz[MAXCOEFS];
__constant__ double c_coefsz[MAXCOEFS];
__constant__ int c_jb[3];
//__constant__ int  c_ncoefs;

#define r_n c_n
#define r_j c_j
#define r_offsets c_offsets
#define r_coefs c_coefs


__global__ void bmgs_fd_cuda_kernel(int ncoefs,int ncoefsxy,int ncoefsz,const double* a,double* b)
{


  int i1=blockIdx.y*blockDim.y+threadIdx.y;
  if (i1>=r_n[1]) return;  
  int i2=blockIdx.x*blockDim.x+threadIdx.x;
  if (i2>=r_n[2]) return;  

  __shared__ double acache[ACACHE_SIZEY][ACACHE_SIZEX];
  double acachez[ACACHE_SIZEZ];
  int i,c/*,xind,yind,xsize,ysize,i0c*/;
  double x;
  const double *aa,*aacache,*aacachez/*,*aaa*/;
  /*double *bb,*bbb*/;
  int rbm=BLOCK_SIZEX*(r_n[2]/BLOCK_SIZEX);
  int blodimy=MIN(blockDim.y,r_n[1]-blockIdx.y*blockDim.y);


  aacache=&acache[0][0]+ACACHE_SIZEX*(threadIdx.y+c_jb[1]/2)+threadIdx.x+r_j[2]/2;
  //  aacachez=&acachez[0]+c_jb[0]/2;
  //  a+=i0*(r_j[1]+r_n[1]*(r_j[2]+r_n[2]))+i1*(r_j[2]+r_n[2]);
  //b+=i0*r_n[1]*r_n[2]+i1*r_n[2];
  a+=i1*(r_j[2]+r_n[2]);
  b+=i1*r_n[2];
  for (int i0=0;i0<r_n[0];i0++){
    
    aa=a+i2-r_j[2]/2-r_j[1]/2;    
#pragma unroll 3
    for (i=threadIdx.y;i<c_jb[1]+blodimy;i+=blodimy){
      acache[i][threadIdx.x]=*(aa);
      if (threadIdx.x<r_j[2])
	acache[i][threadIdx.x+blockDim.x]=*(aa+blockDim.x);
      aa+=blodimy*(r_j[2]+r_n[2]);
    }    
    __syncthreads();  	  
    
    x = 0.0;    
#pragma unroll 13
    for (c = 0; c < ncoefsxy; c++){
      x += aacache[c_offsetsxy[c]] * c_coefsxy[c];
    }	
    aa=a+i2;
    for (c=0;c<=c_jb[0];c++){
      acachez[c]=*(aa+(c-c_jb[0]/2)*(r_j[1]+r_n[1]*(r_j[2]+r_n[2])));
    }
#pragma unroll 6
    for (c = 0; c < ncoefsz; c++){
      x += acachez[c_offsetsz[c]] * c_coefsz[c];
    }	
    
#pragma unroll 1
    for (c = 0; c < ncoefs; c++){	  
      x += aa[r_offsets[c]] * r_coefs[c];
      __syncthreads();  	  
    }	
    
    b[i2] = x;
    
    a+=r_j[1]+r_n[1]*(r_j[2]+r_n[2]);
    b+=r_n[1]*r_n[2];
  }
  return;
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

    long offsets[s->ncoefs],offsetsxy[s->ncoefs],offsetsz[s->ncoefs];
    double coefs[s->ncoefs],coefsxy[s->ncoefs],coefsz[s->ncoefs];
    int jb[3];
    long ncoefs=0,ncoefsxy=0,ncoefsz=0;


    asize=s->j[0]+s->n[0]*(s->j[1]+s->n[1]*(s->n[2]+s->j[2]));
    bsize=s->n[0]*s->n[1]*s->n[2];
    
    jb[2]=s->j[2];
    jb[1]=s->j[1]/(jb[2]+s->n[2]);
    jb[0]=s->j[0]/((jb[1]+s->n[1])*(jb[2]+s->n[2]));

    for(int i = 0; i < s->ncoefs; i++){
      if (abs(s->offsets[i])<=(s->j[2]/2)){
	offsetsxy[ncoefsxy]=s->offsets[i];
	coefsxy[ncoefsxy]=s->coefs[i];
	ncoefsxy++;
      } else if (abs(s->offsets[i])<=(s->j[1]/2)){
	offsetsxy[ncoefsxy]=s->offsets[i]*ACACHE_SIZEX/(s->j[2]+s->n[2]);
	coefsxy[ncoefsxy]=s->coefs[i];
	ncoefsxy++;
      } else if ((s->offsets[i]%(jb[2]+s->n[2])==0) && (s->offsets[i]%((jb[1]+s->n[1])*(jb[2]+s->n[2]))==0)) {
	
	offsetsz[ncoefsz]=jb[0]/2+s->offsets[i]/((jb[1]+s->n[1])*(jb[2]+s->n[2]));
	coefsz[ncoefsz]=s->coefs[i];
	ncoefsz++;
      }
      else{
	offsets[ncoefs]=s->offsets[i];
	coefs[ncoefs]=s->coefs[i];
	ncoefs++;
      }

    }

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
    fprintf(stdout,"%ld\t", ncoefs);
    for(int i = 0; i < ncoefs; ++i)
      fprintf(stdout,"(%lf %ld)\t", coefs[i], offsets[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"%ld\t", ncoefsxy);
    for(int i = 0; i < ncoefsxy; ++i)
      fprintf(stdout,"(%lf %ld)\t", coefsxy[i], offsetsxy[i]);
    fprintf(stdout,"\n");
    fprintf(stdout,"%ld\t", ncoefsz);
    for(int i = 0; i < ncoefsz; ++i)
      fprintf(stdout,"(%lf %ld)\t", coefsz[i], offsetsz[i]);
    fprintf(stdout,"\n");

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
    cudaMemcpyToSymbol(c_offsets,offsets,sizeof(long)*ncoefs);
    cudaMemcpyToSymbol(c_coefs,coefs,sizeof(double)*ncoefs);
    cudaMemcpyToSymbol(c_offsetsxy,offsetsxy,sizeof(long)*ncoefsxy);
    cudaMemcpyToSymbol(c_coefsxy,coefsxy,sizeof(double)*ncoefsxy);
    cudaMemcpyToSymbol(c_offsetsz,offsetsz,sizeof(long)*ncoefsz);
    cudaMemcpyToSymbol(c_coefsz,coefsz,sizeof(double)*ncoefsz);
    cudaMemcpyToSymbol(c_jb,jb,sizeof(int)*3);

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

    int gridy=s->n[1]/BLOCK_SIZEY;
    gridy=(s->n[1]%BLOCK_SIZEY) ? gridy+1 : gridy;

    int gridx=s->n[2]/BLOCK_SIZEX;
    gridx=(s->n[2]%BLOCK_SIZEX) ? gridx+1 : gridx;

    dim3 dimBlock(BLOCK_SIZEX,BLOCK_SIZEY); 
    dim3 dimGrid(gridx,gridy);
    //  dim3 dimGrid(GRID_SIZE, GRID_SIZE); 
    fprintf(stdout,"array: %d x %d x %d\t",s->n[0],s->n[1],s->n[2]);
    fprintf(stdout,"jb: %d x %d x %d\t",jb[0],jb[1],jb[2]);
    fprintf(stdout,"block: %d x %d\t grid: %d x %d\n",BLOCK_SIZEX,BLOCK_SIZEY,gridx,gridy);
    
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
    
    bmgs_fd_cuda_kernel<<<dimGrid, dimBlock, 0>>>(ncoefs,ncoefsxy,ncoefsz,adevp+(s->j[0]+s->j[1]+s->j[2])/2,bdevp);
    
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
