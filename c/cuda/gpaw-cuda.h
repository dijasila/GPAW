
#ifndef GPAW_CUDA_H
#define GPAW_CUDA_H

#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuComplex.h>

#include"gpaw-cuda-common.h"

int bmgs_fd_boundary_test(const bmgsstencil_gpu* s, int boundary, int ndouble);
bmgsstencil_gpu bmgs_stencil_to_gpu(bmgsstencil *s);

double bmgs_fd_cuda_cpu(const bmgsstencil* s, const double* a, double* b,
			int boundary,int blocks);
void bmgs_fd_cuda_gpu(const bmgsstencil_gpu* s, const double* adev, 
		      double* bdev,int boundary,int blocks,cudaStream_t stream);


double bmgs_fd_cuda_cpu_bc(const bmgsstencil* s, const double* a, double* b);
void bmgs_fd_cuda_gpu_bc(const bmgsstencil_gpu* s, const double* adev, 
			 double* bdev,int blocks);

double bmgs_relax_cuda_cpu(const int relax_method, const bmgsstencil* s, 
			   double* a, double* b,const double* src, 
			   const double w);
void bmgs_relax_cuda_gpu(const int relax_method, const bmgsstencil_gpu* s,
			 double* adev, double* bdev,const double* src, 
			 const double w,int boundary,cudaStream_t stream);
/*void bmgs_relax_cuda_gpu(const int relax_method, const bmgsstencil_gpu* s,
			 double* adev, double* bdev,const double* src, 
			 const double w,int boundary,cudaStream_t stream);*/

double bmgs_relax_cuda_cpu_bc(const int relax_method, const bmgsstencil* s,
			      double* a, double* b,const double* src, 
			      const double w);
void bmgs_relax_cuda_gpu_bc(const int relax_method, const bmgsstencil_gpu* s,
			    double* adev, double* bdev,const double* src,
			    const double w);

void bmgs_cut_cuda(const double* a, const int n[3], const int c[3],
		   double* b, const int m[3],enum cudaMemcpyKind kind);

void bmgs_cut_cuda_gpu(const double* a, const int n[3], const int c[3],
		       double* b, const int m[3],int blocks,cudaStream_t stream);

void bmgs_paste_cuda(const double* a, const int n[3],
		     double* b, const int m[3], const int c[3],
		     enum cudaMemcpyKind kind);
void bmgs_paste_cuda_gpu(const double* a, const int n[3],
			 double* b, const int m[3], const int c[3],
			 int blocks,cudaStream_t stream);

void bmgs_paste_zero_cuda_gpu(const double* a, const int n[3],
			      double* b, const int m[3], const int c[3],
			      int blocks,cudaStream_t stream);

double bmgs_paste_cuda_cpu(const double* a, const int n[3],
		     double* b, const int m[3], const int c[3]);
double bmgs_paste_zero_cuda_cpu(const double* a, const int n[3],
		     double* b, const int m[3], const int c[3]);

void bmgs_translate_cuda(double* a, const int sizea[3], const int size[3],
			 const int start1[3], const int start2[3],
			 enum cudaMemcpyKind kind);

void bmgs_translate_cuda_gpu(double* a, const int sizea[3], const int size[3],
			     const int start1[3], const int start2[3],
			     int blocks,cudaStream_t stream);


void bmgs_restrict_cuda_gpu(int k, double* a, const int n[3], double* b, 
			    const int nb[3], int blocks);
double bmgs_restrict_cuda_cpu(int k, double* a, const int n[3], double* b, int blocks);
void bmgs_interpolate_cuda_gpu(int k, int skip[3][2],
			       const double* a, const int n[3],
			       double* b, const int sizeb[3], 
			       int blocks);

double bmgs_interpolate_cuda_cpu(int k, int skip[3][2],
				 const double* a, const int n[3],
				  double* b,int blocks);

// complex routines:
double bmgs_fd_cuda_cpuz(const bmgsstencil* s, const cuDoubleComplex* a, cuDoubleComplex* b,int boundary,int blocks);
void bmgs_fd_cuda_gpuz(const bmgsstencil_gpu* s, const cuDoubleComplex* adev, 
		       cuDoubleComplex* bdev,int boundary,int blocks,cudaStream_t stream);

void bmgs_fd_cuda_gpu_bcz(const bmgsstencil_gpu* s,
			  const cuDoubleComplex* adev,
			  cuDoubleComplex* bdev,int blocks);

void bmgs_cut_cudaz(const cuDoubleComplex* a, const int n[3], const int c[3],
		    cuDoubleComplex* b, const int m[3],
		    enum cudaMemcpyKind kind);
void bmgs_cut_cuda_gpuz(const cuDoubleComplex* a, const int n[3],
			const int c[3], cuDoubleComplex* b, const int m[3],
			cuDoubleComplex,int blocks,cudaStream_t stream);

void bmgs_paste_cudaz(const cuDoubleComplex* a, const int n[3],
		     cuDoubleComplex* b, const int m[3], const int c[3],
		      enum cudaMemcpyKind kind);
void bmgs_paste_cuda_gpuz(const cuDoubleComplex* a, const int n[3],
			  cuDoubleComplex* b, const int m[3], const int c[3],
			  int blocks,cudaStream_t stream);

void bmgs_paste_zero_cuda_gpuz(const cuDoubleComplex* a, const int n[3],
			  cuDoubleComplex* b, const int m[3], const int c[3],
			       int blocks,cudaStream_t stream);

void bmgs_translate_cudaz(cuDoubleComplex* a, const int sizea[3], 
			  const int size[3],  const int start1[3], 
			  const int start2[3],cuDoubleComplex,
			  enum cudaMemcpyKind kind);

void bmgs_translate_cuda_gpuz(cuDoubleComplex* a, const int sizea[3],
			      const int size[3], const int start1[3], 
			      const int start2[3],cuDoubleComplex, int blocks,cudaStream_t stream);

void bmgs_restrict_cuda_gpuz(int k, cuDoubleComplex* a, const int n[3], 
			      cuDoubleComplex* b, const int nb[3], 
			      int blocks);

void bmgs_interpolate_cuda_gpuz(int k, int skip[3][2],
				const cuDoubleComplex* a, const int n[3],
				cuDoubleComplex* b, const int sizeb[3],
				int blocks);

void mdotu_cuda_gpu( const double* a_gpu, const double* b_gpu,double *result,int n,int nvec);
void reducemap_dotuz( const cuDoubleComplex* a_gpu, const cuDoubleComplex* b_gpu,cuDoubleComplex* result,int n,int nvec);

#endif
