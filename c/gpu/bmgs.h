#ifndef GPU_BMGS_H
#define GPU_BMGS_H

#include "gpu.h"
#include "gpu-complex.h"

int bmgs_fd_boundary_test(const bmgsstencil_gpu* s, int boundary,
                          int ndouble);

bmgsstencil_gpu bmgs_stencil_to_gpu(bmgsstencil *s);

void bmgs_fd_cuda_gpu(const bmgsstencil_gpu* s, const double* adev,
                      double* bdev, int boundary, int blocks,
                      gpuStream_t stream);

void bmgs_relax_cuda_gpu(const int relax_method, const bmgsstencil_gpu* s,
                         double* adev, double* bdev, const double* src,
                         const double w, int boundary, gpuStream_t stream);

void bmgs_cut_cuda_gpu(const double* a, const int n[3], const int c[3],
               double* b, const int m[3],int blocks, gpuStream_t stream);

void bmgs_paste_cuda_gpu(const double* a, const int n[3],
                         double* b, const int m[3], const int c[3],
                         int blocks, gpuStream_t stream);

void bmgs_paste_zero_cuda_gpu(const double* a, const int n[3],
                              double* b, const int m[3], const int c[3],
                              int blocks, gpuStream_t stream);

void bmgs_translate_cuda(double* a, const int sizea[3], const int size[3],
                         const int start1[3], const int start2[3],
                         enum gpuMemcpyKind kind);

void bmgs_translate_cuda_gpu(double* a, const int sizea[3], const int size[3],
                             const int start1[3], const int start2[3],
                             int blocks, gpuStream_t stream);

void bmgs_restrict_cuda_gpu(int k, double* a, const int n[3], double* b,
                            const int nb[3], int blocks);

double bmgs_restrict_cuda_cpu(int k, double* a, const int n[3], double* b,
                              int blocks);

void bmgs_interpolate_cuda_gpu(int k, int skip[3][2],
                               const double* a, const int n[3],
                               double* b, const int sizeb[3],
                               int blocks);

// complex routines:
void bmgs_fd_cuda_gpuz(const bmgsstencil_gpu* s, const gpuDoubleComplex* adev,
                       gpuDoubleComplex* bdev, int boundary, int blocks,
                       gpuStream_t stream);

void bmgs_cut_cuda_gpuz(const gpuDoubleComplex* a, const int n[3],
                        const int c[3], gpuDoubleComplex* b, const int m[3],
                        gpuDoubleComplex, int blocks, gpuStream_t stream);

void bmgs_paste_cuda_gpuz(const gpuDoubleComplex* a, const int n[3],
                          gpuDoubleComplex* b, const int m[3], const int c[3],
                          int blocks, gpuStream_t stream);

void bmgs_paste_zero_cuda_gpuz(const gpuDoubleComplex* a, const int n[3],
                               gpuDoubleComplex* b, const int m[3],
                               const int c[3], int blocks,
                               gpuStream_t stream);

void bmgs_translate_cudaz(gpuDoubleComplex* a, const int sizea[3],
                          const int size[3],  const int start1[3],
                          const int start2[3], gpuDoubleComplex,
                          enum gpuMemcpyKind kind);

void bmgs_translate_cuda_gpuz(gpuDoubleComplex* a, const int sizea[3],
                              const int size[3], const int start1[3],
                              const int start2[3], gpuDoubleComplex,
                              int blocks, gpuStream_t stream);

void bmgs_restrict_cuda_gpuz(int k, gpuDoubleComplex* a, const int n[3],
                             gpuDoubleComplex* b, const int nb[3],
                             int blocks);

void bmgs_interpolate_cuda_gpuz(int k, int skip[3][2],
                                const gpuDoubleComplex* a, const int n[3],
                                gpuDoubleComplex* b, const int sizeb[3],
                                int blocks);

void mdotu_cuda_gpu(const double* a_gpu, const double* b_gpu,
                    double* result, int n, int nvec);

void reducemap_dotuz(const gpuDoubleComplex* a_gpu,
                     const gpuDoubleComplex* b_gpu, gpuDoubleComplex* result,
                     int n, int nvec);

#endif
