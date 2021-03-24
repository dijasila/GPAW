#ifndef GPAW_CUDA_DEBUG_H
#define GPAW_CUDA_DEBUG_H

#define GPAW_MALLOC(T, n) (T*)(malloc((n) * sizeof(T)))

extern "C" void bmgs_paste_cpu(double *a_cpu, const int sizea[3],
                               double *b_cpu, const int sizeb[3],
                               const int startb[3]);
extern "C" void bmgs_pastez_cpu(double *a_cpu, const int sizea[3],
                                double *b_cpu, const int sizeb[3],
                                const int startb[3]);
extern "C" void bmgs_cut_cpu(double *a_cpu, const int sizea[3],
                             const int starta[3],
                             double *b_cpu, const int sizeb[3]);
extern "C" void bmgs_cutz_cpu(double *a_cpu, const int sizea[3],
                              const int starta[3],
                              double *b_cpu, const int sizeb[3]);
#endif
