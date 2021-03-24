#include "../bmgs/bmgs.h"

void bmgs_paste_cpu(double *a_cpu, const int sizea[3],
                    double *b_cpu, const int sizeb[3],
                    const int startb[3])
{
    bmgs_paste(a_cpu, sizea, b_cpu, sizeb, startb);
}

void bmgs_pastez_cpu(double *a_cpu, const int sizea[3],
                     double *b_cpu, const int sizeb[3],
                     const int startb[3])
{
    bmgs_pastez((const double_complex*) a_cpu, sizea,
                (double_complex*) b_cpu, sizeb, startb);
}

void bmgs_cut_cpu(double *a_cpu, const int sizea[3],
                  const int starta[3],
                  double *b_cpu, const int sizeb[3])
{
    bmgs_cut(a_cpu, sizea, starta, b_cpu, sizeb);
}

void bmgs_cutz_cpu(double *a_cpu, const int sizea[3],
                   const int starta[3],
                   double *b_cpu, const int sizeb[3])
{
    bmgs_cutz((const double_complex*) a_cpu, sizea, starta,
              (double_complex*) b_cpu, sizeb);
}
