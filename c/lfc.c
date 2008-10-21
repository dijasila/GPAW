#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "spline.h"
#include "bmgs/bmgs.h"

#ifdef GPAW_AIX
#  define dgemm_ dgemm
#endif
int dgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, double *alpha, const double *a, int *lda,
	   const double *b, int *ldb, double *beta,
	   double *c, int *ldc);

PyObject* spline_to_grid(PyObject *self, PyObject *args)
{
  const SplineObject* spline_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* end_c_obj;
  PyArrayObject* pos_v_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyArrayObject* gdcorner_c_obj;
  if (!PyArg_ParseTuple(args, "OOOOOOO", &spline_obj,
                        &beg_c_obj, &end_c_obj, &pos_v_obj, &h_cv_obj,
                        &n_c_obj, &gdcorner_c_obj))
    return NULL; 

  const bmgsspline* spline = (const bmgsspline*)(&(spline_obj->spline));
  long* beg_c = LONGP(beg_c_obj);
  long* end_c = LONGP(end_c_obj);
  double* pos_v = DOUBLEP(pos_v_obj);
  double* h_cv = DOUBLEP(h_cv_obj);
  long* n_c = LONGP(n_c_obj);
  long* gdcorner_c = LONGP(gdcorner_c_obj);

  int l = spline_obj->spline.l;
  double rcut = spline->dr * spline->nbins;

  int ngmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]) *
               (end_c[2] - beg_c[2]));
  double* A_gm = GPAW_MALLOC(double, ngmax * (2 * l + 1));
  
  int nBmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]));
  int* G_B = GPAW_MALLOC(int, 2 * nBmax);

  int nB = 0;
  int ngm = 0;
  int G = n_c[2] * (beg_c[1] - gdcorner_c[1] + n_c[1] 
                    * (beg_c[0] - gdcorner_c[0]));

  for (int g0 = beg_c[0]; g0 < end_c[0]; g0++)
    {
      for (int g1 = beg_c[1]; g1 < end_c[1]; g1++)
        {
	  int g2_beg = -1; // function boundary coordinates
	  int g2_end = -1;
          for (int g2 = beg_c[2]; g2 < end_c[2]; g2++)
            {
              double x = h_cv[0] * g0 + h_cv[1] * g1 + h_cv[2] * g2 - pos_v[0];
              double y = h_cv[3] * g0 + h_cv[4] * g1 + h_cv[5] * g2 - pos_v[1];
              double z = h_cv[6] * g0 + h_cv[7] * g1 + h_cv[8] * g2 - pos_v[2];
              double r2 = x * x + y * y + z * z;
              double r = sqrt(r2);
              if (r < rcut)
                {
		  if (g2_beg < 0)
		    g2_beg = g2; // found boundary
		  g2_end = g2;
                  double Ar = bmgs_splinevalue(spline, r);
                  if (l == 0)
                    A_gm[ngm++] = 0.28209479177387814 * Ar;
                  else
                    {
                      A_gm[ngm++] = 0.48860251190291992 * y * Ar;
                      A_gm[ngm++] = 0.48860251190291992 * z * Ar;
                      A_gm[ngm++] = 0.48860251190291992 * x * Ar;
                    }
                }
            }
	  if (g2_end >= 0)
	    {
	      g2_end++;
	      G_B[nB++] = G + g2_beg;
	      G_B[nB++] = G + g2_end;
	    }
          G += n_c[2];
        }
      G += n_c[2] * (n_c[1] - end_c[1] + beg_c[1]);
    }
  npy_intp gm_dims[2] = {ngm / (2 * l + 1), 2 * l + 1};
  PyArrayObject* A_gm_obj = (PyArrayObject*)PyArray_SimpleNew(2, gm_dims, 
                                                              NPY_DOUBLE);

  memcpy(A_gm_obj->data, A_gm, ngm * sizeof(double));
  free(A_gm);

  npy_intp B_dims[1] = {nB};
  PyArrayObject* G_B_obj = (PyArrayObject*)PyArray_SimpleNew(1, B_dims,
                                                             NPY_INT);
  memcpy(G_B_obj->data, G_B, nB * sizeof(int));
  free(G_B);

  return Py_BuildValue("(OO)", A_gm_obj, G_B_obj);
}


PyObject* calculate_potential_matrix(PyObject *self, PyObject *args)
{
  PyObject* A_Igm_obj;
  const PyArrayObject* vt_sG_obj;
  const PyArrayObject* M_I_obj;
  const PyArrayObject* G_B_obj;
  const PyArrayObject* I_B_obj;
  PyArrayObject* g_I_obj;
  PyArrayObject* I_i_obj;
  PyArrayObject* i_I_obj;
  double dv;
  PyArrayObject* A_gm_obj;
  PyArrayObject* Vt_skMM_obj;

  if (!PyArg_ParseTuple(args, "OOOOOOOOdOO", &A_Igm_obj, &vt_sG_obj, &M_I_obj,
                        &G_B_obj, &I_B_obj, &g_I_obj, &I_i_obj, &i_I_obj,
                        &dv, &A_gm_obj, &Vt_skMM_obj))
    return NULL; 

  const double* vt_sG = (const double*)vt_sG_obj->data;
  const int* M_I = (const int*)M_I_obj->data;
  const int* G_B = (const int*)G_B_obj->data;
  const int* I_B = (const int*)I_B_obj->data;

  int* g_I = (int*)g_I_obj->data;
  int* I_i = (int*)I_i_obj->data;
  int* i_I = (int*)i_I_obj->data;

  double* A1vt_gm = (double*)A_gm_obj->data;
  double* Vt_skMM = (double*)Vt_skMM_obj->data;

  int nM = Vt_skMM_obj->dimensions[2];
  int nB = G_B_obj->dimensions[0];

  int Ga = 0;
  int B = 0;
  int ni = 0;
  while (B < nB)
    {
      int Gb = G_B[B];
      int nG = Gb - Ga;
      if (nG > 0)
        // Do work for [Ga:Gb) range:
        for (int i1 = 0; i1 < ni; i1++)
          {
            int I1 = I_i[i1];
            int M1 = M_I[I1];
            const PyArrayObject* A1_gm_obj = \
              (const PyArrayObject*)PyList_GetItem(A_Igm_obj, I1);
            const double* A1_gm = (const double*)A1_gm_obj->data + g_I[I1];
            int nm1 = A1_gm_obj->dimensions[1];
            
            int gm1 = 0;
            for (int G = Ga; G < Gb; G++)
              for (int m1 = 0; m1 < nm1; m1++, gm1++)
                A1vt_gm[gm1] = vt_sG[G] * A1_gm[gm1];
            
            for (int i2 = i1; i2 < ni; i2++)
              {
                int I2 = I_i[i2];
                int M2 = M_I[I2];
                const PyArrayObject* A2_gm_obj = \
                  (const PyArrayObject*)PyList_GetItem(A_Igm_obj, I2);
                const double* A2_gm = ((const double*)A2_gm_obj->data +
                                       g_I[I2]);
                int nm2 = A2_gm_obj->dimensions[1];
                
                double one = 1.0;
                dgemm_("n", "t", &nm2, &nm1, &nG, &dv, 
                       A2_gm, &nm1, A1vt_gm, &nm2, &one, 
                       Vt_skMM + M1 * nM + M2, &nM);
              }
            g_I[I1] += nG;
          }
      int Inew = I_B[B];
      if (Inew >= 0)
        {
          // Entering new sphere:
          I_i[ni] = Inew;
          i_I[Inew] = ni;
          ni++;
        }
      else
        {
          // Leaving sphere:
          Inew = -1 - Inew;
          ni--;
          int Ilast = I_i[ni];
          int ihole = i_I[Inew];
          I_i[ihole] = Ilast;
          i_I[Ilast] = ihole;
        }
      Ga = Gb;
      B++;
    }
  Py_RETURN_NONE;
}


PyObject* construct_density(PyObject *self, PyObject *args)
{
  PyObject* A_Igm_obj;
  const PyArrayObject* rho_skMM_obj;
  const PyArrayObject* M_I_obj;
  const PyArrayObject* G_B_obj;
  const PyArrayObject* I_B_obj;
  PyArrayObject* g_I_obj;
  PyArrayObject* I_i_obj;
  PyArrayObject* i_I_obj;
  PyArrayObject* A_gm_obj;
  PyArrayObject* nt_sG_obj;

  if (!PyArg_ParseTuple(args, "OOOOOOOOOO", &A_Igm_obj, &rho_skMM_obj,
                        &M_I_obj,
                        &G_B_obj, &I_B_obj, &g_I_obj, &I_i_obj, &i_I_obj,
                        &A_gm_obj, &nt_sG_obj))
    return NULL; 

  const double* rho_skMM = (const double*)rho_skMM_obj->data;
  const int* M_I = (const int*)M_I_obj->data;
  const int* G_B = (const int*)G_B_obj->data;
  const int* I_B = (const int*)I_B_obj->data;

  int* g_I = (int*)g_I_obj->data;
  int* I_i = (int*)I_i_obj->data;
  int* i_I = (int*)i_I_obj->data;

  double* Arho_gm = (double*)A_gm_obj->data;
  double* nt_sG = (double*)nt_sG_obj->data;

  int nM = rho_skMM_obj->dimensions[0];
  int nB = G_B_obj->dimensions[0];

  int Ga = 0;
  int B = 0;
  int ni = 0;
  while (B < nB)
    {
      int Gb = G_B[B];
      int nG = Gb - Ga;
      if (nG > 0)
        // Do work for [Ga:Gb) range:
        for (int i1 = 0; i1 < ni; i1++)
          {
            int I1 = I_i[i1];
            int M1 = M_I[I1];
            const PyArrayObject* A1_gm_obj = \
              (const PyArrayObject*)PyList_GetItem(A_Igm_obj, I1);
            const double* A1_gm = (const double*)A1_gm_obj->data + g_I[I1];
            int nm1 = A1_gm_obj->dimensions[1];
            memset(Arho_gm, 0, nG * nm1 * sizeof(double));

            double factor = 1.0;
            for (int i2 = i1; i2 < ni; i2++)
              {
                int I2 = I_i[i2];
                int M2 = M_I[I2];
                const PyArrayObject* A2_gm_obj = \
                  (const PyArrayObject*)PyList_GetItem(A_Igm_obj, I2);
                const double* A2_gm = ((const double*)A2_gm_obj->data +
                                       g_I[I2]);
                int nm2 = A2_gm_obj->dimensions[1];
                
                double one = 1.0;
                dgemm_("t", "n", &nm1, &nG, &nm2, &factor, 
                       rho_skMM + M1 * nM + M2, &nM,
                       A2_gm, &nm2,
                       &one, Arho_gm, &nm1);
                factor = 2.0;
              }
            int gm1 = 0;
            for (int G = Ga; G < Gb; G++)
              {
                double nt = 0.0;
                for (int m1 = 0; m1 < nm1; m1++, gm1++)
                  nt += A1_gm[gm1] * Arho_gm[gm1];
                nt_sG[G] += nt;
              }
            g_I[I1] += nG;
          }
      int Inew = I_B[B];
      if (Inew >= 0)
        {
          // Entering new sphere:
          I_i[ni] = Inew;
          i_I[Inew] = ni;
          ni++;
        }
      else
        {
          // Leaving sphere:
          Inew = -1 - Inew;
          ni--;
          int Ilast = I_i[ni];
          int ihole = i_I[Inew];
          I_i[ihole] = Ilast;
          i_I[Ilast] = ihole;
        }
      Ga = Gb;
      B++;
    }
  Py_RETURN_NONE;
}


PyObject* construct_density1(PyObject *self, PyObject *args)
{
  PyObject* A_Igm_obj;
  const PyArrayObject* f_sM_obj;
  const PyArrayObject* M_I_obj;
  const PyArrayObject* G_B_obj;
  const PyArrayObject* I_B_obj;
  PyArrayObject* g_I_obj;
  PyArrayObject* I_i_obj;
  PyArrayObject* i_I_obj;
  PyArrayObject* nt_sG_obj;

  if (!PyArg_ParseTuple(args, "OOOOOOOOO", &A_Igm_obj, &f_sM_obj,
                        &M_I_obj,
                        &G_B_obj, &I_B_obj, &g_I_obj, &I_i_obj, &i_I_obj,
                        &nt_sG_obj))
    return NULL; 

  const double* f_sM = (const double*)f_sM_obj->data;
  const int* M_I = (const int*)M_I_obj->data;
  const int* G_B = (const int*)G_B_obj->data;
  const int* I_B = (const int*)I_B_obj->data;

  int* g_I = (int*)g_I_obj->data;
  int* I_i = (int*)I_i_obj->data;
  int* i_I = (int*)i_I_obj->data;

  double* nt_sG = (double*)nt_sG_obj->data;

  int nB = G_B_obj->dimensions[0];

  int Ga = 0;
  int B = 0;
  int ni = 0;
  while (B < nB)
    {
      int Gb = G_B[B];
      int nG = Gb - Ga;
      if (nG > 0)
        // Do work for [Ga:Gb) range:
        for (int i = 0; i < ni; i++)
          {
            int II = I_i[i];
            int M = M_I[II];
            const PyArrayObject* A_gm_obj = \
              (const PyArrayObject*)PyList_GetItem(A_Igm_obj, II);
            const double* A_gm = (const double*)A_gm_obj->data + g_I[II];
            int nm = A_gm_obj->dimensions[1];
            //int gm = 0;
            for (int gm = 0, G = Ga; G < Gb; G++)
              for (int m = 0; m < nm; m++, gm++)
                nt_sG[G] += A_gm[gm] * A_gm[gm] * f_sM[M];
            g_I[II] += nG;
          }
      int Inew = I_B[B];
      if (Inew >= 0)
        {
          // Entering new sphere:
          I_i[ni] = Inew;
          i_I[Inew] = ni;
          ni++;
        }
      else
        {
          // Leaving sphere:
          Inew = -1 - Inew;
          ni--;
          int Ilast = I_i[ni];
          int ihole = i_I[Inew];
          I_i[ihole] = Ilast;
          i_I[Ilast] = ihole;
        }
      Ga = Gb;
      B++;
    }
  Py_RETURN_NONE;
}
