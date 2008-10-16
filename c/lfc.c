#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "spline.h"
#include "bmgs/bmgs.h"

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
  printf("%d %d %f\n", ngmax,l, rcut);
  double* A_gm = GPAW_MALLOC(double, ngmax * (2 * l + 1));
  
  int nBmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]));
  int* G_B = GPAW_MALLOC(int, 2 * nBmax);

  int nB = 0;
  int ngm = 0;
  int G = n_c[2] * (beg_c[1] - gdcorner_c[1] + n_c[1] 
                    * (beg_c[0] - gdcorner_c[0]));
  printf("%d\n", G);
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
                  for (int m = -l; m <= l; m++)
                    {
                      double Y = 0.28209479177387814; //sphericalharmonic(l, m, x/r, y/r, z/r);
                      A_gm[ngm++] = Y * Ar;
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

  npy_intp B_dims[1] = {nB};
  PyArrayObject* G_B_obj = (PyArrayObject*)PyArray_SimpleNew(1, B_dims,
                                                             NPY_INT);
  memcpy(G_B_obj->data, G_B, nB * sizeof(int));

  return Py_BuildValue("(OO)", A_gm_obj, G_B_obj);
}

