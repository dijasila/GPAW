#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"
#include "spline.h"
#include "bmgs/bmgs.h"

PyObject* spline_to_grid(PyObject *self, PyObject *args)
{
  SplineObject* spline_obj;
  PyArrayObject* start_c_obj;
  PyArrayObject* start_c_obj;
  PyArrayObject* start_c_obj;
  PyArrayObject* start_c_obj;
  if (!PyArg_ParseTuple(args, "OOOOO", &spline_obj,
                        &start_c_obj, &end_c_obj, &pos_v_obj, &h_cv_obj))
    return NULL; 

  const bmgsspline* spline = spline_obj->spline;
  long* start_c = LONGP(start_c_obj);
  long* end_c = LONGP(end_c_obj);
  double* pos_v = DOUBLEP(pos_v_obj);
  double* h_cv = DOUBLEP(h_cv_obj);

  int l = spline->l;
  double rcut = spline->rcut;

  int ngmax = ((end_c[0] - begin_c[0]) *
               (end_c[1] - begin_c[1]) *
               (end_c[2] - begin_c[2]));
  double* A_gm = GPAW_MALLOC(double, ngmax * (2 * l + 1));
  
  int nBmax = ((end_c[0] - begin_c[0]) *
               (end_c[1] - begin_c[1]));
  int* G_B = GPAW_MALLOC(double, 2 * nBmax);

  int ng = 0;
  int nB = 0;
  int ngm = 0;
  for (int g0 = start_c[0]; g0 < end_c[0]; g0++)
    {
      for (int g1 = start_c[1]; g1 < end_c[1]; g1++)
        {
	  int gz_start = -1; // function boundary coordinates
	  int gz_end = -1;
          for (int g2 = start_c[2]; g2 < end_c[2]; g2++)
            {
              double x = h_cv[0] * g0 + h_cv[1] * g1 + h_cv[2] * g2 - pos_v[0];
              double y = h_cv[3] * g0 + h_cv[4] * g1 + h_cv[5] * g2 - pos_v[1];
              double z = h_cv[6] * g0 + h_cv[7] * g1 + h_cv[8] * g2 - pos_v[2];
              double r2 = x * x + y * y + z * z;
              double r = sqrt(r2);
              if (r < rcut)
                {
		  if (gz_start < 0)
		    gz_start = g2; // found boundary
		  gz_end = g2
                  double Ar = splinevalue(spline, r);
                  for (int m = -l; m <= l; m++)
                    {
                      double Y = sphericalharmonic(l, m, x/r, y/r, z/r);
                      A_gm[ngm++] = Y * Ar;
                    }
                  /*if (intersection_found)
                    {
                      g3;
		      intersection_found = false;
                    }
		    g = g3;*/
                }
              ng++;
            }
	  if(gz_end >= 0)
	    {
	      gz_end++;
	      int G1 = 0;
	      int G2 = 0;
	      G_B[nB++] = G1;
	      G_B[nB++] = G2;
	    }
        }
    }
  npA_gm = null;
  npG_B = null;
  
  PyObject* AG_tuple = Py_BuildValue("(OO)", npA_gm, npG_B);
  Py_Return AG_tuple;

	/*
        rcut = spline.get_cutoff()
        G_B = []
        A_gm = []
        for gx in range(start_c[0], end_c[0]):
            for gy in range(start_c[1], end_c[1]):
                gz1 = None
                gz2 = None
                for gz in range(start_c[2], end_c[2]):
                    d_v = np.dot((gx, gy, gz), h_cv) - pos_v
                    r = (d_v**2).sum()**0.5
                    if r < rcut:
                        if gz1 is None:
                            gz1 = gz
                        gz2 = gz
                        fr = spline(r)
                        A_gm.append([fr * Y(self.l**2 + m, *(d_v / r))
                                     for m in range(2 * self.l + 1)])
                if gz2 is not None:
                    gz2 += 1
                    G1 = self.gd.flat_index((gx, gy, gz1))
                    G2 = self.gd.flat_index((gx, gy, gz2))
                    G_B.extend((G1, G2))
                    
        return np.array(A_gm), np.array(G_B)
	*/
  
}
