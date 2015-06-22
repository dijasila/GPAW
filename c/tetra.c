#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "extensions.h"

PyObject* tetrahedron_weight(PyObject *self, PyObject *args)
{
  PyArrayObject* epsilon_k;
  PyArrayObject* omega_w;
  PyArrayObject* gi_w;
  PyArrayObject* Ii_kw;
  double f10, f20, f21, f30, f31, f32;
  double f01, f02, f12, f03, f13, f23;
  double omega;
    
  if (!PyArg_ParseTuple(args, "OOOO",
                        &epsilon_k, &omega_w, &gi_w, &Ii_kw))
    return NULL;
        
  int nw = PyArray_DIM(omega_w, 0);    
  double* e_k = (double*)PyArray_DATA(epsilon_k);
  double* o_w = (double*)PyArray_DATA(omega_w);
  double* g_w = (double*)PyArray_DATA(gi_w);
  double* I_kw = (double*)PyArray_DATA(Ii_kw);
  double delta = e_k[3] - e_k[0];
  for (int w = 0; w < nw; w++) {
    omega = o_w[w];
    f10 = (omega - e_k[0]) / (e_k[1] - e_k[0]);
    f20 = (omega - e_k[0]) / (e_k[2] - e_k[0]);
    f21 = (omega - e_k[1]) / (e_k[2] - e_k[1]);
    f30 = (omega - e_k[0]) / (e_k[3] - e_k[0]);
    f31 = (omega - e_k[1]) / (e_k[3] - e_k[1]);
    f32 = (omega - e_k[2]) / (e_k[3] - e_k[2]);

    f01 = 1 - f10;
    f02 = 1 - f20;
    f03 = 1 - f30;
    f12 = 1 - f21;
    f13 = 1 - f31;
    f23 = 1 - f32;

    if (e_k[1] != e_k[0] && e_k[0] <= omega && omega <= e_k[1])
      {
        g_w[w] = 3 * f20 * f30 / (e_k[1] - e_k[0]);
        I_kw[0 * nw + w] = (f01 + f02 + f03) / 3;
        I_kw[1 * nw + w] = f10 / 3;
        I_kw[2 * nw + w] = f20 / 3;
        I_kw[3 * nw + w] = f30 / 3;
      }
    else if (e_k[1] < omega && omega < e_k[2])
      {
        g_w[w] = 3 / delta * (f12 * f20 + f21 * f13);
        I_kw[0 * nw + w] = f03 / 3 + f02 * f20 * f12 / (g_w[w] * delta);
        I_kw[1 * nw + w] = f12 / 3 + f13 * f13 * f21 / (g_w[w] * delta);
        I_kw[2 * nw + w] = f21 / 3 + f20 * f20 * f12 / (g_w[w] * delta);
        I_kw[3 * nw + w] = f30 / 3 + f31 * f13 * f21 / (g_w[w] * delta);
      }
    else if (e_k[2] != e_k[3] && e_k[2] <= omega && omega <= e_k[3])
      {
        g_w[w] = 3 * f03 * f13 / (e_k[3] - e_k[2]);
        I_kw[0 * nw + w] = f03 / 3;
        I_kw[1 * nw + w] = f13 / 3;
        I_kw[2 * nw + w] = f23 / 3;
        I_kw[3 * nw + w] = (f30 + f31 + f32) / 3;
      }
  }
  Py_RETURN_NONE;
}
