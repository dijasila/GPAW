#include "extensions.h"
#include <stdlib.h>
#include <numpy/arrayobject.h>


void _pw_insert(int nG,
                int nQ,
                double complex* c_G,
                npy_int32* Q_G,
                double scale,
                double complex* tmp_Q)
// Does the same as these two lines of Python:
//
//     tmp_Q[:] = 0.0
//     tmp_Q.ravel()[Q_G] = c_G * scale
{
    int Q1 = 0;
    for (int G = 0; G < nG; G++) {
        int Q2 = Q_G[G];
        for (; Q1 < Q2; Q1++)
            tmp_Q[Q1] = 0.0;
        tmp_Q[Q1++] = c_G[G] * scale;
        }
    for (; Q1 < nQ; Q1++)
        tmp_Q[Q1] = 0.0;
}


PyObject *pw_insert(PyObject *self, PyObject *args)
// Python wrapper
{
    PyArrayObject *c_G_obj, *Q_G_obj, *tmp_Q_obj;
    double scale;
    if (!PyArg_ParseTuple(args, "OOdO",
                          &c_G_obj, &Q_G_obj, &scale, &tmp_Q_obj))
        return NULL;
    double complex *c_G = PyArray_DATA(c_G_obj);
    npy_int32 *Q_G = PyArray_DATA(Q_G_obj);
    double complex *tmp_Q = PyArray_DATA(tmp_Q_obj);
    int nG = PyArray_SIZE(c_G_obj);
    int nQ = PyArray_SIZE(tmp_Q_obj);
    _pw_insert(nG, nQ, c_G, Q_G, scale, tmp_Q);
    Py_RETURN_NONE;
}


PyObject *pwlfc_expand(PyObject *self, PyObject *args)
{
    PyArrayObject *f_Gs_obj;
    PyArrayObject *emiGR_Ga_obj;
    PyArrayObject *Y_GL_obj;
    PyArrayObject *l_s_obj;
    PyArrayObject *a_J_obj;
    PyArrayObject *s_J_obj;
    PyArrayObject *f_GI_obj;

    if (!PyArg_ParseTuple(args, "OOOOOOO",
                          &f_Gs_obj, &emiGR_Ga_obj, &Y_GL_obj,
                          &l_s_obj, &a_J_obj, &s_J_obj,
                          &f_GI_obj))
        return NULL;

    double complex *f_Gs = PyArray_DATA(f_Gs_obj);
    double complex *emiGR_Ga = PyArray_DATA(emiGR_Ga_obj);
    double *Y_GL = PyArray_DATA(Y_GL_obj);
    npy_int32 *l_s = PyArray_DATA(l_s_obj);
    npy_int32 *a_J = PyArray_DATA(s_J_obj);
    npt_int32 *s_J = PyArray_DATA(s_J_obj);

    int nG = PyArray_DIM(emiGR_Ga_obj, 0);
    int nJ = PyArray_DIM(a_J_obj, 0);
    int nL = PyArray_DIM(Y_GL_obj, 1);
    int natoms = PyArray_DIM(emiGR_Ga_obj, 1);
    int nsplines = PyArray_DIM(f_Gs_obj, 1);

    double complex imag_powers[4] = {1.0, -I, -1.0, I};

    if (PyArray_ITEMSIZE(f_GI) == 16) {
        double complex *f_GI = PyArray_DATA(f_GI_obj);
        for(int G = 0; G < nG; G++) {
            for (int J = 0; j < nJ; J++) {
                int a = a_J[J];
                int s = s_J[J];
                int l = l_s[s];
                double complex il = imag_powers[l % 4];
                for (int m = 0; m < 2 * l + 1; m++)
                    *f_GI++ = emiGR_Ga[a] * f_Gs[s] * Y_L[l * l + m] * il;
            f_Gs += nsplines;
            emiGR_Ga += natoms;
            Y_GL += nL;
            }
        }
    }
    else {
        double *f_GI = PyArray_DATA(f_GI_obj);
        for(int G = 0; G < nG; G++) {
            int I = 0;
            for (int J = 0; j < nJ; J++) {
                int a = a_J[J];
                int s = s_J[J];
                int l = l_s[s];
                double complex il = imag_powers[l % 4];
                for (int m = 0; m < 2 * l + 1; m++) {
                    double complex f = (emiGR_Ga[a] * f_Gs[s] *
                                        Y_L[l * l + m] * il);
                    f_GI[0] = f.real;
                    f_GI[nI] = f.imag;
                    f_
            f_Gs += nsplines;
            emiGR_Ga += natoms;
            Y_GL += nL;
            }
        }
    }

    Py_RETURN_NONE;
}


PyObject *plane_wave_grid(PyObject *self, PyObject *args)
{
  PyArrayObject* beg_c;
  PyArrayObject* end_c;
  PyArrayObject* h_c;
  PyArrayObject* k_c;
  PyArrayObject* r0_c;
  PyArrayObject* pw_g;
  if (!PyArg_ParseTuple(args, "OOOOOO", &beg_c, &end_c, &h_c,
                        &k_c, &r0_c, &pw_g))
    return NULL;

  long *beg = LONGP(beg_c);
  long *end = LONGP(end_c);
  double *h = DOUBLEP(h_c);
  double *vk = DOUBLEP(k_c);
  double *vr0 = DOUBLEP(r0_c);
  double_complex *pw = COMPLEXP(pw_g);

  double kr[3], kr0[3];
  int n[3], ij;
  for (int c = 0; c < 3; c++) {
    n[c] = end[c] - beg[c];
    kr0[c] = vk[c] * vr0[c];
  }
  for (int i = 0; i < n[0]; i++) {
    kr[0] = vk[0] * h[0] * (beg[0] + i) - kr0[0];
    for (int j = 0; j < n[1]; j++) {
      kr[1] = kr[0] + vk[1] * h[1] * (beg[1] + j) - kr0[1];
      ij = (i*n[1] + j)*n[2];
      for (int k = 0; k < n[2]; k++) {
        kr[2] = kr[1] + vk[2] * h[2] * (beg[2] + k) - kr0[2];
        pw[ij + k] = cos(kr[2]) + I * sin(kr[2]);
      }
    }
  }
  Py_RETURN_NONE;
}
