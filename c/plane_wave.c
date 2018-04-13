#include "extensions.h"
#include <stdlib.h>

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

PyObject *pwlfc_expand(PyObject *self, PyObject *args)
{
    PyArrayObject *G_Qv_obj, *pos_av_obj; //*aji1i2I1I2_obj;
    PyObject *lf_aj_obj;
    PyArrayObject *Y_LG_obj;
    int q, G1, G2;

    PyArrayObject *emiGR_G_obj, *f_IG_obj;

    if (!PyArg_ParseTuple(args, "OOOOiiiOO", &G_Qv_obj, &pos_av_obj,
                          &lf_aj_obj, &Y_LG_obj,
                          &q, &G1, &G2, &f_IG_obj, &emiGR_G_obj))
        return NULL;

    if(q == -1) {
        q = 0;
    }

    double *G_Qv = PyArray_DATA(G_Qv_obj);
    double *pos_av = PyArray_DATA(pos_av_obj);
    double complex *emiGR_G = PyArray_DATA(emiGR_G_obj);
    double complex *f_IG = PyArray_DATA(f_IG_obj);
    double *Y_LG = PyArray_DATA(Y_LG_obj);

    int natoms = PyArray_SIZE(pos_av_obj) / 3;
    int nG = PyArray_SIZE(G_Qv_obj) / 3;
    if(nG != G2 - G1) {
        return NULL;
    }
    int G;

    int v;
    int L;
    //double complex minus_i = 1.0;//-1.0 * I;
    double complex imag_powers[4] = {1.0, -I, -1.0, I};

    int I1 = 0;
    int a;

    npy_intp* Ystrides = PyArray_STRIDES(Y_LG_obj);
    int nGtotal = Ystrides[0] / sizeof(double);

    for(a=0; a < natoms; a++) {
        for(G=0; G < nG; G++) {
            double GR = 0;
            for(v=0; v < 3; v++) {
                GR += G_Qv[3 * G + v] * pos_av[3 * a + v];
            }
            emiGR_G[G] = cexp(-I * GR);
        }

        PyObject *lf_j_obj = PyList_GET_ITEM(lf_aj_obj, a);
        int nj = PyList_GET_SIZE(lf_j_obj);
        int j;
        for(j=0; j < nj; j++) {
            PyObject *l_and_f_qG = PyList_GET_ITEM(lf_j_obj, j);
            PyObject *l_obj = PyTuple_GET_ITEM(l_and_f_qG, 0);
            PyObject *f_qG_obj = PyTuple_GET_ITEM(l_and_f_qG, 1);
            int l = PyLong_AsLong(l_obj);
            int nL = 2 * l + 1;
            int L2 = l * l;

            double complex ilpow = imag_powers[l % 4];
            PyObject *f_G_obj = PyList_GET_ITEM(f_qG_obj, q);
            double *f_G = PyArray_DATA((PyArrayObject *)f_G_obj);
            for(G=0; G < nG; G++) {
                double complex emiGR_f = emiGR_G[G] * f_G[G1 + G] * ilpow;
                for(L=0; L < nL; L++) {
                    // Terrible memory locality!
                    f_IG[(I1 + L) * nG + G] = emiGR_f * Y_LG[(L2 + L) * nGtotal + G1 + G];
                }
            }
            I1 += nL;
        }
    }
    Py_RETURN_NONE;
}


