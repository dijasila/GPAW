#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL GPAW_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>
#include "spline.h"
#include "lfc.h"
#include "bmgs/spherical_harmonics.h"
#include "bmgs/bmgs.h"


static void lfc_dealloc(LFCObject *self)
{
  if (self->bloch_boundary_conditions)
    free(self->phase_i);
  free(self->volume_i);
  free(self->work_gm);
  free(self->ngm_W);
  free(self->i_W);
  free(self->volume_W);
  PyObject_DEL(self);
}

PyObject* calculate_potential_matrix(PyObject *self, PyObject *args);
PyObject* construct_density(PyObject *self, PyObject *args);
PyObject* construct_density1(LFCObject *self, PyObject *args);

static PyMethodDef lfc_methods[] = {
    {"calculate_potential_matrix",
     (PyCFunction)calculate_potential_matrix, METH_VARARGS, 0},
    {"construct_density",
     (PyCFunction)construct_density, METH_VARARGS, 0},
    {"construct_density1",
     (PyCFunction)construct_density1, METH_VARARGS, 0},
#ifdef PARALLEL
    {"broadcast",
     (PyCFunction)localized_functions_broadcast, METH_VARARGS, 0},
#endif
    {NULL, NULL, 0, NULL}
};

static PyObject* lfc_getattr(PyObject *obj, char *name)
{
  return Py_FindMethod(lfc_methods, obj, name);
}

static PyTypeObject LFCType = {
  PyObject_HEAD_INIT(&PyType_Type)
  0,
  "LocalizedFunctionsCollection",
  sizeof(LFCObject),
  0,
  (destructor)lfc_dealloc,
  0,
  lfc_getattr
};

PyObject * NewLFCObject(PyObject *obj, PyObject *args)
{
  PyObject* A_Wgm_obj;
  const PyArrayObject* M_W_obj;
  const PyArrayObject* G_B_obj;
  const PyArrayObject* W_B_obj;
  double dv;

  if (!PyArg_ParseTuple(args, "OOOOd",
                        &A_Wgm_obj, &M_W_obj, &G_B_obj, &W_B_obj, &dv))
    return NULL; 

  LFCObject *self = PyObject_NEW(LFCObject, &LFCType);
  if (self == NULL)
    return NULL;

  self->dv = dv;
  self->bloch_boundary_conditions = false;

  const int* M_W = (const int*)M_W_obj->data;
  self->G_B = (int*)G_B_obj->data;
  self->W_B = (int*)W_B_obj->data;

  int nB = G_B_obj->dimensions[0];
  int nW = PyList_Size(A_Wgm_obj);

  self->nW = nW;
  self->nB = nB;

  int nimax = 0;
  int ngmax = 0;
  int ni = 0; 
  int Ga = 0;                 
  for (int B = 0; B < nB; B++)
    {
      int Gb = self->G_B[B];                                        
      int nG = Gb - Ga;                                         
      if (ni > 0 && nG > ngmax)
        ngmax = nG;
      if (self->W_B[B] >= 0)
        ni += 1;
      else
        {
          if (ni > nimax)
            nimax = ni;
          ni--;
        }
      Ga = Gb;
    }
  assert(ni == 0);
  
  self->volume_W = GPAW_MALLOC(LFVolume, nW);
  self->i_W = GPAW_MALLOC(int, nW);
  self->ngm_W = GPAW_MALLOC(int, nW);

  int nmmax = 0;
  for (int W = 0; W < nW; W++)
    {
      const PyArrayObject* A_gm_obj = \
        (const PyArrayObject*)PyList_GetItem(A_Wgm_obj, W);
      LFVolume* volume = &self->volume_W[W];
      volume->A_gm = (const double*)A_gm_obj->data;
      self->ngm_W[W] = A_gm_obj->dimensions[0] * A_gm_obj->dimensions[1];
      volume->nm = A_gm_obj->dimensions[1];
      volume->M = M_W[W];
      volume->W = W;
      if (volume->nm > nmmax)
        nmmax = volume->nm;
    }

  self->work_gm = GPAW_MALLOC(double, ngmax * nmmax);
  self->volume_i = GPAW_MALLOC(LFVolume, nimax);
  if (self->bloch_boundary_conditions)
    self->phase_i = GPAW_MALLOC(complex double, nimax);

  return (PyObject*)self;
}

PyObject* calculate_potential_matrix(PyObject *self, PyObject *args)
{
  const PyArrayObject* vt_G_obj;
  PyArrayObject* Vt_MM_obj;

  if (!PyArg_ParseTuple(args, "OO", &vt_G_obj, &Vt_MM_obj))
    return NULL; 

  const double* vt_G = (const double*)vt_G_obj->data;
  double* Vt_MM = (double*)Vt_MM_obj->data;

  int nM = Vt_MM_obj->dimensions[0];

  LFCObject* lfc = (LFCObject*)self;

  GRID_LOOP_START(lfc, -1)
    {
      for (int i1 = 0; i1 < ni; i1++)
        {
          LFVolume* v1 = volume_i + i1;
          int M1 = v1->M;
          int nm1 = v1->nm;
          int gm1 = 0;
          for (int G = Ga; G < Gb; G++)
            for (int m1 = 0; m1 < nm1; m1++, gm1++)
              lfc->work_gm[gm1] = vt_G[G] * v1->A_gm[gm1];
          
          for (int i2 = 0; i2 < ni; i2++)
            {
              LFVolume* v2 = volume_i + i2;
              int M2 = v2->M;
              int nm2 = v2->nm;
              if (1)//(M1 > M2)
		{
		  double* Vt_mm = Vt_MM + M1 * nM + M2;
		  for (int g = 0; g < nG; g++)
		    for (int m1 = 0; m1 < nm1; m1++)
		      for (int m2 = 0; m2 < nm2; m2++)
			Vt_mm[m2 + m1 * nM] += (v2->A_gm[g * nm2 + m2] * 
						lfc->work_gm[g * nm1 + m1] *
						lfc->dv);
		}
	      else
		{
		  double* Vt_mm = Vt_MM + M2 * nM + M1;
		  for (int g = 0; g < nG; g++)
		    for (int m2 = 0; m2 < nm2; m2++)
		      for (int m1 = 0; m1 < nm1; m1++)
			Vt_mm[m1 + m2 * nM] += (v2->A_gm[g * nm2 + m2] * 
						lfc->work_gm[g * nm1 + m1] *
						lfc->dv);
		}
            }
        }
    }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}


PyObject* construct_density(PyObject *self, PyObject *args)
{
  const PyArrayObject* rho_MM_obj;
  PyArrayObject* nt_G_obj;
  
  if (!PyArg_ParseTuple(args, "OO", &rho_MM_obj, &nt_G_obj))
    return NULL; 
  
  const double* rho_MM = (const double*)rho_MM_obj->data;
  double* nt_G = (double*)nt_G_obj->data;
  
  int nM = rho_MM_obj->dimensions[0];
  
  LFCObject* lfc = (LFCObject*)self;
  
  GRID_LOOP_START(lfc, -1)
    {
      for (int i1 = 0; i1 < ni; i1++)
        {
          LFVolume* v1 = volume_i + i1;
          int M1 = v1->M;
          int nm1 = v1->nm;
	  memset(lfc->work_gm, 0, nG * nm1 * sizeof(double));
	  double factor = 1.0;
          for (int i2 = i1; i2 < ni; i2++)
            {
              LFVolume* v2 = volume_i + i2;
              int M2 = v2->M;
              int nm2 = v2->nm;
	      const double* rho_mm = rho_MM + M1 * nM + M2;
	      for (int g = 0; g < nG; g++)
		for (int m2 = 0; m2 < nm2; m2++)
		  for (int m1 = 0; m1 < nm1; m1++)
		    lfc->work_gm[m1 + g * nm1] += (v2->A_gm[g * nm2 + m2] * 
						   rho_mm[m2 + m1 * nM] *
						   factor);
	      /*
	      dgemm_("t", "n", &nm1, &nG, &nm2, &factor, 
		     rho_MM + M1 * nM + M2, &nM,
		     v2->A_gm, &nm2,
		     &one, lfc->work_gm, &nm1);*/
	      factor = 2.0;
	    }
	  int gm1 = 0;
	  for (int G = Ga; G < Gb; G++)
	    {
	      double nt = 0.0;
	      for (int m1 = 0; m1 < nm1; m1++, gm1++)
		nt += v1->A_gm[gm1] * lfc->work_gm[gm1];
	      nt_G[G] += nt;
	    }
	}
    }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}

PyObject* construct_density1(LFCObject *lfc, PyObject *args)
{
  const PyArrayObject* f_M_obj;
  PyArrayObject* nt_G_obj;
  
  if (!PyArg_ParseTuple(args, "OO", &f_M_obj, &nt_G_obj))
    return NULL; 
  
  const double* f_M = (const double*)f_M_obj->data;
  double* nt_G = (double*)nt_G_obj->data;
  
  GRID_LOOP_START(lfc, -1)
    {
      for (int i = 0; i < ni; i++)
	{
	  LFVolume* v = volume_i + i;
	  for (int gm = 0, G = Ga; G < Gb; G++)
	    for (int m = 0; m < v->nm; m++, gm++)
	      nt_G[G] += v->A_gm[gm] * v->A_gm[gm] * f_M[v->M];
	}
    }
  GRID_LOOP_STOP(lfc, -1);
  Py_RETURN_NONE;
}

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
  int nm = 2 * l + 1;
  double rcut = spline->dr * spline->nbins;

  int ngmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]) *
               (end_c[2] - beg_c[2]));
  double* A_gm = GPAW_MALLOC(double, ngmax * nm);
  
  int nBmax = ((end_c[0] - beg_c[0]) *
               (end_c[1] - beg_c[1]));
  int* G_B = GPAW_MALLOC(int, 2 * nBmax);

  int nB = 0;
  int ngm = 0;
  int G = -gdcorner_c[2] + n_c[2] * (beg_c[1] - gdcorner_c[1] + n_c[1] 
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
                  double A = bmgs_splinevalue(spline, r);
		  double* p = A_gm + ngm;
		  
		  spherical_harmonics(l, A, x, y, z, r2, p);
		  
		  ngm += nm;
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
