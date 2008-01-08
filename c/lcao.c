#include "localized_functions.h"
#include "extensions.h"
#include "bmgs/bmgs.h"


int dgemv_(char *trans, int *m, int * n,
	   double *alpha, double *a, int *lda, 
	   double *x, int *incx, double *beta, 
	   double *y, int *incy);
int dgemm_(char *transa, char *transb, int *m, int * n,
	   int *k, const double *alpha, double *a, int *lda, 
	   double *b, int *ldb, double *beta, 
	   double *c, int *ldc);


//                    +-----------n
//  +----m   +----m   | +----c+m  |
//  |    |   |    |   | |    |    |
//  |  b | = |  v | * | |  a |    |  
//  |    |   |    |   | |    |    |  
//  0----+   0----+   | c----+    |
//                    |           |
//                    0-----------+
void cut(const double* a, const int n[3], const int c[3],
	 const double* v,
	 double* b, const int m[3])
{
  a += c[2] + (c[1] + c[0] * n[1]) * n[2];
  for (int i0 = 0; i0 < m[0]; i0++)
    {
      for (int i1 = 0; i1 < m[1]; i1++)
        {
	  for (int i2 = 0; i2 < m[2]; i2++)
	    b[i2] = v[i2] * a[i2];
          a += n[2];
          b += m[2];
          v += m[2];
        }
      a += n[2] * (n[1] - m[1]);
    }
}


PyObject * overlap(PyObject* self, PyObject *args)
{
  PyObject* boxes;
  const PyArrayObject* vt_obj;
  PyArrayObject* Vt_obj;
  if (!PyArg_ParseTuple(args, "OOO", &boxes, &vt_obj, &Vt_obj))
    return NULL;

  const double *vt = DOUBLEP(vt_obj);
  double *Vt = DOUBLEP(Vt_obj);

  int nao = Vt_obj->dimensions[0];
  int nb = PyList_Size(boxes);
  int nmem = 0;
  double* a1 = 0;
  int m1 = 0;
  for (int b1 = 0; b1 < nb; b1++)
    {
      const LocalizedFunctionsObject* lf1 =
	(const LocalizedFunctionsObject*)PyList_GetItem(boxes, b1);
      int nao1 = lf1->nf;
      double* f1 = lf1->f;
      double* vt1 = lf1->w;
      bmgs_cut(vt, lf1->size, lf1->start, vt1, lf1->size0);
      int m2 = m1;
      for (int b2 = b1; b2 < nb; b2++)
	{
	  const LocalizedFunctionsObject* lf2 =
	    (const LocalizedFunctionsObject*)PyList_GetItem(boxes, b2);
	  int beg[3];
	  int end[3];
	  int size[3];
	  int beg1[3];
	  int beg2[3];
	  bool overlap = true;
	  for (int c = 0; c < 3; c++)
	    {
	      beg[c] = MAX(lf1->start[c], lf2->start[c]);
	      end[c] = MIN(lf1->start[c] + lf1->size0[c],
			     lf2->start[c] + lf2->size0[c]);
	      size[c] = end[c] - beg[c];
	      if (size[c] <= 0)
		{
		  overlap = false;
		  continue;
		}
	      beg1[c] = beg[c] - lf1->start[c];
	      beg2[c] = beg[c] - lf2->start[c];
	    }
	  int nao2 = lf2->nf;
	  if (overlap)
	    {
	      int ng = size[0] * size[1] * size[2];
	      int n = ng * (nao1 + nao2) + nao1 * nao2;
	      if (n > nmem)
		{
		  if (nmem != 0)
		    free(a1);
		  nmem = n;
		  a1 = GPAW_MALLOC(double, nmem);
		}
	      double* a2 = a1 + ng * nao1;
	      double* H = a2 + ng * nao2;
	      double* f2 = lf2->f;
	      double* vt2 = lf2->w;
	      if (b2 > b1)
		{
		  bmgs_cut(vt1, lf1->size0, beg1, vt2, size);
		  for (int i = 0; i < nao1; i++)
		    cut(f1 + i * lf1->ng0, lf1->size0, beg1, vt2,
			a1 + i * ng, size);
		  for (int i = 0; i < nao2; i++)
		    bmgs_cut(f2 + i * lf2->ng0, lf2->size0, beg2,
			     a2 + i * ng, size);
		}
	      else
		{
		  for (int i1 = 0; i1 < nao1; i1++)
		    for (int g = 0; g < ng; g++)
		      a1[i1 * ng + g] = vt1[g] * f1[i1 * ng + g];
		  a2 = f2;
		}
	      double zero = 0.0;
	      dgemm_("t", "n", &nao2, &nao1, &ng, &(lf1->dv),
		     a2, &ng, a1, &ng, &zero, H, &nao2);
	      for (int i1 = 0; i1 < nao1; i1++)
		for (int i2 = 0; i2 < nao2; i2++)
		  Vt[m1 + i1 + (m2 + i2) * nao] = *H++;
	    }
	  m2 += nao2;
	}
      m1 += nao1;
    }
  if (nmem != 0)
    free(a1);
  Py_RETURN_NONE;
}

/*
PyObject * overlap2(PyObject* self, PyObject *args)
{
  PyListObject* lfs_b_obj;
  const PyArrayObject* m_b_obj;
  const PyArrayObject* phase_bk_obj;
  const PyArrayObject* vt_G_obj;
  PyArrayObject* Vt_kmm_obj;
  if (!PyArg_ParseTuple(args, "OOOOO", &lfs_b_obj, &m_b_obj, &phase_bk_obj,
			&vt_G_obj, &Vt_kmm_obj))
    return NULL;

  int nk = phase_bk_obj->dimensions[1];

  const long *m_b = LONGP(m_b_obj);
  const double_complex *phase_bk = COMPLEXP(phase_bk_obj);
  const double *vt_G = DOUBLEP(vt_G_obj);
  double *Vt_mm = 0;
  double_complex *Vt_kmm = 0;

  if (nk == 0)
    Vt_mm = DOUBLEP(Vt_kmm_obj);
  else
    Vt_kmm = COMPLEXP(Vt_kmm_obj);

  int nb = PyList_Size(boxes);

  int nmem = 0;
  double* a1 = 0;
  for (int b1 = 0; b1 < nb; b1++)
    {
      const LocalizedFunctionsObject* lf1 =
	(const LocalizedFunctionsObject*)PyList_GetItem(boxes, b1);
      int m1 = m_b[b1];
      int nao1 = lf1->nf;
      double* f1 = lf1->f;
      double* vt1 = lf1->w;
      bmgs_cut(vt, lf1->size, lf1->start, vt1, lf1->size0);
      for (int b2 = b1; b2 < nb; b2++)
	{
	  const LocalizedFunctionsObject* lf2 =
	    (const LocalizedFunctionsObject*)PyList_GetItem(boxes, b2);
	  int beg[3];
	  int end[3];
	  int size[3];
	  int beg1[3];
	  int beg2[3];
	  bool overlap = true;
	  for (int c = 0; c < 3; c++)
	    {
	      beg[c] = MAX(lf1->start[c], lf2->start[c]);
	      end[c] = MIN(lf1->start[c] + lf1->size0[c],
			     lf2->start[c] + lf2->size0[c]);
	      size[c] = end[c] - beg[c];
	      if (size[c] <= 0)
		{
		  overlap = false;
		  continue;
		}
	      beg1[c] = beg[c] - lf1->start[c];
	      beg2[c] = beg[c] - lf2->start[c];
	    }
	  int nao2 = lf2->nf;
	  if (overlap)
	    {
	      int ng = size[0] * size[1] * size[2];
	      int n = ng * (nao1 + nao2) + nao1 * nao2;
	      if (n > nmem)
		{
		  if (nmem != 0)
		    free(a1);
		  nmem = n;
		  a1 = GPAW_MALLOC(double, nmem);
		}
	      double* a2 = a1 + ng * nao1;
	      double* H = a2 + ng * nao2;
	      double* f2 = lf2->f;
	      double* vt2 = lf2->w;
	      double dv = lf1->dv; 
	      int m2 = m_b[b2];
	      if (b2 > b1)
		{
		  bmgs_cut(vt1, lf1->size0, beg1, vt2, size);
		  for (int i = 0; i < nao1; i++)
		    cut(f1 + i * lf1->ng0, lf1->size0, beg1, vt2,
			a1 + i * ng, size);
		  for (int i = 0; i < nao2; i++)
		    bmgs_cut(f2 + i * lf2->ng0, lf2->size0, beg2,
			     a2 + i * ng, size);
		  if (m1 == m2)
		    dv *= 2;
		}
	      else
		{
		  for (int i1 = 0; i1 < nao1; i1++)
		    for (int g = 0; g < ng; g++)
		      a1[i1 * ng + g] = vt1[g] * f1[i1 * ng + g];
		  a2 = f2;
		}
	      double zero = 0.0;
	      dgemm_("t", "n", &nao2, &nao1, &ng, &dv,
		     a2, &ng, a1, &ng, &zero, H, &nao2);
	      for (int i1 = 0; i1 < nao1; i1++)
		for (int i2 = 0; i2 < nao2; i2++)
		  Vt[m1 + i1 + (m2 + i2) * nao] = *H++;
	    }
	  m2 += nao2;
	}
      m1 += nao1;
    }
  if (nmem != 0)
    free(a1);
  Py_RETURN_NONE;
}
 
*/
