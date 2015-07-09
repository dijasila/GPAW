/*  Copyright (C) 2010 CAMd
 *  Please see the accompanying LICENSE file for further information. */

#include "extensions.h"
#include "spline.h"
#include "lfc.h"
#include "bmgs/spherical_harmonics.h"

PyObject* second_derivative(LFCObject *lfc, PyObject *args)
{
  PyArrayObject* a_xG_obj;
  PyArrayObject* c_xMvv_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyObject* spline_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;
  int q;

  if (!PyArg_ParseTuple(args, "OOOOOOOi", &a_xG_obj, &c_xMvv_obj,
                        &h_cv_obj, &n_c_obj,
                        &spline_M_obj, &beg_c_obj,
                        &pos_Wc_obj, &q))
    return NULL; 

  // number of dimensions of a_xG
  int nd = PyArray_NDIM(a_xG_obj);
  // length of each of these
  npy_intp* dims = PyArray_DIMS(a_xG_obj);
  // length of x, whatever it is
  int nx = PyArray_MultiplyList(dims, nd - 3);
  // total entries in grid a_G
  int nG = PyArray_MultiplyList(dims + nd - 3, 3);
  // number of splines
  int nM = PyArray_DIM(c_xMvv_obj, PyArray_NDIM(c_xMvv_obj) - 3);
  
  // convert input parameters to c
  const double* h_cv = (const double*)PyArray_DATA(h_cv_obj);
  const long* n_c = (const long*)PyArray_DATA(n_c_obj);
  const double (*pos_Wc)[3] = (const double (*)[3])PyArray_DATA(pos_Wc_obj);

  long* beg_c = LONGP(beg_c_obj);

  // ???
  const double Y00dv = lfc->dv / sqrt(4.0 * M_PI);

  if (!lfc->bloch_boundary_conditions) {
    // convert input parameters to c
    const double* a_G = (const double*)PyArray_DATA(a_xG_obj);
    double* c_Mvv = (double*)PyArray_DATA(c_xMvv_obj);
    // Loop over number of x-dimension in a_xG (not relevant yet)
    for (int x = 0; x < nx; x++) {
      // JJs old stuff
      GRID_LOOP_START(lfc, -1) {
        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        for (int G = Ga; G < Gb; G++) {
          for (int i = 0; i < ni; i++) {
            LFVolume* vol = volume_i + i;
            int M = vol->M;
            double* c_mvv = c_Mvv + 9 * M;
            const bmgsspline* spline = (const bmgsspline*) \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;

            int nm = vol->nm;
            int l = (nm-1)/2;
            double x = xG - pos_Wc[vol->W][0];
            double y = yG - pos_Wc[vol->W][1];
            double z = zG - pos_Wc[vol->W][2];
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);

            double af;
            double dfdr;
            double d2fdr2;
            bmgs_get_value_and_second_derivative(spline, r, &af, &dfdr, &d2fdr2);
            af *= a_G[G] * lfc->dv;
            dfdr *= a_G[G] * lfc->dv;
            d2fdr2 *= a_G[G] * lfc->dv;
            // Second derivative has 4 terms ( or 3 )
            // 1. Term: a*f* d^2(Y * r^l)/dxdy
            double afd2rlYdxdy_m[nm];
            spherical_harmonics_derivative_xx(l, af, x, y, z, r2, afd2rlYdxdy_m);
            for (int m = 0; m < nm; m++)
              c_mvv[9 * m] += afd2rlYdxdy_m[m];
            spherical_harmonics_derivative_xy(l, af, x, y, z, r2, afd2rlYdxdy_m);
            for (int m = 0; m < nm; m++){
              c_mvv[9 * m + 1] += afd2rlYdxdy_m[m];
              c_mvv[9 * m + 3] += afd2rlYdxdy_m[m];}
            spherical_harmonics_derivative_xz(l, af, x, y, z, r2, afd2rlYdxdy_m);
            for (int m = 0; m < nm; m++){
              c_mvv[9 * m + 2] += afd2rlYdxdy_m[m];
              c_mvv[9 * m + 6] += afd2rlYdxdy_m[m];}
            spherical_harmonics_derivative_yy(l, af, x, y, z, r2, afd2rlYdxdy_m);
            for (int m = 0; m < nm; m++)
              c_mvv[9 * m + 4] += afd2rlYdxdy_m[m];
            spherical_harmonics_derivative_yz(l, af, x, y, z, r2, afd2rlYdxdy_m);
            for (int m = 0; m < nm; m++){
              c_mvv[9 * m + 5] += afd2rlYdxdy_m[m];
              c_mvv[9 * m + 7] += afd2rlYdxdy_m[m];}
            spherical_harmonics_derivative_zz(l, af, x, y, z, r2, afd2rlYdxdy_m);
            for (int m = 0; m < nm; m++)
              c_mvv[9 * m + 8] += afd2rlYdxdy_m[m];
            // 2. and 3. Term: a * df/dr x/r * d(Y * r^l)/dy + a * df/dr * y/r * d(Y * r^l)/dx
            if (r > 1e-15){
              double adfdxdrlYdy_m[nm];
              double adfdydrlYdx_m[nm];
              double dfdrxr[3];
              dfdrxr[0] = dfdr * x / r;
              dfdrxr[1] = dfdr * y / r;
              dfdrxr[2] = dfdr * z / r;
              spherical_harmonics_derivative_x(l, dfdrxr[0], x, y, z, r2, adfdxdrlYdy_m);
              for (int m = 0; m < nm; m++)
                c_mvv[9 * m] += 2. * adfdxdrlYdy_m[m] ;
              spherical_harmonics_derivative_y(l, dfdrxr[0], x, y, z, r2, adfdxdrlYdy_m);
              spherical_harmonics_derivative_x(l, dfdrxr[1], x, y, z, r2, adfdydrlYdx_m);
              for (int m = 0; m < nm; m++){
                c_mvv[9 * m + 1] += adfdxdrlYdy_m[m] +  adfdydrlYdx_m[m];
                c_mvv[9 * m + 3] += adfdxdrlYdy_m[m] +  adfdydrlYdx_m[m];}
              spherical_harmonics_derivative_x(l, dfdrxr[0], x, y, z, r2, adfdxdrlYdy_m);
              spherical_harmonics_derivative_x(l, dfdrxr[2], x, y, z, r2, adfdydrlYdx_m);
              for (int m = 0; m < nm; m++){
                c_mvv[9 * m + 2] += adfdxdrlYdy_m[m] +  adfdydrlYdx_m[m];
                c_mvv[9 * m + 6] += adfdxdrlYdy_m[m] +  adfdydrlYdx_m[m];}
              spherical_harmonics_derivative_y(l, dfdrxr[1], x, y, z, r2, adfdxdrlYdy_m);
              for (int m = 0; m < nm; m++)
                c_mvv[9 * m + 4] += 2. * adfdxdrlYdy_m[m] ;
              spherical_harmonics_derivative_z(l, dfdrxr[1], x, y, z, r2, adfdxdrlYdy_m);
              spherical_harmonics_derivative_y(l, dfdrxr[2], x, y, z, r2, adfdydrlYdx_m);
              for (int m = 0; m < nm; m++){
                c_mvv[9 * m + 5] += adfdxdrlYdy_m[m] +  adfdydrlYdx_m[m];
                c_mvv[9 * m + 7] += adfdxdrlYdy_m[m] +  adfdydrlYdx_m[m];}
              spherical_harmonics_derivative_z(l, dfdrxr[2], x, y, z, r2, adfdxdrlYdy_m);
              for (int m = 0; m < nm; m++)
                c_mvv[9 * m + 8] += 2. * adfdxdrlYdy_m[m] ;
            }
            // 4. Term: a * d^2f/dr^2 * (x*y)/r^2 * Y * r^l - a df/dr (x*y)/r^3       * Y * r^l
            //      OR  a * d^2f/dr^2 *   x^2/r^2 * Y * r^l + a df/dr (1/r - x^2/r^3) * Y * r^l
            if (r > 1e-15){
              double ad2fdxdyrlY_m[nm];
              double d2fdr2xyr2;
              d2fdr2xyr2 = d2fdr2 * x * x / r2;
              d2fdr2xyr2 += dfdr * (r - x * x / r) / r2;
              spherical_harmonics(l, d2fdr2xyr2, x, y, z, r2, ad2fdxdyrlY_m);
              for (int m = 0; m < nm; m++)
                c_mvv[9 * m] += ad2fdxdyrlY_m[m];
              d2fdr2xyr2 = d2fdr2 * x * y / r2;
              d2fdr2xyr2 += dfdr * (0. - x * y / r) / r2;
              spherical_harmonics(l, d2fdr2xyr2, x, y, z, r2, ad2fdxdyrlY_m);
              for (int m = 0; m < nm; m++){
                c_mvv[9 * m + 1] += ad2fdxdyrlY_m[m];
                c_mvv[9 * m + 3] += ad2fdxdyrlY_m[m];}
              d2fdr2xyr2 = d2fdr2 * x * z / r2;
              d2fdr2xyr2 += dfdr * (0. - x * z / r) / r2;
              spherical_harmonics(l, d2fdr2xyr2, x, y, z, r2, ad2fdxdyrlY_m);
              for (int m = 0; m < nm; m++){
                c_mvv[9 * m + 2] += ad2fdxdyrlY_m[m];
                c_mvv[9 * m + 6] += ad2fdxdyrlY_m[m];}
              d2fdr2xyr2 = d2fdr2 * y * y / r2;
              d2fdr2xyr2 += dfdr * (r - y * y / r) / r2;
              spherical_harmonics(l, d2fdr2xyr2, x, y, z, r2, ad2fdxdyrlY_m);
              for (int m = 0; m < nm; m++)
                c_mvv[9 * m + 4] += ad2fdxdyrlY_m[m];
              d2fdr2xyr2 = d2fdr2 * y * z / r2;
              d2fdr2xyr2 += dfdr * (0. - y * z / r) / r2;
              spherical_harmonics(l, d2fdr2xyr2, x, y, z, r2, ad2fdxdyrlY_m);
              for (int m = 0; m < nm; m++){
                c_mvv[9 * m + 5] += ad2fdxdyrlY_m[m];
                c_mvv[9 * m + 7] += ad2fdxdyrlY_m[m];}
              d2fdr2xyr2 = d2fdr2 * z * z / r2;
              d2fdr2xyr2 += dfdr * (r - z * z / r) / r2;
              spherical_harmonics(l, d2fdr2xyr2, x, y, z, r2, ad2fdxdyrlY_m);
              for (int m = 0; m < nm; m++)
                c_mvv[9 * m + 8] += ad2fdxdyrlY_m[m];
            }
          }
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, -1);
      c_Mvv += 9 * nM;
      a_G += nG;
    }
  }
  else {
    printf("ATTENTION: This branch of LFC second derivatives doens't work with l>0!!!\n");
    const complex double* a_G = (const complex double*)PyArray_DATA(a_xG_obj);
    complex double* c_Mvv = (complex double*)PyArray_DATA(c_xMvv_obj);

    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, q) {
        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        for (int G = Ga; G < Gb; G++) {
          for (int i = 0; i < ni; i++) {
            LFVolume* vol = volume_i + i;
            int M = vol->M;
            complex double* c_mvv = c_Mvv + 9 * M;
            const bmgsspline* spline = (const bmgsspline*) \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
              
            double x = xG - pos_Wc[vol->W][0];
            double y = yG - pos_Wc[vol->W][1];
            double z = zG - pos_Wc[vol->W][2];
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double dfdror;

            // use bmgs_get_value_and_derivative instead ??!!
            int bin = r / spline->dr;
            assert(bin <= spline->nbins);
            double u = r - bin * spline->dr;
            double* s = spline->data + 4 * bin;

            if (bin == 0)
              dfdror = 2.0 * s[2] + 3.0 * s[3] * r;
            else
              dfdror = (s[1] + u * (2.0 * s[2] + u * 3.0 * s[3])) / r;
            // phase added here
            complex double a = a_G[G] * phase_i[i] * Y00dv;
            // dfdror *= a;
            c_mvv[0] += a * dfdror;
            c_mvv[4] += a * dfdror;
            c_mvv[8] += a * dfdror;
            if (r > 1e-15) {
              double b = (2.0 * s[2] + 6.0 * s[3] * u - dfdror) / r2;
              c_mvv[0] += a * b * x * x;
              c_mvv[1] += a * b * x * y;
              c_mvv[2] += a * b * x * z;
              c_mvv[3] += a * b * y * x;
              c_mvv[4] += a * b * y * y;
              c_mvv[5] += a * b * y * z;
              c_mvv[6] += a * b * z * x;
              c_mvv[7] += a * b * z * y;
              c_mvv[8] += a * b * z * z;
            }
          }
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, q);
      c_Mvv += 9 * nM;
      a_G += nG;
    }
  }
  Py_RETURN_NONE;
}

PyObject* add_derivative(LFCObject *lfc, PyObject *args)
{
  // Coefficients for the lfc's
  PyArrayObject* c_xM_obj;
  // Array 
  PyArrayObject* a_xG_obj;
  PyArrayObject* h_cv_obj;
  PyArrayObject* n_c_obj;
  PyObject* spline_M_obj;
  PyArrayObject* beg_c_obj;
  PyArrayObject* pos_Wc_obj;
  // Atom index
  int a;
  // Cartesian coordinate
  int v;
  // k-point index
  int q;

  if (!PyArg_ParseTuple(args, "OOOOOOOiii", &c_xM_obj, &a_xG_obj,
                        &h_cv_obj, &n_c_obj, &spline_M_obj, &beg_c_obj,
                        &pos_Wc_obj, &a, &v, &q))
    return NULL;

  // Number of dimensions
  int nd = PyArray_NDIM(a_xG_obj);
  // Array with lengths of array dimensions
  npy_intp* dims = PyArray_DIMS(a_xG_obj);
  // Number of extra dimensions
  int nx = PyArray_MultiplyList(dims, nd - 3);
  // Number of grid points
  int nG = PyArray_MultiplyList(dims + nd - 3, 3);
  // Number of lfc's 
  int nM = PyArray_DIM(c_xM_obj, PyArray_NDIM(c_xM_obj) - 1);

  const double* h_cv = (const double*)PyArray_DATA(h_cv_obj);
  const long* n_c = (const long*)PyArray_DATA(n_c_obj);
  const double (*pos_Wc)[3] = (const double (*)[3])PyArray_DATA(pos_Wc_obj);

  long* beg_c = LONGP(beg_c_obj);

  if (!lfc->bloch_boundary_conditions) {

    const double* c_M = (const double*)PyArray_DATA(c_xM_obj);
    double* a_G = (double*)PyArray_DATA(a_xG_obj);
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, -1) {

        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        // Grid point position
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        // Loop over grid points in current stride
        for (int G = Ga; G < Gb; G++) {
          // Loop over volumes at current grid point
          for (int i = 0; i < ni; i++) {

            LFVolume* vol = volume_i + i;
            int M = vol->M;
            // Check that the volume belongs to the atom in consideration later
            int W = vol->W;
            int nm = vol->nm;
            int l = (nm - 1) / 2;

            const bmgsspline* spline = (const bmgsspline*)              \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;
              
            double x = xG - pos_Wc[W][0];
            double y = yG - pos_Wc[W][1];
            double z = zG - pos_Wc[W][2];
            double R_c[] = {x, y, z};
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double f;
            double dfdr;

            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);
            
            // First contribution: f * d(r^l * Y)/dv
            double fdrlYdx_m[nm];
            if (v == 0)
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdrlYdx_m);
            else if (v == 1)
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdrlYdx_m);
            else
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdrlYdx_m);

            for (int m = 0; m < nm; m++)
              a_G[G] += fdrlYdx_m[m] * c_M[M + m];

            // Second contribution: r^(l-1) * Y * df/dr * R_v
            if (r > 1e-15) {
              double rlm1Ydfdr_m[nm]; // r^(l-1) * Y * df/dr
              double rm1dfdr = 1. / r * dfdr;
              spherical_harmonics(l, rm1dfdr, x, y, z, r2, rlm1Ydfdr_m);
              for (int m = 0; m < nm; m++)
                  a_G[G] += rlm1Ydfdr_m[m] * R_c[v] * c_M[M + m];
            }
          }
          // Update coordinates of current grid point
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, -1);
      c_M += nM;
      a_G += nG;
    }
  }
  else {
    const double complex* c_M = (const double complex*)PyArray_DATA(c_xM_obj);
    double complex* a_G = (double complex*)PyArray_DATA(a_xG_obj);
    for (int x = 0; x < nx; x++) {
      GRID_LOOP_START(lfc, q) {

        // In one grid loop iteration, only i2 changes.
        int i2 = Ga % n_c[2] + beg_c[2];
        int i1 = (Ga / n_c[2]) % n_c[1] + beg_c[1];
        int i0 = Ga / (n_c[2] * n_c[1]) + beg_c[0];
        // Grid point position
        double xG = h_cv[0] * i0 + h_cv[3] * i1 + h_cv[6] * i2;
        double yG = h_cv[1] * i0 + h_cv[4] * i1 + h_cv[7] * i2;
        double zG = h_cv[2] * i0 + h_cv[5] * i1 + h_cv[8] * i2;
        // Loop over grid points in current stride
        for (int G = Ga; G < Gb; G++) {
          // Loop over volumes at current grid point
          for (int i = 0; i < ni; i++) {
            // Phase of volume
            double complex conjphase = conj(phase_i[i]);
            LFVolume* vol = volume_i + i;
            int M = vol->M;
            // Check that the volume belongs to the atom in consideration later
            int W = vol->W;
            int nm = vol->nm;
            int l = (nm - 1) / 2;

            const bmgsspline* spline = (const bmgsspline*)              \
              &((const SplineObject*)PyList_GetItem(spline_M_obj, M))->spline;

            double x = xG - pos_Wc[W][0];
            double y = yG - pos_Wc[W][1];
            double z = zG - pos_Wc[W][2];
            double R_c[] = {x, y, z};
            double r2 = x * x + y * y + z * z;
            double r = sqrt(r2);
            double f;
            double dfdr;

            bmgs_get_value_and_derivative(spline, r, &f, &dfdr);

            // First contribution: f * d(r^l * Y)/dv
            double fdrlYdx_m[nm];
            if (v == 0)
              spherical_harmonics_derivative_x(l, f, x, y, z, r2, fdrlYdx_m);
            else if (v == 1)
              spherical_harmonics_derivative_y(l, f, x, y, z, r2, fdrlYdx_m);
            else
              spherical_harmonics_derivative_z(l, f, x, y, z, r2, fdrlYdx_m);

            for (int m = 0; m < nm; m++)
              a_G[G] += fdrlYdx_m[m] * c_M[M + m] * conjphase;

            // Second contribution: r^(l-1) * Y * df/dr * R_v
            if (r > 1e-15) {
              double rlm1Ydfdr_m[nm]; // r^(l-1) * Y * df/dr
              double rm1dfdr = 1. / r * dfdr;
              spherical_harmonics(l, rm1dfdr, x, y, z, r2, rlm1Ydfdr_m);
              for (int m = 0; m < nm; m++)
                  a_G[G] += rlm1Ydfdr_m[m] * R_c[v] * c_M[M + m] * conjphase;
            }
          }
          // Update coordinates of current grid point
          xG += h_cv[6];
          yG += h_cv[7];
          zG += h_cv[8];
        }
      }
      GRID_LOOP_STOP(lfc, q);
      c_M += nM;
      a_G += nG;
    }
  }
  Py_RETURN_NONE;
}
