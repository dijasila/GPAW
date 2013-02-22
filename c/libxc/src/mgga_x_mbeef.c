#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

#define XC_MGGA_X_MBEEF          209 /* mBEEF-v5 exchange*/


/*changes static with const*/
const XC(func_info_type) XC(func_info_mgga_x_mbeef) = {
  XC_MGGA_X_MBEEF,
  XC_EXCHANGE,
  "mBEEF",
  XC_FAMILY_MGGA,
  "mBEEF", 
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};


void XC(mgga_x_mbeef_init)(XC(mgga_type) *p)
{
  p->info = &XC(func_info_mgga_x_mbeef);

  p->lda_aux = (XC(lda_type) *) malloc(sizeof(XC(lda_type)));
  XC(lda_x_init)(p->lda_aux, XC_UNPOLARIZED, 3, XC_NON_RELATIVISTIC);
}


void XC(mgga_x_mbeef_end)(XC(mgga_type) *p)
{
  free(p->lda_aux);
}


static void
mbeef_exchange(XC(mgga_type) *pt, double *rho, double sigma, double tau_,
	    double *energy, double *dedd, double *vsigma, double *dedtau)
{
  double gdms, s2, ds2dd, ds2dsigma;
  double k, xi, xj, tmp, tmp1, tmp2, tmp3, tmp4, dxids2, dxjdalpha;
  double alpha, tau, tauw, tau_lsda, aux;
  double dtau_lsdadd, dalphadsigma, dalphadtau, dalphadd;
  double exunif, vxunif;
  double Fx, dFdxi, dFdxj;

  /* HEG energy and potential */
  XC(lda_vxc)(pt->lda_aux, rho, &exunif, &vxunif);

  /* calculate |nabla rho|^2 */
  gdms = max(MIN_GRAD*MIN_GRAD, sigma);
  
  /* reduced density gradient in transformation t1(s) */
  s2 = gdms/(4.0*POW(3.0*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));
  ds2dd = -(8.0/3.0)*s2/rho[0];
  ds2dsigma = 1.0/(4.0*POW(3.0*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));
  k = 6.5124; // PBEsol transformation
  tmp = k + s2;
  xi = 2.0 * s2 / tmp - 1.0;
  dxids2 = 2.0 * k / POW(tmp, 2.0);

  /* kinetic energy densities */
  tauw = max(gdms/(8.0*rho[0]), 1.0e-8);
  tau = max(tau_, tauw);
  aux = (3./10.) * POW((3.0*M_PI*M_PI), 2.0/3.0);
  tau_lsda = aux * POW(rho[0], 5.0/3.0); 
  dtau_lsdadd = aux * 5.0/3.0 * POW(rho[0], 2.0/3.0);

  /* alpha in transformation t2(a) */
  alpha = (tau - tauw)/tau_lsda;
  assert(alpha >= 0.0);
  tmp1 = POW(1.0 - POW(alpha, 2.0), 3.0);
  tmp2 = 1.0 + POW(alpha, 3.0) + POW(alpha, 6.0);
  xj = -1.0 * tmp1 / tmp2;
  tmp3 = -6.0*alpha +12.0*POW(alpha, 3.0) -6.0*POW(alpha, 5.0);
  tmp4 = 3.0*POW(alpha, 2.0) +6.0*POW(alpha, 5.0);
  dxjdalpha = (6.0*alpha + 3.0*POW(alpha, 2.0) - 12.0*POW(alpha, 3.0)
    - 3.0*POW(alpha, 4.0) + 12.0*POW(alpha, 5.0) - 3.0*POW(alpha, 6.0)
    - 12.0*POW(alpha, 7.0) + 3.0*POW(alpha, 8.0) + 6.0*POW(alpha, 9.0)) / POW(tmp2, 2.0);

  if(ABS(tau - tauw) < 1.0e-5)
    {
    dalphadsigma = 0.0;
    dalphadtau = 0.0;
    dalphadd = 0.0; 
    }
  else
    {
    dalphadtau = 1.0/tau_lsda;
    dalphadsigma = -1.0/(tau_lsda*8.0*rho[0]);
    dalphadd = (tauw/rho[0]*tau_lsda - (tau-tauw)*dtau_lsdadd) / POW(tau_lsda, 2.0); 
    }

  /* product exchange enhancement factor and derivatives */
  // mgga_pbesol fit (pbesol transformation with pbesol correlation)
  double coefs[64] = {
         1.18029330e+00,   8.53027860e-03,  -1.02312143e-01,
         6.85757490e-02,  -6.61294786e-03,  -2.84176163e-02,
         5.54283363e-03,   3.95434277e-03,  -1.98479086e-03,
         1.00339208e-01,  -4.34643460e-02,  -1.82177954e-02,
         1.62638575e-02,  -8.84148272e-03,  -9.57417512e-03,
         9.40675747e-03,   6.37590839e-03,  -8.79090772e-03,
        -1.50103636e-02,   2.80678872e-02,  -1.82911291e-02,
        -1.88495102e-02,   1.69805915e-07,  -2.76524680e-07,
         1.44642135e-03,  -3.03347141e-03,   2.93253041e-03,
        -8.45508103e-03,   6.31891628e-03,  -8.96771404e-03,
        -2.65114646e-08,   5.05920757e-08,   6.65511484e-04,
         1.19130546e-03,   1.82906057e-03,   3.39308972e-03,
        -7.90811707e-08,   1.62238741e-07,  -4.16393106e-08,
         5.54588743e-08,  -1.16063796e-04,   8.22139896e-04,
        -3.51041030e-04,   8.96739466e-04,   2.09603871e-08,
        -3.76702959e-08,   2.36391411e-08,  -3.38128188e-08,
        -5.54173599e-06,  -5.14204676e-05,   6.68980219e-09,
        -2.16860568e-08,   9.12223751e-09,  -1.38472194e-08,
         6.94482484e-09,  -7.74224962e-09,   7.36062570e-07,
        -9.40351563e-06,  -2.23014657e-09,   6.74910119e-09,
        -4.93824365e-09,   8.50272392e-09,  -6.91592964e-09,
         8.88525527e-09 };

  int order = 8; 
  double Li[order];
  double dLi[order];
  double Lj[order];
  double dLj[order];

  /* initializing */
  Li[0] = 1.0;
  Li[1] = xi;
  dLi[0] = 0.0;
  dLi[1] = 1.0;
  Lj[0] = 1.0;
  Lj[1] = xj;
  dLj[0] = 0.0;
  dLj[1] = 1.0;
  Fx = 0.0;
  dFdxi = 0.0;
  dFdxj = 0.0;

  /* recursively building polynomia and their derivatives */
  for(int i = 2; i < order; i++)
    {
    Li[i] = 2.0 * xi * Li[i-1] - Li[i-2] - (xi * Li[i-1] - Li[i-2])/i;
    Lj[i] = 2.0 * xj * Lj[i-1] - Lj[i-2] - (xj * Lj[i-1] - Lj[i-2])/i;
    dLi[i] = i * Li[i-1] + xi * dLi[i-1];
    dLj[i] = i * Lj[i-1] + xj * dLj[i-1];
    }

  /* building enhancement factor and derivatives */
  int m = 0;
  for(int j = 0; j < order; j++)
    {
    for(int i = 0; i < order; i++)
      {
      Fx += coefs[m] * Li[i] * Lj[j];
      dFdxi += coefs[m] * dLi[i] * Lj[j];
      dFdxj += coefs[m] * dLj[j] * Li[i];
      m += 1;
      }
    }

  /* exchange energy */
  *energy = exunif * Fx * rho[0];

  /* exunif is energy per particle already
     so we multiply by n the terms with exunif*/

  *dedd = vxunif * Fx + exunif * rho[0] * (dFdxi * dxids2 * ds2dd + dFdxj * dxjdalpha * dalphadd);
  *vsigma = exunif * rho[0] * (dFdxi * dxids2 * ds2dsigma + dFdxj * dxjdalpha * dalphadsigma);
  *dedtau = exunif * rho[0] * dFdxj * dxjdalpha * dalphadtau;
}


void
XC(mgga_x_mbeef)(XC(mgga_type) *p, double *rho, double *sigma, double *tau,
	    double *e, double *dedd, double *vsigma, double *dedtau)
{
  if(p->nspin == XC_UNPOLARIZED)
    {
    double en;
    mbeef_exchange(p, rho, sigma[0], tau[0], &en, dedd, vsigma, dedtau);
    *e = en/(rho[0] + rho[1]);
    }
  else
    {
    /* The spin polarized version is handled using the exact spin scaling
          Ex[n1, n2] = (Ex[2*n1] + Ex[2*n2])/2
    */

    double e2na, e2nb, rhoa[2], rhob[2];
    double vsigmapart[3];

    rhoa[0] = 2.0 * rho[0];
    rhoa[1] = 0.0;
    rhob[0] = 2.0 * rho[1];
    rhob[1] = 0.0;

    mbeef_exchange(p, rhoa, 4.0*sigma[0], 2.0*tau[0], &e2na, &(dedd[0]), &(vsigmapart[0]), &(dedtau[0]));
    mbeef_exchange(p, rhob, 4.0*sigma[2], 2.0*tau[1], &e2nb, &(dedd[1]), &(vsigmapart[2]), &(dedtau[1]));

    *e = (e2na + e2nb) / (2.0*(rho[0] + rho[1]));
    vsigma[0] = 2.0 * vsigmapart[0];
    vsigma[2] = 2.0 * vsigmapart[2];
    }
}
