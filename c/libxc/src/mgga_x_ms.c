#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "util.h"

/************************************************************************
  Meta-GGA exchange "made simple"
************************************************************************/

#define XC_MGGA_X_MS0          210 /* MS0: Sun et al., JCP 137, 051101 (2012) */
#define XC_MGGA_X_MS1          211 /* MS1: Sun et al., JCP 138, 044113 (2013) */
#define XC_MGGA_X_MS2          212 /* MS2: Sun et al., JCP 138, 044113 (2013) */

/* Fx = F1(p)*[1 - f(alpha)] + F2(p)     */
/* Fx = F1(p) + f(alpha)*[F0(p) - F1(p)] */
/* G  = 1 + K - K/(1 + x/K)              */
/* x  = c_x + mu*p, c_1 = 0, c_0 = c     */
/* f  = (1 - alpha**2)**3 / (1 + alpha**2 + alpha**6) */

/* parameters */
static const FLOAT c[3] = {
  0.28771, /* MS0 */
  0.18150, /* MS1 */
  0.14601  /* MS2 */
};
static const FLOAT kappa[3] = {
  0.29,    /* MS0 */
  0.404,   /* MS1 */
  0.504    /* MS2 */
};
static const FLOAT b[3] = {
  1.0,     /* MS0 */
  1.0,     /* MS1 */
  4.0      /* MS2 */
};
static FLOAT mu = 10.0/81.0;
static int func;

void XC(mgga_x_ms_init)(XC(mgga_type) *p)
{
  p->lda_aux = (XC(lda_type) *) malloc(sizeof(XC(lda_type)));
  XC(lda_x_init)(p->lda_aux, XC_UNPOLARIZED, 3, XC_NON_RELATIVISTIC);
  switch(p->info->number){
    case XC_MGGA_X_MS1:   func = 1; break;
    case XC_MGGA_X_MS2:   func = 2; break;
    default:              func = 0;
    }
}

void XC(mgga_x_ms_end)(XC(mgga_type) *p)
{
  free(p->lda_aux);
}

static void 
ms_f(FLOAT alpha, FLOAT *f, FLOAT *dfdalpha)
{
  FLOAT num   = POW(1.0 - POW(alpha, 2.0), 3.0);
  FLOAT denom = 1.0 + POW(alpha, 3.0) + b[func]*POW(alpha, 6.0);
  *f = num / denom;

  FLOAT q = (2.0 + alpha - 4.0*POW(alpha,2.0) - POW(alpha,3.0)
             + 2.0*POW(alpha,4.0) + 2.0*b[func]*POW(alpha,4.0) - POW(alpha,5.0)
             - 4.0*b[func]*POW(alpha,6.0) + POW(alpha,7.0) + 2.0*b[func]*POW(alpha,8.0));
  *dfdalpha = -3.0 * alpha * q / POW(denom, 2.0);
}

static void
ms_F(FLOAT p, FLOAT c_, FLOAT *F, FLOAT *dFdp)
{
  FLOAT x = c_*c[func] + mu*p;
  *F = 1.0 + kappa[func] - kappa[func]/(1.0 + x/kappa[func]);

  FLOAT denom = POW(1.0 + x/kappa[func], 2.0);
  *dFdp = mu / denom;
}

static void
ms_Fx(FLOAT p, FLOAT alpha, FLOAT *Fx, FLOAT *dFxdp, FLOAT *dFxda)
{
  FLOAT F0, F1, dF0dp, dF1dp;
  FLOAT f, dfda;
  ms_F(p, 1.0, &F0, &dF0dp);
  ms_F(p, 0.0, &F1, &dF1dp);
  ms_f(alpha, &f, &dfda);
  *Fx = F1 + f*(F0 - F1);

  *dFxdp = dF1dp + f*(dF0dp - dF1dp);
  *dFxda = dfda*(F0 - F1);
}

static void 
x_ms_para(XC(mgga_type) *pt, FLOAT *rho, FLOAT sigma, FLOAT tau_,
	    FLOAT *energy, FLOAT *dedn, FLOAT *vsigma, FLOAT *dedtau)
{
  FLOAT gdms, p, tau, tauw, alpha;
  FLOAT dpdn, dpdsigma;
  FLOAT dadn, dadsigma, dadtau;
  FLOAT Fx, dFxdp, dFxda;
  FLOAT tau_lsda, exunif, vxunif;
  FLOAT aux = (3.0/10.0) * POW((3.0*M_PI*M_PI), 2.0/3.0);

  /* uniform electron gas energy and potential */
  XC(lda_vxc)(pt->lda_aux, rho, &exunif, &vxunif);

  /* |nabla rho|**2 */
  gdms = max(MIN_GRAD*MIN_GRAD, sigma);
  
  /* p = s**2 */
  p = gdms/(4.0*POW(3*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));

  /* alpha = (tau - tauw) / tau_lsda */
  tauw = max(gdms/(8.0*rho[0]), 1.0e-12);
  tau = max(tau_, tauw);
  tau_lsda = aux * POW(rho[0], 5./3.); 
  alpha = (tau - tauw)/tau_lsda;

  /* derivatives */
  if(tau_lsda < 1.e-20)
    {
    dpdn     = 0.0;
    dpdsigma = 0.0;
    dadn     = 0.0;
    dadsigma = 0.0;
    dadtau   = 0.0;
    }
  else
    {
    dpdn     = -(8.0/3.0) * p/rho[0];
    dpdsigma =  1.0/(4.0*POW(3.0*M_PI*M_PI, 2.0/3.0)*POW(rho[0], 8.0/3.0));
    dadn     = -(5.0/3.0) * (tau/(rho[0]*tau_lsda) + dpdn); 
    dadsigma = -1.0/(8.0*tau_lsda*rho[0]);
    dadtau   =  1.0/tau_lsda;
    }

  /* get Fx(p, alpha) */
  ms_Fx(p, alpha, &Fx, &dFxdp, &dFxda);

  /* exchange energy */
  *energy = exunif * Fx * rho[0];

  /* derivatives */
  /* exunif is en per particle already so we multiply by n the terms with exunif */
  *dedn   = vxunif*Fx + exunif*rho[0]*(dFxdp*dpdn + dFxda*dadn);
  *vsigma = exunif*rho[0]*(dFxdp*dpdsigma + dFxda*dadsigma);
  *dedtau = exunif*rho[0]*(dFxda*dadtau);
}

void 
XC(mgga_x_ms)(XC(mgga_type) *p, FLOAT *rho, FLOAT *sigma, FLOAT *tau,
	       FLOAT *e, FLOAT *dedn, FLOAT *vsigma, FLOAT *dedtau)
{
  /* energy and derivatives */
  *e = 0.0;
  if(p->nspin == XC_UNPOLARIZED)
  {
    FLOAT en;
    x_ms_para(p, rho, sigma[0], tau[0], &en, dedn, vsigma, dedtau);
    *e = en / (rho[0] + rho[1]);
  }
  else
  { 
  /* The spin polarized version is handle using the exact spin scaling relation
     Ex[n1, n2] = (Ex[2*n1] + Ex[2*n2])/2 */
    FLOAT e2na, e2nb, rhoa[2], rhob[2];
    FLOAT vsigmapart[3]; 
    rhoa[0] = 2.0*rho[0];
    rhoa[1] = 0.0;
    rhob[0] = 2.0*rho[1];
    rhob[1] = 0.0;

    x_ms_para(p, rhoa, 4.0*sigma[0], 2.0*tau[0], &e2na, &(dedn[0]), &(vsigmapart[0]), &(dedtau[0]));
    x_ms_para(p, rhob, 4.0*sigma[2], 2.0*tau[1], &e2nb, &(dedn[1]), &(vsigmapart[2]), &(dedtau[1]));

    *e = (e2na + e2nb) / (2.0*(rho[0] + rho[1]));
    vsigma[0] = 2.0*vsigmapart[0];
    vsigma[2] = 2.0*vsigmapart[2];
  }
}

const XC(func_info_type) XC(func_info_mgga_x_ms0) = {
  XC_MGGA_X_MS0,
  XC_EXCHANGE,
  "Sun et al., JCP 137, 051101 (2012)",
  XC_FAMILY_MGGA,
  "Sun et al., JCP 137, 051101 (2012)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};

const XC(func_info_type) XC(func_info_mgga_x_ms1) = {
  XC_MGGA_X_MS1,
  XC_EXCHANGE,
  "Sun et al., JCP 138, 044113 (2013)",
  XC_FAMILY_MGGA,
  "Sun et al., JCP 138, 044113 (2013)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};

const XC(func_info_type) XC(func_info_mgga_x_ms2) = {
  XC_MGGA_X_MS2,
  XC_EXCHANGE,
  "Sun et al., JCP 138, 044113 (2013)",
  XC_FAMILY_MGGA,
  "Sun et al., JCP 138, 044113 (2013)",
  XC_PROVIDES_EXC | XC_PROVIDES_VXC
};
