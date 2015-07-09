/*  Copyright (C) 2003-2007  CAMP
 *  Copyright (C) 2007-2009  CAMd
 *  Please see the accompanying LICENSE file for further information. */

#define spherical_harmonics(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0.28209479177387814;\
            break;\
          case 1:\
            p[0] = f * 0.48860251190291992 * y;\
            p[1] = f * 0.48860251190291992 * z;\
            p[2] = f * 0.48860251190291992 * x;\
            break;\
          case 2:\
            p[0] = f * 1.0925484305920792 * x*y;\
            p[1] = f * 1.0925484305920792 * y*z;\
            p[2] = f * 0.31539156525252005 * (-r2+3*z*z);\
            p[3] = f * 1.0925484305920792 * x*z;\
            p[4] = f * 0.54627421529603959 * (-y*y+x*x);\
            break;\
          case 3:\
            p[0] = f * 0.59004358992664352 * (-y*y*y+3*x*x*y);\
            p[1] = f * 2.8906114426405538 * x*y*z;\
            p[2] = f * 0.45704579946446577 * (5*y*z*z-y*r2);\
            p[3] = f * 0.3731763325901154 * (-3*z*r2+5*z*z*z);\
            p[4] = f * 0.45704579946446577 * (-x*r2+5*x*z*z);\
            p[5] = f * 1.4453057213202769 * (-y*y*z+x*x*z);\
            p[6] = f * 0.59004358992664352 * (x*x*x-3*x*y*y);\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * (x*x*x*y-x*y*y*y);\
            p[1] = f * 1.7701307697799307 * (3*x*x*y*z-y*y*y*z);\
            p[2] = f * 0.94617469575756008 * (-x*y*r2+7*x*y*z*z);\
            p[3] = f * 0.66904654355728921 * (-3*y*z*r2+7*y*z*z*z);\
            p[4] = f * 0.10578554691520431 * (3*r2*r2-30*z*z*r2+35*z*z*z*z);\
            p[5] = f * 0.66904654355728921 * (7*x*z*z*z-3*x*z*r2);\
            p[6] = f * 0.47308734787878004 * (y*y*r2+7*x*x*z*z-x*x*r2-7*y*y*z*z);\
            p[7] = f * 1.7701307697799307 * (x*x*x*z-3*x*y*y*z);\
            p[8] = f * 0.62583573544917614 * (-6*x*x*y*y+x*x*x*x+y*y*y*y);\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_x(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0.48860251190291992;\
            break;\
          case 2:\
            p[0] = f * 1.0925484305920792 * y;\
            p[1] = f * 0;\
            p[2] = f * 0.63078313050504009 * -x;\
            p[3] = f * 1.0925484305920792 * z;\
            p[4] = f * 1.0925484305920792 * x;\
            break;\
          case 3:\
            p[0] = f * 3.5402615395598613 * x*y;\
            p[1] = f * 2.8906114426405538 * y*z;\
            p[2] = f * 0.91409159892893155 * -x*y;\
            p[3] = f * 2.2390579955406924 * -x*z;\
            p[4] = f * 0.45704579946446577 * (-r2-2*x*x+5*z*z);\
            p[5] = f * 2.8906114426405538 * x*z;\
            p[6] = f * 1.7701307697799307 * (-y*y+x*x);\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * (-y*y*y+3*x*x*y);\
            p[1] = f * 10.620784618679583 * x*y*z;\
            p[2] = f * 0.94617469575756008 * (7*y*z*z-y*r2-2*x*x*y);\
            p[3] = f * 4.0142792613437353 * -x*y*z;\
            p[4] = f * 1.2694265629824517 * (x*r2-5*x*z*z);\
            p[5] = f * 0.66904654355728921 * (-3*z*r2-6*x*x*z+7*z*z*z);\
            p[6] = f * 0.94617469575756008 * (-x*r2-x*x*x+x*y*y+7*x*z*z);\
            p[7] = f * 5.3103923093397913 * (-y*y*z+x*x*z);\
            p[8] = f * 2.5033429417967046 * (-3*x*y*y+x*x*x);\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_y(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0.48860251190291992;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 1.0925484305920792 * x;\
            p[1] = f * 1.0925484305920792 * z;\
            p[2] = f * 0.63078313050504009 * -y;\
            p[3] = f * 0;\
            p[4] = f * 1.0925484305920792 * -y;\
            break;\
          case 3:\
            p[0] = f * 1.7701307697799307 * (-y*y+x*x);\
            p[1] = f * 2.8906114426405538 * x*z;\
            p[2] = f * 0.45704579946446577 * (-2*y*y-r2+5*z*z);\
            p[3] = f * 2.2390579955406924 * -y*z;\
            p[4] = f * 0.91409159892893155 * -x*y;\
            p[5] = f * 2.8906114426405538 * -y*z;\
            p[6] = f * 3.5402615395598613 * -x*y;\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * (x*x*x-3*x*y*y);\
            p[1] = f * 5.3103923093397913 * (-y*y*z+x*x*z);\
            p[2] = f * 0.94617469575756008 * (-x*r2-2*x*y*y+7*x*z*z);\
            p[3] = f * 0.66904654355728921 * (-6*y*y*z-3*z*r2+7*z*z*z);\
            p[4] = f * 1.2694265629824517 * (-5*y*z*z+y*r2);\
            p[5] = f * 4.0142792613437353 * -x*y*z;\
            p[6] = f * 0.94617469575756008 * (y*y*y-7*y*z*z+y*r2-x*x*y);\
            p[7] = f * 10.620784618679583 * -x*y*z;\
            p[8] = f * 2.5033429417967046 * (y*y*y-3*x*x*y);\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_z(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0.48860251190291992;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 0;\
            p[1] = f * 1.0925484305920792 * y;\
            p[2] = f * 1.2615662610100802 * z;\
            p[3] = f * 1.0925484305920792 * x;\
            p[4] = f * 0;\
            break;\
          case 3:\
            p[0] = f * 0;\
            p[1] = f * 2.8906114426405538 * x*y;\
            p[2] = f * 3.6563663957157262 * y*z;\
            p[3] = f * 1.1195289977703462 * (-r2+3*z*z);\
            p[4] = f * 3.6563663957157262 * x*z;\
            p[5] = f * 1.4453057213202769 * (-y*y+x*x);\
            p[6] = f * 0;\
            break;\
          case 4:\
            p[0] = f * 0;\
            p[1] = f * 1.7701307697799307 * (-y*y*y+3*x*x*y);\
            p[2] = f * 11.354096349090721 * x*y*z;\
            p[3] = f * 2.0071396306718676 * (5*y*z*z-y*r2);\
            p[4] = f * 1.6925687506432689 * (-3*z*r2+5*z*z*z);\
            p[5] = f * 2.0071396306718676 * (-x*r2+5*x*z*z);\
            p[6] = f * 5.6770481745453605 * (-y*y*z+x*x*z);\
            p[7] = f * 1.7701307697799307 * (x*x*x-3*x*y*y);\
            p[8] = f * 0;\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_xx(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * -0.63078313050504009;\
            p[3] = f * 0;\
            p[4] = f * 1.0925484305920792 ;\
            break;\
          case 3:\
            p[0] = f * 3.5402615395598613 * y;\
            p[1] = f * 0;\
            p[2] = f * 0.91409159892893155 * -y;\
            p[3] = f * 2.2390579955406924 * -z;\
            p[4] = f * 0.45704579946446577 * -6*x;\
            p[5] = f * 2.8906114426405538 * z;\
            p[6] = f * 1.7701307697799307 * 2*x;\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * (6*x*y);\
            p[1] = f * 10.620784618679583 * y*z;\
            p[2] = f * 0.94617469575756008 * (-6*x*y);\
            p[3] = f * 4.0142792613437353 * -y*z;\
            p[4] = f * 1.2694265629824517 * (r2+2*x*x-5*z*z);\
            p[5] = f * 0.66904654355728921 * (-6*z*x-12*x*z);\
            p[6] = f * 0.94617469575756008 * (-r2-5*x*x+y*y+7*z*z);\
            p[7] = f * 5.3103923093397913 * (2*x*z);\
            p[8] = f * 2.5033429417967046 * (-3*y*y+3*x*x);\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_yy(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * -0.63078313050504009;\
            p[3] = f * 0;\
            p[4] = f * -1.0925484305920792;\
            break;\
          case 3:\
            p[0] = f * 1.7701307697799307 * -2*y;\
            p[1] = f * 0;\
            p[2] = f * 0.45704579946446577 * (-4*y-2*y);\
            p[3] = f * 2.2390579955406924 * -z;\
            p[4] = f * 0.91409159892893155 * -x;\
            p[5] = f * 2.8906114426405538 * -z;\
            p[6] = f * 3.5402615395598613 * -x;\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * -6*x*y;\
            p[1] = f * 5.3103923093397913 * -2*y*z;\
            p[2] = f * 0.94617469575756008 * -6*x*y;\
            p[3] = f * 0.66904654355728921 * -18*y*z;\
            p[4] = f * 1.2694265629824517 * (-5*z*z+r2+2*y*y);\
            p[5] = f * 4.0142792613437353 * -x*z;\
            p[6] = f * 0.94617469575756008 * (5*y*y-7*z*z+r2-x*x);\
            p[7] = f * 10.620784618679583 * -x*z;\
            p[8] = f * 2.5033429417967046 * (3*y*y-3*x*x);\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_zz(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 1.2615662610100802;\
            p[3] = f * 0;\
            p[4] = f * 0;\
            break;\
          case 3:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 3.6563663957157262 * y;\
            p[3] = f * 1.1195289977703462 * 4*z;\
            p[4] = f * 3.6563663957157262 * x;\
            p[5] = f * 0;\
            p[6] = f * 0;\
            break;\
          case 4:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 11.354096349090721 * x*y;\
            p[3] = f * 2.0071396306718676 * 8*y*z;\
            p[4] = f * 1.6925687506432689 * (-3*r2+9*z*z);\
            p[5] = f * 2.0071396306718676 * 8*x*z;\
            p[6] = f * 5.6770481745453605 * (-y*y+x*x);\
            p[7] = f * 0;\
            p[8] = f * 0;\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_xy(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 1.0925484305920792;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            p[3] = f * 0;\
            p[4] = f * 0;\
            break;\
          case 3:\
            p[0] = f * 3.5402615395598613 * x;\
            p[1] = f * 2.8906114426405538 * z;\
            p[2] = f * 0.91409159892893155 * -x;\
            p[3] = f * 0;\
            p[4] = f * 0.45704579946446577 * -2*y;\
            p[5] = f * 0;\
            p[6] = f * 1.7701307697799307 * -2*y;\
            break;\
          case 4:\
            p[0] = f * 2.5033429417967046 * (-3*y*y+3*x*x);\
            p[1] = f * 10.620784618679583 * x*z;\
            p[2] = f * 0.94617469575756008 * (7*z*z-r2-2*y*y-2*x*x);\
            p[3] = f * 4.0142792613437353 * -x*z;\
            p[4] = f * 1.2694265629824517 * 2*x*y;\
            p[5] = f * 0.66904654355728921 * -2*y*z;\
            p[6] = f * 0.94617469575756008 * (-2*x*y+2*x*y);\
            p[7] = f * 5.3103923093397913 * -2*y*z;\
            p[8] = f * 2.5033429417967046 * -6*x*y;\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_xz(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            p[3] = f * 1.0925484305920792;\
            p[4] = f * 0;\
            break;\
          case 3:\
            p[0] = f * 0;\
            p[1] = f * 2.8906114426405538 * y;\
            p[2] = f * 0;\
            p[3] = f * 2.2390579955406924 * -x;\
            p[4] = f * 0.45704579946446577 * (-2*z+10*z);\
            p[5] = f * 2.8906114426405538 * x;\
            p[6] = f * 0;\
            break;\
          case 4:\
            p[0] = f * 0;\
            p[1] = f * 10.620784618679583 * x*y;\
            p[2] = f * 0.94617469575756008 * 12*y*z;\
            p[3] = f * 4.0142792613437353 * -x*y;\
            p[4] = f * 1.2694265629824517 * -8*x*z;\
            p[5] = f * 0.66904654355728921 * (-3*r2-6*z*z-6*x*x+21*z*z);\
            p[6] = f * 0.94617469575756008 * 12*x*z;\
            p[7] = f * 5.3103923093397913 * (-y*y+x*x);\
            p[8] = f * 0;\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

#define spherical_harmonics_derivative_yz(l, f, x, y, z, r2, p) (\
    {\
      switch(l)\
        {\
          case 0:\
            p[0] = f * 0;\
            break;\
          case 1:\
            p[0] = f * 0;\
            p[1] = f * 0;\
            p[2] = f * 0;\
            break;\
          case 2:\
            p[0] = f * 0;\
            p[1] = f * 1.0925484305920792;\
            p[2] = f * 0;\
            p[3] = f * 0;\
            p[4] = f * 0;\
            break;\
          case 3:\
            p[0] = f * 0;\
            p[1] = f * 2.8906114426405538 * x;\
            p[2] = f * 0.45704579946446577 * 12*z;\
            p[3] = f * 2.2390579955406924 * -y;\
            p[4] = f * 0;\
            p[5] = f * 2.8906114426405538 * -y;\
            p[6] = f * 0;\
            break;\
          case 4:\
            p[0] = f * 0;\
            p[1] = f * 5.3103923093397913 * (-y*y+x*x);\
            p[2] = f * 0.94617469575756008 * 12*x*z;\
            p[3] = f * 0.66904654355728921 * (-6*y*y-3*r2+15*z*z);\
            p[4] = f * 1.2694265629824517 * -8*y*z;\
            p[5] = f * 4.0142792613437353 * -x*y;\
            p[6] = f * 0.94617469575756008 * -12*y*z;\
            p[7] = f * 10.620784618679583 * -x*y;\
            p[8] = f * 0;\
            break;\
          default:\
            assert(0 == 1);\
        }\
    }\
)\

