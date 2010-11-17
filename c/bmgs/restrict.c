/*  Copyright (C) 2003-2007  CAMP
 *  Please see the accompanying LICENSE file for further information. */

#include "bmgs.h"
#include "../extensions.h"

#ifdef K

void
RST1D(const T *a, const int n, const int m, T* restrict b)
{
    a += K - 1;

    for (int j = 0; j < m; j++) {
        const T* aa = a + j * (n * 2 + K * 2 - 3);
        T* restrict bb = b + j;

        for (int i = 0; i < n; i++) {
            if (K == 2)
                bb[0] = 0.5 * (aa[0] + 0.5 * (aa[1] + aa[-1]));
            else if (K == 4)
                bb[0] = 0.5 * (aa[0] +
                        0.5625 * (aa[1] + aa[-1]) +
                       -0.0625 * (aa[3] + aa[-3]));
            else if (K == 6)
                bb[0] = 0.5 * (aa[0] +
                        0.58593750 * (aa[1] + aa[-1]) +
                       -0.09765625 * (aa[3] + aa[-3]) +
                        0.01171875 * (aa[5] + aa[-5]));
            else  /* K == 8 */
                bb[0] = 0.5 * (aa[0] +
                        0.59814453125 * (aa[1] + aa[-1]) +
                       -0.11962890625 * (aa[3] + aa[-3]) +
                        0.02392578125 * (aa[5] + aa[-5]) +
                       -0.00244140625 * (aa[7] + aa[-7]));
            aa += 2;
            bb += m;
        }
    }
}

#else
#  define K 2
#  define RST1D Z(bmgs_restrict1D2)
#  include "restrict.c"
#  undef RST1D
#  undef K
#  define K 4
#  define RST1D Z(bmgs_restrict1D4)
#  include "restrict.c"
#  undef RST1D
#  undef K
#  define K 6
#  define RST1D Z(bmgs_restrict1D6)
#  include "restrict.c"
#  undef RST1D
#  undef K
#  define K 8
#  define RST1D Z(bmgs_restrict1D8)
#  include "restrict.c"
#  undef RST1D
#  undef K

void
Z(bmgs_restrict)(int k, T* a, const int n[3], T* restrict b, T* restrict w)
{
    void (*plg)(const T*, int, int, T*);
    int e;

    if (k == 2)
        plg = Z(bmgs_restrict1D2);
    else if (k == 4)
        plg = Z(bmgs_restrict1D4);
    else if (k == 6)
        plg = Z(bmgs_restrict1D6);
    else  /* Presumably k == 8 ... */
        plg = Z(bmgs_restrict1D8);

    e = k * 2 - 3;
    plg(a, (n[2] - e) / 2, n[0] * n[1], w);
    plg(w, (n[1] - e) / 2, n[0] * (n[2] - e) / 2, a);
    plg(a, (n[0] - e) / 2, (n[1] - e) * (n[2] - e) / 4, b);
}

#endif
