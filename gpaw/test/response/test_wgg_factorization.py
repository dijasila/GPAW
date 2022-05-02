def factorize(N):
    for n in range(1, N + 1):
        if N % n == 0:
            yield N // n, n


def get_products(N):
    for a1, a2 in factorize(N):
        for a2p, a3 in factorize(a2):
            yield a1, a2p, a3


def choose_parallelization(nW, nG, commsize):
    min_badness = 10000000

    for wGG in get_products(commsize):
        wsize, gsize1, gsize2 = wGG
        nw = (nW + wsize - 1) // wsize

        if nw > nW:
            continue

        number_of_cores_with_zeros = (wsize * nw - nW) // nw
        scalapack_skew = (gsize1 - gsize2)**2
        scalapack_size = gsize1 * gsize2
        badness = (number_of_cores_with_zeros * 1000
                   + 10 * scalapack_skew + scalapack_size)

        # print(wsize, gsize1, gsize2, nw, number_of_cores_with_zeros, badness)
        if badness < min_badness:
            wGG_min = wGG
            min_badness = badness
    return wGG_min


def test_parallelizations():
    assert choose_parallelization(131, 1455, 160) == (10, 4, 4)
    assert choose_parallelization(470, 10000, 1) == (1, 1, 1)
    assert choose_parallelization(1, 10000, 256) == (1, 16, 16)
    assert choose_parallelization(470, 10000, 256) == (16, 4, 4)
