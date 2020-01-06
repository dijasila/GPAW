import numpy as np


ni = 10
nx = 10
ny = 10
nz = 10

A = np.random.rand(ni, nx, ny, nz)

inds = np.zeros((nx, ny, nz), dtype=int)
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            inds[ix, iy, iz] = np.random.randint(ni)

xs = range(nx)
ys = range(ny)
zs = range(nz)

X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

for _ in range(100):
    cx = np.random.randint(ni)
    cy = np.random.randint(ni)
    cz = np.random.randint(ni)
    index = inds[cx, cy, cz]
    #expected = A[index, (0, 1, 5)]
    expected = A[index, cx, cy, cz]



    assert np.allclose(A[inds, X, Y, Z][cx, cy, cz], expected)
# assert np.allclose(A[inds, xs, ys, zs][choice], expected)

# assert np.allclose(A[inds, xs, ys, zs], A[inds, X, Y, Z])
            
