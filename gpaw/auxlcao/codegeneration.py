from sympy import symbols, sqrt, diff, sqrt, pi, Matrix, diag

# Q = simple monimial multipole index
# L = spherical harmonic index
# Mapping below

M_Qc =  [ [ 0, 0, 0 ],   # Monopole
          [ 1, 0, 0 ],   # x dipole
          [ 0, 1, 0 ],   # y dipole
          [ 0, 0, 1 ],   # z dipole
          [ 2, 0, 0 ],   # x^2 quadrupole
          [ 0, 2, 0 ],   # y^2 quadrupole
          [ 0, 0, 2 ],   # z^2 quadrupole (x^2+y^2+z^2 terms should vanish) 6 terms span S+5D.
          [ 1, 1, 0 ],   # xy
          [ 1, 0, 1 ],   # xz
          [ 0, 1, 1 ] ] # yz


N_L = [ 1,            # 1
        1/sqrt(3),    # y
        1/sqrt(3),    # z
        1/sqrt(3),    # x
        1/sqrt(15),   # xy
        1/sqrt(15),   # yz
        1/sqrt(3*15), # 2z2-z2-y2
        1/sqrt(15),   # zx
        1/sqrt(15)  ] # z2-y2


#                                            Q-index
#                   1     x     y     z     x^2   y^2   z^2   xy    xz    yz

map_LQ =  Matrix([[ 1,    0,    0,    0,    0,    0,    0,    0,    0,    0  ],   # 1
                  [ 0,    0,    1,    0,    0,    0,    0,    0,    0,    0  ],   # y
                  [ 0,    0,    0,    1,    0,    0,    0,    0,    0,    0  ],   # z
                  [ 0,    1,    0,    0,    0,    0,    0,    0,    0,    0  ],   # x
                  [ 0,    0,    0,    0,    0,    0,    0,    1,    0,    0  ],   # xy   GPAW's L-index
                  [ 0,    0,    0,    0,    0,    0,    0,    0,    0,    1  ],   # yz
                  [ 0,    0,    0,    0,   -1,   -1,    2,    0,    0,    0  ],   # 2z2-x2-y2
                  [ 0,    0,    0,    0,    0,    0,    0,    0,    1,    0  ],   # zx
                  [ 0,    0,    0,    0,    1,   -1,    0,    0,    0,    0  ]])  # x2-y2

map_LQ = diag(*N_L) * map_LQ

dx, dy, dz = symbols('dx dy dz')
x1, y1, z1 = symbols('x1 y1 z1')
x2, y2, z2 = symbols('x2 y2 z2')
d, d2, d9 = symbols('d d2 d9')

V = 1 / sqrt( (x1+dx-x2)**2 + (y1+dy-y2)**2 + (z1+dz-z2)**2 )

rows = []
for Q1, M1_c in enumerate(M_Qc):
    print(Q1)
    expr1 = diff(V, x1, M1_c[0])
    expr1 = diff(expr1, x1, M1_c[0])
    expr1 = expr1.subs(x1, 0)
    expr1 = diff(expr1, y1, M1_c[1])
    expr1 = expr1.subs(y1, 0)
    expr1 = diff(expr1, z1, M1_c[2])
    expr1 = expr1.subs(z1, 0)
    row = []
    for Q2, M2_c in enumerate(M_Qc):
        expr2 = diff(expr1, x2, M2_c[0])
        expr2 = expr2.subs(x2, 0)
        expr2 = diff(expr2, y2, M2_c[1])
        expr2 = expr2.subs(y2, 0)
        expr2 = diff(expr2, z2, M2_c[2])
        expr2 = expr2.subs(z2, 0).simplify()
        expr2 = expr2.subs(sqrt(dx**2 + dy**2 + dz**2), d)
        #expr2 = expr2.subs(d**9, d9)
        expr2 = expr2.simplify()
        row.append(expr2)
    rows.append(row)
M_QQ = Matrix(rows) 
M_LL = map_LQ @ M_QQ @ map_LQ.T

print(M_LL)

