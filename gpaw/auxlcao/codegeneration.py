
from sympy import symbols, sqrt, diff, sqrt, pi, Matrix, diag, erfc, cse, pprint, numbered_symbols
from sympy.codegen.pyutils import PythonCodePrinter

printer = PythonCodePrinter({'standard':'python3'})
def pyprint(content):
    return printer.doprint(content)

with open("generatedcode.py", "w") as output:
    pass


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
omega = symbols('omega')

maxQ = 4
maxL = 4

map_LQ = map_LQ[:maxL,:maxQ]
M_Qc = M_Qc[:maxQ]

for screening in [True, False]:
    if screening:
        V = erfc( omega * sqrt( (x1+dx-x2)**2 + (y1+dy-y2)**2 + (z1+dz-z2)**2 ) ) / sqrt( (x1+dx-x2)**2 + (y1+dy-y2)**2 + (z1+dz-z2)**2 )
    else:
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
            print((Q1,Q2))
            expr2 = diff(expr1, x2, M2_c[0])
            print('1')
            expr2 = expr2.subs(x2, 0)
            print('2')
            expr2 = diff(expr2, y2, M2_c[1])
            print('3')
            expr2 = expr2.subs(y2, 0)
            print('4')
            expr2 = diff(expr2, z2, M2_c[2])
            print('5')
            expr2 = expr2.subs(z2, 0)
            print('6')
            expr2 = expr2.subs(sqrt(dx**2 + dy**2 + dz**2), d)
            print('7')
            #expr2 = expr2.subs(d**9, d9)
            print('s')
            expr2 = expr2.simplify()
            print('s end')
            row.append(expr2)
            print(expr2)
        rows.append(row)
    M_QQ = Matrix(rows) 
    M_LL = map_LQ @ M_QQ @ map_LQ.T
    CSE = cse(M_LL, numbered_symbols('x_'))
    with open("generatedcode.py", "a") as output:
        for helper in CSE[0]:
            write('    ' + pyprint(helper[0]) + '=' + pyprint(helper[1]))
            write("\n")

        for i,result in enumerate(CSE[1]):
            assert i == 0
            for I in range(maxL):
                for J in range(maxL):
                    output.write('    W_LL[%d::%d, %d::%d] = %s\n' % (I, maxL, J, maxL, pyprint(result[I,J])))
            output.write("\n")
   


