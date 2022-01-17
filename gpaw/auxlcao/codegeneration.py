
"""


 M_L =   n(r) Y_Ln(r)    Y_Lc(r) r^l

"""

def generate():
    from sympy import symbols, sqrt, diff, sqrt, pi, Matrix, diag, erfc, cse, pprint, numbered_symbols
    from sympy.codegen.pyutils import PythonCodePrinter
    from math import factorial, prod
    printer = PythonCodePrinter({'standard':'python3'})
    def pyprint(content):
        return printer.doprint(content).replace('math.erfc','scipy.special.erfc')
    
    with open("generatedcode.py", "w") as output:
        output.write('import numpy as math # :)\n')
        output.write('import scipy\n')
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
              [ 0, 1, 1 ],   # yz
              [ 3, 0, 0 ],   # x^3
              [ 0, 3, 0 ],   # y^3
              [ 0, 0, 3 ],   # z^3
              [ 2, 1, 0 ],   # x^2 y
              [ 2, 0, 1 ],   # x^2
              [ 1, 2, 0 ],   # x y^2
              [ 0, 2, 1 ],   # y^2 z
              [ 1, 0, 2 ],   # x z^2
              [ 0, 1, 2 ],   # y z^2
              [ 1, 1, 1 ] ]  # xyz
    
    N_L = [ 1,            # 1
            1/sqrt(3),    # y
            1/sqrt(3),    # z
            1/sqrt(3),    # x
            1/sqrt(15),   # xy
            1/sqrt(15),   # yz
            1/sqrt(3*15), # 2z2-z2-y2
            1/sqrt(15),   # zx
xx1	            1/sqrt(15)  ] # z2-y2

    [(0.4886025119029199, (0, 1, 0))],
    [(0.4886025119029199, (0, 0, 1))],
    [(0.4886025119029199, (1, 0, 0))],


    [(1.0925484305920792, (1, 1, 0))],
    [(1.0925484305920792, (0, 1, 1))],
    [(-0.31539156525252, (2, 0, 0)),
     (-0.31539156525252, (0, 2, 0)),
     (0.63078313050504, (0, 0, 2))],
    [(1.0925484305920792, (1, 0, 1))],
    [(-0.5462742152960396, (0, 2, 0)),
     (0.5462742152960396, (2, 0, 0))],

    
    
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
    
    maxQ = 1  + 3 + 6
    maxL = 1  + 3 + 5
    
    map_LQ = map_LQ[:maxL,:maxQ]
    M_Qc = M_Qc[:maxQ]
    
    for screening in [False, True]:
        if screening:
            V = erfc( omega * sqrt( (x1+dx-x2)**2 + (y1+dy-y2)**2 + (z1+dz-z2)**2 ) ) / sqrt( (x1+dx-x2)**2 + (y1+dy-y2)**2 + (z1+dz-z2)**2 )
        else:
            V = 1 / sqrt( (x1+dx-x2)**2 + (y1+dy-y2)**2 + (z1+dz-z2)**2 )
    
        rows = []
        for Q1, M1_c in enumerate(M_Qc):
            print(Q1)
            expr1 = diff(V, x1, M1_c[0])
            print(end='.')
            expr1 = expr1.subs(x1, 0)
            print(end='.')
            expr1 = diff(expr1, y1, M1_c[1])
            print(end='.')
            expr1 = expr1.subs(y1, 0)
            print(end='.')
            expr1 = diff(expr1, z1, M1_c[2])
            print(end='.')
            expr1 = expr1.subs(z1, 0)
            
            expr1 /= prod([ factorial(M_c) for M_c in M1_c ])
            print(end='.')
            row = []
            for Q2, M2_c in enumerate(M_Qc):
                expr2 = diff(expr1, x2, M2_c[0])
                print(end='.')
                expr2 = expr2.subs(x2, 0)
                print(end='.')
                expr2 = diff(expr2, y2, M2_c[1])
                print(end='.')
                expr2 = expr2.subs(y2, 0)
                print(end='.')
                expr2 = diff(expr2, z2, M2_c[2])
                print(end='.')
                expr2 = expr2.subs(z2, 0)
                print(end='.')
                expr2 = expr2.subs(sqrt(dx**2 + dy**2 + dz**2), d)
                print(end='.')
                #expr2 = expr2.subs(d**9, d9)
                print(end='.')
                expr2 = expr2.simplify()
                expr2 /= prod([ factorial(M_c) for M_c in M2_c ])
                row.append(expr2)
                print()
                print(expr2)
    
            rows.append(row)
        M_QQ = Matrix(rows) 
        M_LL = map_LQ @ M_QQ @ map_LQ.T
        CSE = cse(M_LL, numbered_symbols('x_'))
        #CSE = ( (), M_LL )
        with open("generatedcode.py", "a") as output:
            if screening:
                output.write('def generated_W_LL_screening(W_LL, d, dx, dy, dz, omega):\n')
            else:
                output.write('def generated_W_LL(W_LL, d, dx, dy, dz):\n')
            for helper in CSE[0]:
                output.write('    ' + pyprint(helper[0]) + '=' + pyprint(helper[1]))
                output.write("\n")
    
            for i,result in enumerate(CSE[1]):
                assert i == 0
                for I in range(maxL):
                    for J in range(maxL):
                        output.write('    W_LL[%d::%d, %d::%d] += %s\n' % (I, maxL, J, maxL, pyprint(result[I,J])))
                output.write("\n")
       
    
    
