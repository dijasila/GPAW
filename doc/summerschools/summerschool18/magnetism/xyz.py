# creates: VI2.xyz, CrI3.xyz
from pathlib import Path
Path('VI2.xyz').write_text("""\
3
Lattice="4.12605070262 0.0 0.0 -2.06302535131 3.57326472577 0.0 0.0 0.0 18.1312754335" Properties=species:S:1:pos:R:3:initial_magmoms:R:1:Z:I:1:forces:R:3:magmoms:R:1:dipole:R:1 energy=-12.9519900734255 free_energy=-12.95199117836 pbc="T T F" magmom=2.99999999326358 stress="9.71555832107e-05 1.65911424597e-18 0.0 1.65911424597e-18 9.71555832107e-05 7.777098028e-20 0.0 7.777098028e-20 -7.17524346066e-05"
V       0.00000000      -0.00000000       9.06563772       1.00000000       23       0.00000000       0.00000000       0.00000003       2.63358805      -0.00000000
I       2.06302535       1.19108824      10.63303664       0.10000000       53       0.00000000       0.00000000       0.00792029      -0.03990743       0.00000000
I      -0.00000000       2.38217648       7.49823879       0.10000000       53      -0.00000000       0.00000000      -0.00792046      -0.03990743      -0.00000000
""")
Path('CrI3.xyz').write_text("""\
8
Lattice="7.00794410138 5.69308307302e-19 0.0 -3.50397205069 6.0690576201 0.0 -7.55612427101e-19 0.0 18.0635365764" Properties=species:S:1:pos:R:3:initial_magmoms:R:1:Z:I:1:forces:R:3:magmoms:R:1 energy=-31.3985425600531 dipole="-0.00239032749259 0.00138005621976 3.42821467339e-07" free_energy=-31.3985446012773 pbc="T T F" magmom=6.00000000876689 stress="-6.01056204961e-05 0.0 -6.48091502334e-21 0.0 -6.01056204961e-05 -2.59236600933e-20 -6.48091502334e-21 -2.59236600933e-20 6.05488486747e-05"
Cr       3.50397205       2.02301921       9.03176873       1.00000000       24       0.00000000       0.00000000       0.00001571       2.98036758
Cr       0.00000000       4.04603841       9.03176919       1.00000000       24      -0.00000000       0.00000000      -0.00003090       2.98034166
I       4.48565621      -0.00001343       7.46488009       1.00000000       53       0.00468920       0.00013810      -0.00213546      -0.05825951
I       1.26113232       2.18437210       7.46488009       1.00000000       53      -0.00222500      -0.00413001      -0.00213546      -0.05825951
I      -2.24281648       3.88469895       7.46488009       1.00000000       53      -0.00246420       0.00399192      -0.00213546      -0.05825951
I       2.52228800      -0.00001162      10.59865633       1.00000000       53      -0.00469075       0.00014771       0.00214167      -0.05825936
I       2.24281799       3.88469794      10.59865633       1.00000000       53       0.00247329       0.00398846       0.00214167      -0.05825936
I      -1.26113394       2.18437130      10.59865633       1.00000000       53       0.00221746      -0.00413616       0.00214167      -0.05825936
""")
