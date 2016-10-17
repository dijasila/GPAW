scalapack = True
libraries = ['gfortran',
             'scalapack',
             'openblas',
             'readline',
             'xc']
extra_compile_args = ['-O3', '-std=c99', '-fPIC', '-Wall']
define_macros += [('GPAW_NO_UNDERSCORE_CBLACS', '1'),
                  ('GPAW_NO_UNDERSCORE_CSCALAPACK', '1')]
platform_id = 'el7'
