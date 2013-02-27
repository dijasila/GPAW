import sys
import re
import numpy as np
import datetime

def write_parallel_cube(basename, data, gd, atoms, rank):
    hx = gd.h_cv[0,0]
    hy = gd.h_cv[1,1]
    hz = gd.h_cv[2,2]

    f = open('%s_%06d.cube' % (basename, rank), 'w', 256*1024)

    f.write('GPAW Global (natom  nx ny nz  hx hy hz):  %6d  %6d %6d %6d  %12.6lf %12.6lf %12.6lf\n' % (len(atoms), gd.N_c[0], gd.N_c[1], gd.N_c[2], hx,hy,hz))
    f.write('GPAW Local (xbeg xend  ybeg yend  zbeg zend):  %6d %6d  %6d %6d  %6d %6d\n' % ( gd.beg_c[0], gd.end_c[0],  gd.beg_c[1], gd.end_c[1],  gd.beg_c[2], gd.end_c[2]))
    
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (len(atoms), hx*gd.beg_c[0], hy*gd.beg_c[1], hz*gd.beg_c[2]))
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (gd.n_c[0], hx, 0.0, 0.0))
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (gd.n_c[1], 0.0, hy, 0.0))
    f.write('%6d %12.6lf %12.6lf %12.6lf\n' % (gd.n_c[2], 0.0, 0.0, hz))
    
    for (i,atom) in enumerate(atoms):
        f.write('%6d %12.6lf %12.6lf %12.6lf %12.6lf\n' % (atom.number, 0.0, atom.position[0]/0.529177, atom.position[1]/0.529177, atom.position[2]/0.529177))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                f.write('%18.8lf\n' % data[i,j,k])
    f.close()


def cubify(out_filename, filenames):
    print >> sys.stderr, 'Reading partial cube files...', datetime.datetime.now()
    data = None
    for fname in filenames:
        f = open(fname,'r', 256*1024)
        line0 = f.readline()
        elems = re.compile('[\d.e+-]+').findall(line0)
        natom = int(elems[0])
        (nx,ny,nz) = map(int, elems[1:4])
        (hx,hy,hz) = map(float, elems[4:7])
        line = f.readline()
        (xb,xe,yb,ye,zb,ze) = map(int,re.compile('\d+').findall(line))
        print >> sys.stderr, xb,xe,yb,ye,zb,ze 

        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()

        atom_lines = []
        for n in range(natom):
            atom_lines.append(f.readline())

        if data is None:
            data = np.zeros((nx,ny,nz))
    
        for i in range(xb,xe):
            for j in range(yb,ye):
                for k in range(zb,ze):
                    data[i,j,k] = float(f.readline())
    
        f.close()

    print >> sys.stderr, 'Reading done. ', datetime.datetime.now()
    print >> sys.stderr, 'Writing the cube file... %d %d %d' % (nx,ny,nz)

    out = open(out_filename, 'w', 256*1024)
    out.write(line0)
    out.write('OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n')
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (natom, 0.0, 0.0, 0.0))
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (nx, hx, 0.0, 0.0))
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (ny, 0.0, hy, 0.0))
    out.write('%8d %12.6lf %12.6lf %12.6lf\n' % (nz, 0.0, 0.0, hz))
    for n in range(natom):
        out.write(atom_lines[n])
    for i in range(nx):
        print >> sys.stderr, '.',
        for j in range(ny):
            for k in range(nz):
                out.write('%18.8lf\n' % data[i,j,k])
    out.close()
    print >> sys.stderr, ''
    print >> sys.stderr, 'Writing done. ', datetime.datetime.now()

if __name__ == '__main__':
    cubify(sys.argv[1], sys.argv[2:])
