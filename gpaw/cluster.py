import math
import re
import Numeric as num

from ASE import Atom, ListOfAtoms
from ASE.Utilities.GeometricTransforms import Translate as GTTranslate
from ASE.Utilities.GeometricTransforms import RotateAboutAxis
from ASE.IO.xyz import ReadXYZ, WriteXYZ
from ASE.IO.PDB import WritePDB

class Cluster(ListOfAtoms):
    """A class for cluster structures
    to enable simplified manipulation"""

    def __init__(self, atoms=None, cell=None,
                 filename=None, filetype=None,
                 timestep=0.0):
        if atoms is None:
            ListOfAtoms.__init__(self,[],cell=cell)
        else:
            ListOfAtoms.__init__(self,atoms, cell=cell,periodic=False)

        if filename is not None:
            self.Read(filename,filetype)

        self.Timestep(timestep)
        
    def Center(self):
        """Center the structure to unit cell"""
        extr = self.extreme_positions()
        cntr = 0.5 * (extr[0] + extr[1])
        cell = num.diagonal(self.GetUnitCell())
        GTTranslate(self,tuple(.5*cell-cntr),'cartesian')

    def CenterOfMass(self):
        """Return the structures center of mass"""
        cm = num.zeros((3,),num.Float)
        M = 0.
        for atom in self:
            m = atom.GetMass()
            M += m
            cm += m * atom.GetCartesianPosition()
        return cm/M
    
    def extreme_positions(self):
        """get the extreme positions of the structure"""
        pos = self.GetCartesianPositions()
        return num.array([num.minimum.reduce(pos),num.maximum.reduce(pos)])

    def MinimalBox(self,border=0):
        """The box needed to fit the structure in.
        The structure is moved to fit into the box [(0,x),(0,y),(0,z)]
        with x,y,z > 0 (fitting the ASE constriction).
        The border argument can be used to add a border of empty space
        around the structure.
        """

        if len(self) == 0:
            return None

        extr = self.extreme_positions()
 
        # add borders
        if type(border)==type([]):
            b=border
        else:
            b=[border,border,border]
        for i in range(3):
            extr[0][i]-=b[i]
            extr[1][i]+=b[i]-extr[0][i] # shifted already
            
        # move lower corner to (0,0,0)
        GTTranslate(self,tuple(-1.*num.array(extr[0])),'cartesian')
        self.SetUnitCell(tuple(extr[1]),fix=True)

        return self.GetUnitCell()

    def Read(self,filename,filetype=None):
        """Read the strcuture from some file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'xyz':
            loa = ReadXYZ(filename)
            self.__init__(loa)
        
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
        return len(self)

    def Rotate(self,axis,angle=None,unit='rad'):
        if angle is None:
            angle = axis.length()
        axis.length(1.)
        if unit == 'rad':
            angle *= 180. / math.pi
        RotateAboutAxis(self,axis,angle)

    def Timestep(self,timestep=None):
        """Set and/or get the timestep label of this structure"""
        if timestep is not None:
            self.timestep = float(timestep)
        return self.timestep

    def Translate(self,trans_vector):
        """Translate the whole structure"""
        GTTranslate(self,tuple(trans_vector),'cartesian')

    def Write(self,filename,filetype=None):
        """Write the strcuture to file. The type can be given
        or it will be guessed from the filename."""

        if filetype is None:
            # estimate file type from name ending
            filetype = filename.split('.')[-1]
        filetype.lower()

        if filetype == 'xyz':
            uc = self.GetUnitCell()
            if uc:
                id=' unit cell'
                for v in uc:
                    id+=' (%g,%g,%g)' % (v[0],v[1],v[2])
            WriteXYZ(filename,self,id=id)
        elif filetype == 'pdb':
            WritePDB(filename,self)
        else:
            raise NotImplementedError('unknown file type "'+filetype+'"')
                
       
