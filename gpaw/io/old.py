"""Code for reading old gpw files."""
import numpy as np
import ase.io.aff as aff
from ase import Atoms
from ase.io.trajectory import write_atoms
from ase.units import AUT, Bohr, Ha

from gpaw.io.tar import Reader


def wrap_old_gpw_reader(filename):
    r = Reader(filename)
    assert not r.byteswap
    
    data = {'version': -1,
            'ha': Ha,
            'bohr': Bohr,
            'scf.': {'converged': True},
            'atoms.': {},
            'wave_functions.': {}}
    
    class DictBackend:
        def write(self, **kwargs):
            data['atoms.'].update(kwargs)
            
    write_atoms(DictBackend(), read_atoms(r))
    
    e_total_extrapolated = r.get('PotentialEnergy').item() * Ha
    data['results.'] = {
        'energy': e_total_extrapolated}
    
    data['parameters.'] = {}
    
    data['occupations.'] = {
        'fermilevel': r['FermiLevel'] * Ha,
        'split': r.parameters.get('FermiLevel', 0) * Ha,
        'homo': np.nan,
        'lumo': np.nan}
    
    data['density.'] = {
        'density': r.get('PseudoElectronDensity') * Bohr**-3,
        'atomic_density_matrices': r.get('AtomicDensityMatrices')}
    
    data['hamiltonian.'] = {
        'e_coulomb': r['Epot'] * Ha,
        'e_entropy': -r['S'] * Ha,
        'e_external': r['Eext'] * Ha,
        'e_kinetic': r['Ekin'] * Ha,
        'e_total_extrapolated': e_total_extrapolated,
        'e_xc': r['Exc'] * Ha,
        'e_zero': r['Ebar'] * Ha,
        'potential': r.get('PseudoPotential') * Ha,
        'atomic_hamiltonian_matrices': r.get('NonLocalPartOfHamiltonian') * Ha}
    data['hamiltonian.']['e_total_free'] = (
        sum(data['hamiltonian.'][e] for e in ['e_coulomb', 'e_entropy',
                                              'e_external', 'e_kinetic',
                                              'e_xc', 'e_zero']))
    
    for name, old in [('values', 'PseudoWaveFunctions'),
                      ('eigenvalues', 'Eigenvalues'),
                      ('occupations', 'OccupationNumbers'),
                      ('projections', 'Projections')]:
        fd, shape, size, dtype = r.get_file_object(old, ())
        offset = fd
        data['wave_functions.'][name + '.'] = {
            'ndarray': (shape, dtype.name, offset)}

    new = aff.Reader(None, data=data)
    
    for ref in new._data['wave_functions']._data.values():
        ref.fd = ref.offset
        ref.offset = 0
        
    class File:
        def close(self):
            pass
            
    new._fd = File()
    
    return new
    
    
def read_atoms(reader):
    positions = reader.get('CartesianPositions', broadcast=True) * Bohr
    numbers = reader.get('AtomicNumbers', broadcast=True)
    cell = reader.get('UnitCell', broadcast=True) * Bohr
    pbc = reader.get('BoundaryConditions', broadcast=True)
    tags = reader.get('Tags', broadcast=True)
    magmoms = reader.get('MagneticMoments', broadcast=True)

    # Create instance of Atoms object, and set_tags and magnetic moments
    atoms = Atoms(positions=positions,
                  numbers=numbers,
                  cell=cell,
                  pbc=pbc)

    if tags.any():
        atoms.set_tags(tags)
    if magmoms.any():
        atoms.set_initial_magnetic_moments(magmoms)

    if reader.has_array('CartesianVelocities'):
        velocities = reader.get('CartesianVelocities',
                                broadcast=True) * Bohr / AUT
        atoms.set_velocities(velocities)

    return atoms
