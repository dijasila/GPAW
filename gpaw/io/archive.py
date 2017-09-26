import os

from gpaw.mpi import world


def archive(gpwfile, folder=None, key_value_pairs={}):
    """Add gpw-file to archive and add row to database.

    Copy gpw-file to ARC/<id>.gpw and add row to database
    in ARC/gpw.db@id=<id> where <id> is a uniqe integer id.  The folder ARC
    defaults to ~/.gpaw/archive/ or $GPAW_ARCHIVE_FOLDER.

    Note that the wave-functions will *not* be written to the archive.

    The keys and values in key_value_pairs will be added to the row written
    to the database file.
    """

    import ase.io.ulm as ulm
    from ase.db import connect
    import gpaw
    from gpaw import GPAW
    # os.environ['USER'] = user
    assert 'filepath' not in key_value_pairs
    assert 'filename' not in key_value_pairs
    if folder is None:
        folder = os.environ.get('GPAW_ARCHIVE_FOLDER')
        if folder is None:
            gpaw.read_rc_file()
            folder = gpaw.archive_folder
            if folder is None:
                folder = os.path.join(os.environ['HOME'], '.gpaw', 'archive')
                os.makedirs(folder, exist_ok=True)

    calc = GPAW(gpwfile, txt=None)
    db = connect(os.path.join(folder, 'gpw.db'))
    id = db.write(calc.get_atoms(),
                  filepath=os.path.abspath(gpwfile),
                  filename=os.path.basename(gpwfile),
                  key_value_pairs=key_value_pairs)
    if world.rank == 0:
        ulm.copy(gpwfile, os.path.join(folder, str(id) + '.gpw'),
                 exclude={'.wave_functions.values'})
    return id, folder
