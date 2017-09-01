import os


def archive(gpwfile, folder=None, key_value_pairs={}):
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
    ulm.copy(gpwfile, os.path.join(folder, str(id) + '.gpw'),
             exclude={'.wave_functions.values'})
    return id, folder
