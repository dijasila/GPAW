from __future__ import print_function
import os
import fnmatch
import urllib2
from StringIO import StringIO
import tarfile
from optparse import OptionParser, OptionGroup
import re


usage = '%prog [OPTION]\n  or:  %prog [OPTION] INSTALLDIR'
description = ('In first form, show available setups and GPAW setup paths.  '
               'In second form, download and install gpaw-setups into '
               'INSTALLDIR/[setups-package-name-and-version].')


sources = [('gpaw', 'official GPAW setups releases [default]'),
           ('sg15', 'SG15 pseudopotentials'),
           ('basis', 'basis sets for LCAO mode'),
           ('test', 'small file for testing this script'),
           ]
names = [r for r, d in sources]


def main(args):
    p = OptionParser(usage=usage, description=description)
    add = p.add_option
    add('--version', metavar='VERSION',
        help='download VERSION of package.  '
        'Run without arguments to display a list of versions.  '
        'VERSION can be the full URL or a part such as  '
        '\'0.8\' or \'0.6.6300\'')
    add('--tarball', metavar='FILE',
        help='unpack and install from local tarball FILE '
        'instead of downloading')
    add('--list-all', action='store_true',
        help='list packages from all sources')
    g = OptionGroup(p, 'Sources')
    for name, help in sources:
        g.add_option('--%s' % name, action='store_const',
                     const=name, dest='source',
                     help=help)
    p.add_option_group(g)
    add('--register', action='store_true',
        help='run non-interactively and register install path in '
        'GPAW setup search paths.  This is done by adding lines to '
        '~/.gpaw/rc.py')
    add('--no-register', action='store_true',
        help='run non-interactively and do not register install path in '
        'GPAW setup search paths')
    opts, args = p.parse_args(args)
    nargs = len(args)
    
    if opts.source is None:
        opts.source = sources[0][0]
    
    if opts.register and opts.no_register:
        p.error('Conflicting options specified on whether to register '
                'setup install paths in configuration file.  Try not '
                'specifying some options.')

    # The sg15 file is a tarbomb.  We will later defuse it by untarring
    # into a subdirectory, so we don't leave a ghastly mess on the
    # unsuspecting user's system.
    
    if not opts.tarball:
        if opts.list_all:
            urls = []
            for source in names:
                urls1 = get_urls(source)
                urls.extend(urls1)
        else:
            urls = get_urls(opts.source)
    
        def print_urls(urls, marked=None):
            for url in urls:
                pageurl, fname = url.rsplit('/', 1)
                if url == marked:
                    marking = ' [*]'
                else:
                    marking = '    '
                print(' %s %s' % (marking, url))
    
        if len(urls) == 0:
            p.error(
                'For some reason, no files were found. Probably this script '
                'is out of date.  Please rummage around GPAW web page until '
                'solution is found.  Writing e-mails to '
                'gpaw-developers@lists@listserv.fysik.dtu.dk is also likely'
                'to help.')
    
        if opts.version:
            matching_urls = [url for url in urls if opts.version in url]
            if len(matching_urls) > 1:
                p.error('More than one setup file matches version "%s":\n'
                        '%s' % (opts.version, '\n'.join(matching_urls)))
            elif len(matching_urls) == 0:
                p.error('\nNo setup matched the specified version "%s".\n'
                        'Available setups are:\n'
                        '%s' % (opts.version, '\n'.join(urls)))
            assert len(matching_urls) == 1
            url = matching_urls[0]
        else:
            url = urls[0]
    
        print('Available setups and pseudopotentials')
        print_urls(urls, url)
        print()
        
    if nargs == 0:
        print_setups_info(p)
        print()
        progname = p.get_prog_name()
        print('Run %s DIR to install newest setups into DIR.' % progname)
        print('Run %s DIR --version=VERSION to install VERSION (from above).'
              % progname)
        print('See %s --help for more info.' % progname)
        raise SystemExit
    elif len(args) != 1:
        p.error('No more than one DIR expected.  Please try --help.')
    
    targetpath = args[0]
    
    if opts.tarball:
        print('Reading local tarball %s' % opts.tarball)
        targzfile = tarfile.open(opts.tarball)
        tarfname = opts.tarball
    else:
        tarfname = url.rsplit('/', 1)[1]
        print('Selected %s.  Downloading...' % tarfname)
        response = urllib2.urlopen(url)
        targzfile = tarfile.open(fileobj=StringIO(response.read()))
    
    if not os.path.exists(targetpath):
        os.makedirs(targetpath)
    
    assert tarfname.endswith('.tar.gz')
    setup_dirname = tarfname.rsplit('.', 2)[0]  # remove .tar.gz ending
    setup_path = os.path.abspath(os.path.join(targetpath, setup_dirname))
    if tarfname.startswith('sg15'):
        # Defuse tarbomb
        if not os.path.isdir(setup_path):
            os.mkdir(setup_path)
        targetpath = os.path.join(targetpath, setup_dirname)
    
    print('Extracting tarball into %s' % targetpath)
    targzfile.extractall(targetpath)
    assert os.path.isdir(setup_path)
    print('Setups installed into %s.' % setup_path)
    
    # Okay, now we have to maybe edit people's rc files.
    rcfiledir = os.path.join(os.environ['HOME'], '.gpaw')
    rcfilepath = os.path.join(rcfiledir, 'rc.py')
    
    # We could do all this by importing the rcfile as well and checking
    # whether things are okay or not.
    rcline = "setup_paths.insert(0, '%s')" % setup_path
    
    # Run interactive mode unless someone specified a flag requiring otherwise
    interactive_mode = not (opts.register or opts.no_register)
    
    register_path = False
    
    if interactive_mode:
        answer = raw_input('Register this setup path in %s? [y/n] ' %
                           rcfilepath)
        if answer.lower() in ['y', 'yes']:
            register_path = True
        elif answer.lower() in ['n', 'no']:
            print('As you wish.')
        else:
            print('What do you mean by "%s"?  Assuming "n".' % answer)
    else:
        if opts.register:
            assert not opts.no_register
            register_path = True
        else:
            assert opts.no_register
    
    if register_path:
        # First we create the file
        if not os.path.exists(rcfiledir):
            os.makedirs(rcfiledir)
        if not os.path.exists(rcfilepath):
            tmpfd = open(rcfilepath, 'w')  # Just create empty file
            tmpfd.close()
    
        for line in open(rcfilepath):
            if line.startswith(rcline):
                print('It looks like the path is already registered in %s.'
                      % rcfilepath)
                print('File will not be modified at this time.')
                break
        else:
            rcfd = open(rcfilepath, 'a')
            print(rcline, file=rcfd)
            print('Setup path registered in %s.' % rcfilepath)
            # Need to explicitly flush/close the file so print_setups_info
            # sees the change in rc.py
            rcfd.close()
    
            print_setups_info(p)
    else:
        print('You can manually register the setups by adding the')
        print('following line to %s:' % rcfilepath)
        print()
        print(rcline)
        print()
    print('Installation complete.')
        
        
def get_urls(source):
    if source == 'gpaw':
        page = 'https://wiki.fysik.dtu.dk/gpaw/_sources/setups/setups.txt'
        response = urllib2.urlopen(page)
        pattern = 'https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-*.tar.gz'
        urls = [line.strip() for line in response
                if fnmatch.fnmatch(line.strip(), pattern)]
        return urls
    elif source == 'sg15':
        page = 'http://fpmd.ucdavis.edu/qso/potentials/sg15_oncv/'
        response = urllib2.urlopen(page)
        # Extract filename from:
        # <a href="...">sg15_oncv_upf_YYYY-MM-DD.tar.gz</a>
        pattern = r'>(sg15_oncv_upf_\d\d\d\d-\d\d-\d\d\.tar\.gz)</a>'
        files = re.compile(pattern).findall(response.read())
        files.sort(reverse=True)
        urls = [page + fname for fname in files]
        return urls
    elif source == 'basis':
        page = 'http://dcwww.camd.dtu.dk/~askhl/files/gpaw-lcao-basis-sets/'
        pattern = re.compile('>(gpaw-basis-.+?.tar.gz)</a>')
        response = urllib2.urlopen(page)
        files = sorted(pattern.findall(response.read()), reverse=True)
        return [page + fname for fname in files]
    elif source == 'test':
        return ['http://dcwww.camd.dtu.dk/~askhl/files/gpaw-test-source/'
                'gpaw-dist-test-source.tar.gz']
    else:
        raise ValueError('Unknown source: %s' % source)

        
def print_setups_info(p):
    try:
        import gpaw
    except ImportError as e:
        p.error('Cannot import \'gpaw\'.  GPAW does not appear to be '
                'installed. %s' % e)

    # GPAW may already have been imported, and the contents of the rc
    # file may have changed since then.  Thus, we re-import gpaw to be
    # sure that everything is as it should be.
    reload(gpaw)

    npaths = len(gpaw.setup_paths)
    if npaths == 0:
        print('GPAW currently has no setup search paths')
    else:
        print('Current GPAW setup paths in order of search priority:')
        for i, path in enumerate(gpaw.setup_paths):
            print('%4d. %s' % (i + 1, path))
