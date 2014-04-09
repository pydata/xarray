 #!/usr/bin/env python
import os
import re
import sys
import warnings
try:
    from setuptools import setup
except:
    from distutils.core import setup

MAJOR = 0
MINOR = 1
MICRO = 0
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

# code to extract and write the version copied from pandas, which is available
# under the BSD license:
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if os.path.exists('src/xray/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing xray/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis does this)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v')

else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(
            os.path.dirname(__file__), 'src', 'xray', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()

if write_version:
    write_version_py()


setup(name='xray',
      version=FULLVERSION,
      description='Extended arrays for working with scientific datasets',
      author='Stephan Hoyer, Alex Kleeman, Eugene Brevdo',
      author_email='TODO',
      install_requires=['numpy >= 1.8', 'pandas >= 0.13.1'],
      tests_require=['mock >= 1.0.1', 'nose >= 1.0'],
      url='https://github.com/akleeman/xray',
      test_suite='nose.collector',
      packages=['xray'],
      package_dir={'': 'src'})
