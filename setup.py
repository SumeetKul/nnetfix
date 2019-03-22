#!/usr/bin/env python

from setuptools import setup
import subprocess
import os

def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = ''

    version_file = '.version'
    if os.path.isfile(version_file) is False:
        with open('nnetfix/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file

VERSION = '0.0.1'
version_file = write_version_file(VERSION)


setup(name='nnetfix',
      description="A Neural Network to 'fix' Gravitational Wave signals coincident with short-duration glitches in LIGO-Virgo data.",
      url='https://git.ligo.org/sumeet.kulkarni/nnetfix',
      author='Sumeet Kulkarni',
      author_email='sskulkar@go.olemiss..edu',
      license="LICENSE.md",
      version=VERSION,
      packages=['nnetfix', 'nnetfix.tools', 'nnetfix.trainingset'],
      package_dir={'nnetfix': 'nnetfix'},
      install_requires=[
          'future',
          'dynesty',
          'numpy>=1.9',
          'matplotlib>=2.0',
          'scipy',
          'h5py==2.7.1',
          'gwpy',
          'pycbc',
          'lalsuite'],
      classifiers=[
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"])




