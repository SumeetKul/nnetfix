import logging
import os
from math import fmod
import argparse
import traceback
import inspect
import subprocess



def run_commandline(cl, log_level=20, raise_error=True, return_output=False):
    """Run a string cmd as a subprocess, check for errors and return output.

    Parameters
    ----------
    cl: str
        Command to run
    log_level: int
        See https://docs.python.org/2/library/logging.html#logging-levels,
        default is '20' (INFO)

    """

#    logger.log(log_level, 'Now executing: ' + cl)
    if return_output:
        try:
            out = subprocess.check_output(
                cl, stderr=subprocess.STDOUT, shell=True,
                universal_newlines=True)
        except subprocess.CalledProcessError as e:
#            logger.log(log_level, 'Execution failed: {}'.format(e.output))
            if raise_error:
                raise
            else:
                out = 0
        os.system('\n')
        return(out)
    else:
        process = subprocess.Popen(cl, shell=True)
        process.communicate()

#def _make_executable(filename, outdir=params.outdir):

#    """ Make scripts executable to run using Condor.
#    filename: name of the script
#    """

#    path_to_file = os.path.join(outdir,filename)

#    command = "chmod u+x {}".format(path_to_file)

#    return "Script {} made executable".format(filename)
