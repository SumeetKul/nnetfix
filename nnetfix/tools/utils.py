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


## Normalisation of all vectors:
def normalise(tbank):
    for i in range(tbank.shape[0]):
        tbank[i] = tbank[i]/np.linalg.norm(tbank[i])
    return tbank    
        
    
## Pairwise Inner Products:
def InnerProduct(R):
    CD = pdist(R,'cosine')
    CD = list(1 - CD)
    m,n = R.shape
    for i in xrange(0,m):
        CD.append(np.inner(R[i],R[i]))
    return CD 


## Pairwise Euclidean distances:
def EuclideanDist(R):
    ED = pdist(R,'euclidean')
    return ED

## To normalise individiual vectors:
def vectornorm(v):
    v = v/np.linalg.norm(v)
    return v


def load_dataset(hdf5_file):
    try:
	f = h5py.File(os.path.join(os.path.abspath('datasets/'),hdf5_file),'r')
	keys = f.keys()
	# Load dataset as a numpy array:
	Main_Training_dataset = f[keys[0]][:]

	#Close hdf5
	f.close()

	return Main_Training_dataset

    except NameError:
	print("The Training dataset does not exist as the given hdf5 filename")






	#def _make_executable(filename, outdir=params.outdir):

#    """ Make scripts executable to run using Condor.
#    filename: name of the script
#    """

#    path_to_file = os.path.join(outdir,filename)

#    command = "chmod u+x {}".format(path_to_file)

#    return "Script {} made executable".format(filename)
