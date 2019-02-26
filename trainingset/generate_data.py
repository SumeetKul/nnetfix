import numpy as np
import h5py
import params
import os
def make_hdf5(txtfile,multiplier=params.multiplier,dur=params.duration,sample_rate=params.sample_rate):
    """
    Generates a blank hdf5 dataset file for storing the trainingset samples having simulated data. 

    Inputs:
    txtfile: Name of the text file having mass parameters (m1,m2) of the templatebank used for generating the trainingset; templatebank_{label}.txt
    
    multiplier: The number of copies of each template to include in the trainingset. 
    """
   
    output_cache_file = os.path.join(params.outdir, txtfile)

    n_templates = np.loadtxt(txtfile,delimiter=',').shape[0]

    signal_nsample_points = int(dur*sample_rate)
    # Total number of samples in the entire training set.
    n_samples = ntemplates * multiplier

    f = h5py.File("trainingset_{}.hdf5".format(params.label))

    f.create_dataset("trainingset",(n_samples,signal_nsample_points))

    f.close()
    print("hdf5 file successfully generated")
    return output_cache_file

