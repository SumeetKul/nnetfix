import numpy as np
import h5py
from nnetfix import params
import os
import subprocess
from nnetfix.tools.utils import run_commandline


def _xml_to_txt(filename, label):
    
    # Convert xml into txt:
    xmlfile = filename
    #xmlfile = os.path.join(params.outdir,xml_filename)
    cache_file = "templatebank_{}.txt".format(label)
    output_cache_file = os.path.join(params.outdir,cache_file)

    cl_list = ['ligolw_print ']
    cl_list.append('{}'.format(xmlfile))
    cl_list.append('-t sngl_inspiral')
    cl_list.append('-c mass1')
    cl_list.append('-c mass2')
    
    cl_list.append('> {}'.format(output_cache_file))
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_cache_file




def make_hdf5(txtfile, multiplier=params.multiplier, dur=params.duration, sample_rate=params.sample_rate, outdir=params.outdir, label=params.label):
    """
    Generates a blank hdf5 dataset file for storing the trainingset samples having simulated data. 

    Inputs:
    txtfile: Name of the text file having mass parameters (m1,m2) of the templatebank used for generating the trainingset; templatebank_{label}.txt
    
    multiplier: The number of copies of each template to include in the trainingset. 
    """
   
   

    output_cache_file = os.path.join(outdir, txtfile)

    n_templates = np.loadtxt(txtfile,delimiter=',').shape[0]

    signal_nsample_points = int(dur*sample_rate)
    # Total number of samples in the entire training set.
    n_samples = n_templates * multiplier
    
    hdffilename = "trainingset_{}.hdf5".format(label)
    hdf_file = os.path.join(os.path.abspath('datasets'),hdffilename)
    f = h5py.File(hdf_file,"w")

    f.create_dataset("trainingset",(n_samples,signal_nsample_points))

    f.close()
    print("hdf5 file successfully generated")
    return n_templates

def write_condor_submit_file(exec_name,n_templates,label=params.label, outdir = params.outdir, multiplier = params.multiplier):

    log_dir = "LOG"
    err_dir = "ERR"
    out_dir = "OUT"

    run_commandline("mkdir {}".format(os.path.join(outdir,log_dir)))
    run_commandline("mkdir {}".format(os.path.join(outdir,err_dir)))
    run_commandline("mkdir {}".format(os.path.join(outdir,out_dir)))

    con_list = ['Universe = vanilla\n']
    con_list.append('getenv = True\n')
    con_list.append('Executable = {}\n'.format(os.path.abspath(exec_name)))
    con_list.append('Arguments  = $(Process)\n')
    con_list.append('Log = {}/mainlog.log\n'.format(os.path.abspath(os.path.join(outdir,log_dir))))
    con_list.append('Error = {}/$(Process).err\n'.format(os.path.abspath(os.path.join(outdir,err_dir))))
    con_list.append('Output = {}/$(Process).out\n'.format(os.path.abspath(os.path.join(outdir,out_dir))))
    con_list.append('accounting_group = ligo.dev.o3.detchar.explore.test\n')
    con_list.append('Queue {}'.format(n_templates))
    con_tent = ''.join(con_list)

    condor_filename = "condor_{}.txt".format(params.label)

    f = open(os.path.join(os.path.abspath('datasets'),condor_filename),'w')
    f.write(con_tent)
    f.close()

    run_commandline("mv {0}/condor_{1}.txt {0}/condor_{1}.sub".format(os.path.abspath('datasets'),params.label))

