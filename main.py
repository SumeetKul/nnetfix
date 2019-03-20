import numpy as np
import params
import trainingset.templatebank as tbank
import trainingset.trainingset_utils as tsutils
from tools.utils import run_commandline
import datetime

print(datetime.datetime.now().time())

run_commandline("mkdir {}".format(params.outdir))
print(params.label)

xml_filename = tbank.generate_tbank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)

txt_filename = tsutils._xml_to_txt(xml_filename,params.label)

n_templates = tsutils.make_hdf5(txt_filename)
tsutils.write_condor_submit_file("generate_trainingset.py",n_templates)
#seg = simulate_single_data_segment(13,7,2)
run_commandline("condor_submit condor_{}.sub".format(params.label))
run_commandline("condor_wait {}/LOG/mainlog.log".format(params.outdir))

print("training dataset saved")
print(datetime.datetime.now().time())
#run_commandline("./generate_trainingset.py 13")
