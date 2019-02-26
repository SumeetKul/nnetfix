import numpy as np
import params
import trainingset.templatebank as tbank
from trainingset.generate_data import make_hdf5
from core.utils import run_commandline

run_commandline("mkdir {}".format(params.outdir))
print(params.label)

xml_filename = tbank.generate_tbank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)

txt_filename = tbank._xml_to_txt(xml_filename,params.label)

hdf = make_hdf5(txt_filename)
