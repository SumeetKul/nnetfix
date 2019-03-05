import numpy as np
import params
import trainingset.templatebank as tbank
import trainingset.trainingset_utils as tsutils
from core.utils import run_commandline

run_commandline("mkdir {}".format(params.outdir))
print(params.label)

xml_filename = tbank.generate_tbank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)

txt_filename = tsutils._xml_to_txt(xml_filename,params.label)

hdf = tsutils.make_hdf5(txt_filename)

#seg = simulate_single_data_segment(13,7,2)
