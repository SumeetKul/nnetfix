
from nnetfix import params
import nnetfix.trainingset.templatebank as tbank
import nnetfix.trainingset.trainingset_utils as tsutils
from nnetfix.tools.utils import run_commandline
import datetime

print(datetime.datetime.now().time())
run_commandline("mkdir {}".format(params.outdir))
run_commandline("mkdir models/{}".format(params.label))
print(params.label)

xml_filename = tbank.generate_tbank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)

txt_filename = tsutils._xml_to_txt(xml_filename,params.label)

n_templates = tsutils.make_hdf5(txt_filename)

tsutils.write_condor_submit_file("generate_trainingset_new.py",n_templates)