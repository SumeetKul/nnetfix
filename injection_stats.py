# Runs NNETFIX (Evaluates a saved NNETFIX model) on a series of injections with defined parameters. The parameters of the glitch are the same as the ones used while training the model (check model metadata). 

import numpy as np
from nnetfix.tools import make_injections as mk_inj
from nnetfix.tools import metrics
from nnetfix.tools import utils
from nnetfix import mlp
from nnetfix import params
import os 
import pickle

# Parameters for Injections:

mass1_interval = (34,36)
mass2_interval = (28,30)
snr_range = (8,25)

# Create directory to save data:
#data_dir = "tg_{0}_dur_{1}".format(params.glitch_t*100, params.glitch_dur * 100)
#utils.run_commandline("mkdir injections/{}".format(data_dir))

# Model name:
model_name = "model_{}.pkl".format(params.label)
filepath = os.path.join(os.path.abspath("models/"),model_name)

nnetmodel = pickle.load(open(filepath, 'rb'))
print("model loaded successfully")

# Load scaler:
scaler_name = "scaler_{}.pkl".format(params.label)
sc_filepath = os.path.join(os.path.abspath("models/"),scaler_name)

scaler = pickle.load(open(sc_filepath,'rb'))
print("scaler loaded successfully")

# Generate test injection and find snr, chisq

testseg, test_arr = mk_inj.inject_signal(35, 29, 30, 'L1')

snr, snrp, snrp_loc = metrics.calculate_snr(testseg, 35, 29)

X_testdata_full, X_testdata, y_testglitch = mlp.process_dataframe(testseg, scaler)
print("Data processed")
print y_testglitch.shape

NNet_prediction = mlp.NNetfix(nnetmodel, X_testdata, y_testglitch)
print("Frame fixed")
OriginalData, CutData, PredictData = mlp.reconstruct_frame(NNet_prediction, scaler, X_testdata_full, y_testglitch)
print PredictData.shape
print("Reconstructed successfully")
print snrp
print snrp_loc

chisq, chisq_min = metrics.calculate_chisq(testseg, 35, 29, snrp_loc)

print chisq_min
