# Runs NNETFIX (Evaluates a saved NNETFIX model) on a series of injections with defined parameters. The parameters of the glitch are the same as the ones used while training the model (check model metadata). 

import numpy as np
from nnetfix.tools import make_injections as mk_inj
from nnetfix.tools import metrics
from nnetfix.tools import utils
from nnetfix import mlp
from nnetfix import params
import os 
import pickle
from astropy.table import Table

# Parameters for Injections:

n_injections = 32 # Number of injections
mass1 = (34,36) # Small interval around actual mass value
mass2 = (28,30)
snr_range = (8,25)

# Create directory to save data:
data_dir = "tbm_{0}ms_dur_{1}ms".format(int(params.glitch_tbm*1000), int(params.glitch_dur * 1000))
utils.run_commandline("mkdir injections/{0}".format(params.label))
utils.run_commandline("mkdir injections/{0}/{1}".format(params.label, data_dir))

# Model name:
model_name = "model_{}.pkl".format(params.label)
filepath = os.path.join(os.path.abspath("models/{}/".format(params.label)),model_name)

nnetmodel = pickle.load(open(filepath, 'rb'))
print("model loaded successfully")

# Load scaler:
scaler_name = "scaler_{}.pkl".format(params.label)
sc_filepath = os.path.join(os.path.abspath("models/{}/".format(params.label)),scaler_name)

scaler = pickle.load(open(sc_filepath,'rb'))
print("scaler loaded successfully")

# Create arrays to save injection parameters and snr, chi_sq values:

inj_param_array = np.zeros((n_injections,7))
SNR_array = np.zeros((n_injections, 4))
chisq_array = np.zeros((n_injections, 4))

for i in range(n_injections):



	m1 = np.round(np.random.uniform(mass1[0],mass1[1]),2)
	m2 = np.round(np.random.uniform(mass2[0],mass2[1]),2)
	snr = np.round(np.random.uniform(snr_range[0],snr_range[1]),2)
	
	testseg, inj_arr = mk_inj.inject_signal(m1, m2, snr, 'L1')
	
	inj_param_array[i] = [i, m1, m2] + inj_arr
	X_testdata_full, X_testdata, y_testglitch = mlp.process_dataframe(testseg, scaler)

	NNet_prediction = mlp.NNetfix(nnetmodel, X_testdata, y_testglitch)

	OriginalData, CutData, PredictData = mlp.reconstruct_frame(NNet_prediction, scaler, X_testdata_full, y_testglitch)
	
	
	snr_orig, snrp_orig, snrp_loc_orig = metrics.calculate_snr(OriginalData, m1, m2)
        snr, snrp, snrp_loc = metrics.calculate_snr(testseg, m1, m2)
	snr_cut, snrp_cut, snrp_loc_cut = metrics.calculate_snr(CutData, m1, m2)
	snr_pred, snrp_pred, snrp_loc_pred = metrics.calculate_snr(PredictData, m1, m2)

	
	SNR_array[i] = np.array([i, snrp_orig, snrp_cut, snrp_pred])

	chisq_orig, chisq_min_orig = metrics.calculate_chisq(OriginalData, m1, m2, snrp_loc_orig)
	chisq, chisq_min = metrics.calculate_chisq(testseg, m1, m2, snrp_loc)
	chisq_cut, chisq_min_cut = metrics.calculate_chisq(CutData, m1, m2, snrp_loc_cut)
	chisq_pred, chisq_min_pred = metrics.calculate_chisq(PredictData, m1, m2, snrp_loc_pred)

	
	chisq_array[i] = [i, chisq_min_orig, chisq_min_cut, chisq_min_pred]

	if i % 10 == 0:
		print("{} out of {} injection frames NNETFIXED successfully!".format(i,n_injections))
		np.savetxt(os.path.abspath("injections/{0}/{1}/reconstruct_{2}.txt".format(params.label, data_dir, i/10)),PredictData)
		np.savetxt(os.path.abspath("injections/{0}/{1}/original_{2}.txt".format(params.label, data_dir, i/10)),OriginalData)

np.savetxt(os.path.abspath("injections/{0}/{1}/Inj_snr.csv".format(params.label,data_dir)),SNR_array, fmt='%1.3f',delimiter=',',header="Index, Original, Gated, Reconstructed")
np.savetxt(os.path.abspath("injections/{0}/{1}/Inj_chisq.csv".format(params.label,data_dir)),chisq_array,fmt='%1.3f',delimiter=',',header="Index, Original, Gated, Reconstructed")
np.savetxt(os.path.abspath("injections/{0}/{1}/Inj_params.csv".format(params.label,data_dir)),inj_param_array,fmt='%1.3f',delimiter=',',header="Index, mass1, mass2, RA, DEC, Polarization, SNR")

#table = Table(SNR_array)
#table.write(os.path.abspath("injections/{}/Inj_SNR.csv".format(data_dir)), format='csv')

