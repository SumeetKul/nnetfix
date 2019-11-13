# Runs NNETFIX (Evaluates a saved NNETFIX model) on a series of injections with defined parameters. The parameters of the glitch are the same as the ones used while training the model (check model metadata). 

import numpy as np
import os 
import pickle

import json

from astropy.table import Table

from nnetfix.tools import make_injections as mk_inj
from nnetfix.tools import metrics, process_data
#from nnetfix.tools import utils
from nnetfix import mlp
from nnetfix import params

# Create directory to save data:
base_dir = "simulation_data_output"
data_sub_dir = "tbm_{0}ms_dur_{1}ms".format(int(params.glitch_tbm*1000), int(params.glitch_dur * 1000))
data_dir = os.path.join(base_dir, "injections", params.label, data_sub_dir)
#os.makedirs(data_dir, exists_ok = True)
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

snr_json_filename = os.path.join(data_dir, "snr.json")

with open(snr_json_filename, "r") as infile:
    snr_json = json.load(infile)

# Parameters for Injections:

#n_injections = 50 # Number of injections
#n_injections = 5 # Number of injections
n_injections = snr_json["n_injections"] # Number of injections
#mass1 = (34,36) # Small interval around actual mass value
#mass2 = (28,30)
#mass1 = 35
#mass2 = 29
#snr_range = (8,25)
mass1 = snr_json["mass1"]
mass2 = snr_json["mass2"]
snr_range = snr_json["snr_range"]

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

ifo = "L1"
#ifo = "H1"

for i in range(n_injections):
    print("i")
    print(i)


    m1 = mass1
    m2 = mass2
#    m1 = np.round(np.random.uniform(mass1[0],mass1[1]),2)
#    m2 = np.round(np.random.uniform(mass2[0],mass2[1]),2)
    #snr = np.round(np.random.uniform(snr_range[0],snr_range[1]),2)
    snr = snr_json["snrs"][i]
    right_ascension = snr_json["RAs"][i]
    declination = snr_json["declinations"][i]
    polarization = snr_json["polarizations"][i]

    #for ifo in ["L1", "H1"]:
    #for ifo in ["H1"]:
    #print(ifo)
    
    #testseg, inj_arr = mk_inj.inject_signal(m1, m2, snr, 'L1')
    #testseg, inj_arr = mk_inj.inject_signal(m1, m2, snr, ifo)
    testseg, inj_arr = mk_inj.inject_signal(m1, m2, snr, ifo, right_ascension, declination, polarization)
    
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

    #if i % 10 == 0:
    #    print("{} out of {} injection frames NNETFIXED successfully!".format(i,n_injections))
    #    #np.savetxt(os.path.abspath("injections/{0}/{1}/reconstruct_{2}.txt".format(params.label, data_dir, i/10)),PredictData)
    #    np.savetxt(os.path.abspath(os.path.join(data_dir, "reconstruct_{0}.txt".format(i/10))),PredictData)
    #    np.savetxt(os.path.abspath(os.path.join(data_dir, "gated_{0}.txt".format(i/10))),CutData)
    #    #np.savetxt(os.path.abspath("injections/{0}/{1}/original_{2}.txt".format(params.label, data_dir, i/10)),OriginalData)
    #    np.savetxt(os.path.abspath(os.path.join(data_dir, "original_{0}.txt".format(i/10))),OriginalData)
    np.savetxt(os.path.abspath(os.path.join(data_dir, "reconstruct_{0}_{1}.txt".format(i, ifo))),PredictData)
    np.savetxt(os.path.abspath(os.path.join(data_dir, "gated_{0}_{1}.txt".format(i, ifo))),CutData)
    np.savetxt(os.path.abspath(os.path.join(data_dir, "original_{0}_{1}.txt".format(i, ifo))),OriginalData)

np.savetxt(os.path.abspath(os.path.join(data_dir, "Inj_snr_" + ifo + ".csv")),SNR_array, fmt='%1.3f',delimiter=',',header="Index, Original, Gated, Reconstructed")
np.savetxt(os.path.abspath(os.path.join(data_dir, "Inj_chisq_" + ifo + ".csv")),chisq_array,fmt='%1.3f',delimiter=',',header="Index, Original, Gated, Reconstructed")
np.savetxt(os.path.abspath(os.path.join(data_dir, "Inj_params_" + ifo + ".csv")),inj_param_array,fmt='%1.3f',delimiter=',',header="Index, mass1, mass2, RA, DEC, Polarization, SNR")
#np.savetxt(os.path.abspath("injections/{0}/{1}/Inj_snr.csv".format(params.label,data_dir)),SNR_array, fmt='%1.3f',delimiter=',',header="Index, Original, Gated, Reconstructed")
#np.savetxt(os.path.abspath("injections/{0}/{1}/Inj_chisq.csv".format(params.label,data_dir)),chisq_array,fmt='%1.3f',delimiter=',',header="Index, Original, Gated, Reconstructed")
#np.savetxt(os.path.abspath("injections/{0}/{1}/Inj_params.csv".format(params.label,data_dir)),inj_param_array,fmt='%1.3f',delimiter=',',header="Index, mass1, mass2, RA, DEC, Polarization, SNR")

################################################

orig = SNR_array.T[1]
cut = SNR_array.T[2]
recon = SNR_array.T[3]

cut_err = abs(orig-cut)/orig * 100
recon_err = abs(orig-recon)/orig * 100

snr_mean = [np.mean(recon_err),np.mean(cut_err)]
snr_stdev = [np.std(recon_err),np.std(cut_err)]
#np.savetxt(os.path.abspath("injections/{0}/{1}/snr_mean.txt".format(params.label, data_dir)),snr_mean)
#np.savetxt(os.path.abspath("injections/{0}/{1}/snr_stdev.txt".format(params.label, data_dir)),snr_stdev)
np.savetxt(os.path.abspath(os.path.join(data_dir, "snr_mean_" + ifo + ".txt")),snr_mean)
np.savetxt(os.path.abspath(os.path.join(data_dir, "snr_stdev_" + ifo + ".txt")),snr_stdev)

##############################################

ch_orig = chisq_array.T[1]
ch_cut = chisq_array.T[2]
ch_recon = chisq_array.T[3]

ch_cut_err = abs(ch_orig-ch_cut)/ch_orig * 100
ch_recon_err = abs(ch_orig-ch_recon)/ch_orig * 100

chisq_mean = [np.mean(ch_recon_err),np.mean(ch_cut_err)]
chisq_stdev = [np.std(ch_recon_err),np.std(ch_cut_err)]

#np.savetxt(os.path.abspath("injections/{0}/{1}/chisq_mean.txt".format(params.label, data_dir)),chisq_mean)
#np.savetxt(os.path.abspath("injections/{0}/{1}/chisq_stdev.txt".format(params.label, data_dir)),chisq_stdev)
np.savetxt(os.path.abspath(os.path.join(data_dir, "chisq_mean_" + ifo + ".txt")),chisq_mean)
np.savetxt(os.path.abspath(os.path.join(data_dir, "chisq_stdev_" + ifo + ".txt")),chisq_stdev)


#################################################################
#table = Table(SNR_array)
#table.write(os.path.abspath("injections/{}/Inj_SNR.csv".format(data_dir)), format='csv')
GWData = dict()

#strain_raw = process_data.load_data(params.IFO)
strain_raw = process_data.load_data(ifo)
print("Data loaded")
#strain_clean = process_data.clean(strain_raw, spec_lines = process_data.spec_lines['{}_lines'.format(params.IFO), 30., 600.)
#print("Data cleaned")

#GWData['{}_strain_raw'.format(params.IFO)] = strain_raw
#GWData['{}_strain_clean'.format(params.IFO)] = strain_clean

strain_nnetfix, start, end = process_data.crop_for_nnetfix(strain_raw)
print("Data cropped for NNetfixing")

X_testdata_full, X_testdata, y_testglitch = mlp.process_dataframe(strain_nnetfix, scaler)

NNet_prediction = mlp.NNetfix(nnetmodel, X_testdata, y_testglitch)

OriginalData, CutData, PredictData = mlp.reconstruct_frame(NNet_prediction, scaler, X_testdata_full, y_testglitch)
print("Data NNETFIXED")

strain_reconstructed = process_data.rejoin_frame(PredictData, strain_raw, start, end)
strain_gated = process_data.rejoin_frame(CutData, strain_raw, start, end)
strain_original = process_data.rejoin_frame(OriginalData, strain_raw, start, end)

snrlist = []

print("Data rejoined")
snr, snrp, snrl = metrics.calculate_snr(strain_raw, 35, 29)
print("The raw SNR is {} at {}.".format(snrp, snrl))
snrlist.append(snrp)

snr, snrp, snrl = metrics.calculate_snr(strain_reconstructed, 35, 29)
print("The reconstructed SNR is {} at {}.".format(snrp, snrl))
snrlist.append(snrp)

snr, snrp, snrl = metrics.calculate_snr(strain_gated, 35, 29)
print("The gated SNR is {} at {}.".format(snrp, snrl))
snrlist.append(snrp)

snr, snrp, snrl = metrics.calculate_snr(strain_original, 35, 29)
print("The new original SNR is {} at {}.".format(snrp, snrl))

#np.savetxt(os.path.abspath("injections/{0}/{1}/real_data.txt".format(params.label, data_dir)),np.array(snrlist))
np.savetxt(os.path.abspath(os.path.join(data_dir, "real_data_" + ifo + ".txt")),np.array(snrlist))


#strain_reconstructed.name = "{}:GDS-CALIB_STRAIN".format(params.IFO)
#strain_gated.name = "{}:GDS-CALIB_STRAIN".format(params.IFO)
strain_reconstructed.name = "{}:GDS-CALIB_STRAIN".format(ifo)
strain_gated.name = "{}:GDS-CALIB_STRAIN".format(ifo)


np.savetxt(os.path.abspath(os.path.join(data_dir, "original_{0}_{1}.txt".format(i, ifo))),OriginalData)
#strain_reconstructed.write(os.path.abspath("injections/{0}/{1}/{2}_{3}_reconstructed.gwf".format(params.label, data_dir,params.IFO,params.label)))
#strain_gated.write(os.path.abspath("injections/{0}/{1}/{2}_{3}_gated.gwf".format(params.label, data_dir,params.IFO,params.label)))
#strain_original.write(os.path.abspath("injections/{0}/{1}/{2}_{3}_original.gwf".format(params.label, data_dir,params.IFO,params.label)))

#strain_reconstructed.write(os.path.abspath(os.path.join(data_dir, "{0}_{1}_reconstructed.gwf".format(IFO,params.label))))
#strain_gated.write(os.path.abspath(os.path.join(data_dir, "{0}_{1}_gated.gwf".format(IFO,params.label))))
#strain_original.write(os.path.abspath(os.path.join(data_dir, "{0}_{1}_original.gwf".format(IFO,params.label))))

strain_reconstructed.write(os.path.abspath(os.path.join(data_dir, "{0}_{1}_reconstructed.gwf".format(ifo,params.label))))
strain_gated.write(os.path.abspath(os.path.join(data_dir, "{0}_{1}_gated.gwf".format(ifo,params.label))))
strain_original.write(os.path.abspath(os.path.join(data_dir, "{0}_{1}_original.gwf".format(ifo,params.label))))