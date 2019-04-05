import numpy as np
from nnetfix import params
import nnetfix.trainingset.templatebank as tbank
import nnetfix.trainingset.trainingset_utils as tsutils
from nnetfix.tools.utils import run_commandline
from nnetfix.tools import metrics, process_data
from nnetfix import mlp
import os 
import datetime
import h5py
import pickle

print(datetime.datetime.now().time())
run_commandline("mkdir {}".format(params.outdir))
#run_commandline("mkdir models/{}".format(params.label))
print(params.label)

xml_filename = tbank.generate_tbank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)

txt_filename = tsutils._xml_to_txt(xml_filename,params.label)

n_templates = tsutils.make_hdf5(txt_filename)
#tsutils.write_condor_submit_file("generate_trainingset.py",n_templates)

#run_commandline("condor_submit {0}/condor_{1}.sub".format(os.path.abspath('datasets'),params.label))
#run_commandline("condor_wait {}/LOG/mainlog.log".format(params.outdir))

print("training dataset saved")


f = h5py.File(os.path.join(os.path.abspath("datasets/"),"trainingset_{}.hdf5".format(params.label)),'r')

keys = f.keys()
Training_dataset = f[keys[0]][:]

ML_data, scaler = mlp.scale_data(Training_dataset)

f.close()

Xdata, n_samples = mlp.prepare_X_data(ML_data)

print("X_data loaded")
#print Xdata

y_glitch = mlp.prepare_Y_data(ML_data)
print("y-data created")
#print y_glitch

X_train, X_test, X_train_full, X_test_full, y_train, y_test = mlp.split_trainingset(Xdata, y_glitch)

print("data split")

nnetmodel = mlp.NNetfit(X_train, y_train)

print("model trained. GREAT SUCCESS!")

# # SAVE ML_model using pickle

pkl_filename = "model_{}.pkl".format(params.label)
with open(os.path.join(os.path.abspath('models/'.format(params.label)),pkl_filename), 'wb') as file:
     pickle.dump(nnetmodel, file)
     print("model saved successfully")

scaler_filename = "scaler_{}.pkl".format(params.label)
with open(os.path.join(os.path.abspath('models/'.format(params.label)),scaler_filename), 'wb') as file:
     pickle.dump(scaler, file)
     print("scaler saved successfully")


#NNet_prediction =  mlp.NNetfix(nnetmodel,X_test,y_test)

# Reconstruct samples in the testing set.

#OriginalData, CutData, PredictData = mlp.reconstruct_testing_set(NNet_prediction, X_test_full, y_test)

#np.savetxt("{}/OriginalData.txt".format(params.outdir),OriginalData)
#np.savetxt("{}/CutData.txt".format(params.outdir),CutData)
#np.savetxt("{}/PredictData.txt".format(params.outdir),PredictData)

#print PredictData.shape
#print("GREATER SUCCESS")

# Load real data corresponding to the given GPS time and reconstruct the signal that lies within:

GWData = dict()

strain_raw = process_data.load_data(params.IFO)
print("{} Data loaded".format(params.label))
strain_clean = process_data.clean(strain_raw, 30., 600., spec_lines = process_data.spec_lines['{}_lines'.format(params.IFO)])
print("{} Data cleaned".format(params.label))

GWData['{}_strain_raw'.format(params.IFO)] = strain_raw
GWData['{}_strain_clean'.format(params.IFO)] = strain_clean

strain_nnetfix, start, end = process_data.crop_for_nnetfix(strain_clean)
print("Data cropped for NNetfixing")

X_testdata_full, X_testdata, y_testglitch = mlp.process_dataframe(strain_nnetfix, scaler)

NNet_prediction = mlp.NNetfix(nnetmodel, X_testdata, y_testglitch)

OriginalData, CutData, PredictData = mlp.reconstruct_frame(NNet_prediction, scaler, X_testdata_full, y_testglitch)
print("{} NNETFIXED".format(params.label))

strain_reconstructed = process_data.rejoin_frame(PredictData, strain_raw, start, end)
strain_gated = process_data.rejoin_frame(CutData, strain_raw, start, end)
strain_original = process_data.rejoin_frame(OriginalData, strain_raw, start, end)

print("Data rejoined")
snr, snrp, snrl = metrics.calculate_snr(strain_clean, 35, 29)
print("The raw SNR is {} at {}.".format(snrp, snrl))

snr, snrp, snrl = metrics.calculate_snr(strain_reconstructed, 35, 29)
print("The reconstructed SNR is {} at {}.".format(snrp, snrl))

snr, snrp, snrl = metrics.calculate_snr(strain_gated, 35, 29)
print("The gated SNR is {} at {}.".format(snrp, snrl))

snr, snrp, snrl = metrics.calculate_snr(strain_original, 35, 29)
print("The new original SNR is {} at {}.".format(snrp, snrl))

strain_reconstructed.write(os.path.join(params.outdir,"{}_{}_reconstructed.gwf".format(params.IFO,params.label)))
strain_gated.write(os.path.join(params.outdir,"{}_{}_gated.gwf".format(params.IFO,params.label)))
strain_raw.write(os.path.join(params.outdir,"{}_{}_original.gwf".format(params.IFO,params.label)))



print(datetime.datetime.now().time())
#run_commandline("./generate_trainingset.py 13")
