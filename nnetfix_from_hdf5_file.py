
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

run_commandline("mkdir {}".format(params.outdir))
run_commandline("mkdir models/{}".format(params.label))

print(datetime.datetime.now().time())
print(params.label)

f = h5py.File(os.path.join(os.path.abspath("datasets/"),"trainingset_{}.hdf5".format(params.label)),'r')

keys = f.keys()
Training_dataset = f[keys[0]][:]

ML_data, scaler = mlp.scale_data(Training_dataset)

# Close hdf5 file
f.close()
# IMPORTANT: Remove the original Training_dataset to free memory ,expecially for large datasets.
del Training_dataset

# Prepare X-data from the scaled trainingset.
Xdata, n_samples = mlp.prepare_X_data(ML_data)
print("X_data loaded")
#print Xdata

# Prepare y-data (the predicted part) from the scaled trainingset.
y_glitch = mlp.prepare_Y_data(ML_data)
print("y-data created")
#print y_glitch

# Delete the ML_data to free memory.
del ML_data

# Split the entire X- and y- datasets into Training and Testing sets.
X_train, X_test, X_train_full, X_test_full, y_train, y_test = mlp.split_trainingset(Xdata, y_glitch)
print("data split.")

# Fit an MLP Neural Network model to reconstruct the simulated data segments.
nnetmodel = mlp.NNetfit(X_train, y_train)
print("NNetfix model trained.")

# # SAVE ML_model using pickle

pkl_filename = "model_{}.pkl".format(params.label)
with open(os.path.join(os.path.abspath('models/{}'.format(params.label)),pkl_filename), 'wb') as file:
     pickle.dump(nnetmodel, file)
     print("model saved successfully")

scaler_filename = "scaler_{}.pkl".format(params.label)
with open(os.path.join(os.path.abspath('models/{}'.format(params.label)),scaler_filename), 'wb') as file:
     pickle.dump(scaler, file)
     print("scaler saved successfully")


# Reconstruct the testing set:
if params.save_testing_set:
    NNet_prediction =  mlp.NNetfix(nnetmodel,X_test,y_test)
    OriginalData, CutData, PredictData = mlp.reconstruct_testing_set(NNet_prediction, X_test_full, y_test)
    # Save it to check results.
    np.savetxt("{}/OriginalData.txt".format(params.outdir),OriginalData)
    np.savetxt("{}/CutData.txt".format(params.outdir),CutData)
    np.savetxt("{}/PredictData.txt".format(params.outdir),PredictData)


########### NNETFIX has now been trained. Now we use it to reconstruct the real GW data affected by the glitch. ##############

# Load real data corresponding to the given GPS time and reconstruct the signal that lies within:

GWData = dict()

#strain_raw = process_data.load_data(params.IFO)
strain_raw, whiten_psd = process_data.load_data(params.IFO, return_whiten_psd=True)
print("{} Data loaded".format(params.label))
#strain_clean = process_data.clean(strain_raw, spec_lines = process_data.spec_lines['{}_lines'.format(params.IFO)])
#print("{} Data cleaned".format(params.label))

GWData['{}_strain_raw'.format(params.IFO)] = strain_raw
#GWData['{}_strain_clean'.format(params.IFO)] = strain_clean

# Crop the data into a 10 sec. segment that NNETFIX can work with,
strain_nnetfix, start, end = process_data.crop_for_nnetfix(strain_raw)
print("Data cropped for NNetfixing")

# Scale the data using the NNETFIX model's scaler:
X_testdata_full, X_testdata, y_testglitch = mlp.process_dataframe(strain_nnetfix, scaler)

# Reconstruct the gated part. ## This is where the magic happens!
NNet_prediction = mlp.NNetfix(nnetmodel, X_testdata, y_testglitch)

OriginalData, CutData, PredictData = mlp.reconstruct_frame(NNet_prediction, scaler, X_testdata_full, y_testglitch)
print("{} NNETFIXED!".format(params.label))

# Connect the NNETFIX segments back into the original raw data segments. These can be analyzed by PE pipelines.
strain_reconstructed = process_data.rejoin_frame(PredictData, strain_raw, start, end)
strain_gated = process_data.rejoin_frame(CutData, strain_raw, start, end)
strain_original = process_data.rejoin_frame(OriginalData, strain_raw, start, end)

print("Data rejoined")

# Get initial estimates of the SNR with the best-matching template. ## This needs to be soft coded!
snr, snrp, snrl = metrics.calculate_snr(strain_raw, 35, 29)
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

print("{} NNETFIXED. GREAT SUCCESS!".format(params.label))

print(datetime.datetime.now().time())
#run_commandline("./generate_trainingset.py 13")

print("Recoloring whitened data and saving...")

print("Recoloring...")
recolored_predict_data = process_data.recolor_np_array(PredictData, whiten_psd, params.sample_rate)
recolored_cut_data = process_data.recolor_np_array(CutData, whiten_psd, params.sample_rate)
recolored_original_data = process_data.recolor_np_array(OriginalData, whiten_psd, params.sample_rate)

print("Processing...")
strain_reconstructed_recolored = process_data.rejoin_frame(recolored_predict_data, strain_raw, start, end)
strain_gated_recolored = process_data.rejoin_frame(recolored_cut_data, strain_raw, start, end)
strain_original_recolored = process_data.rejoin_frame(recolored_original_data, strain_raw, start, end)

print("Saving...")
strain_reconstructed.write(os.path.join(params.outdir,"{}_{}_reconstructed_recolored.gwf".format(params.IFO,params.label)))
strain_gated.write(os.path.join(params.outdir,"{}_{}_gated_recolored.gwf".format(params.IFO,params.label)))
strain_raw.write(os.path.join(params.outdir,"{}_{}_original_recolored.gwf".format(params.IFO,params.label)))
print("Frames saved!")
