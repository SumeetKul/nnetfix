import numpy as np
from nnetfix.tools import make_injections as mk_inj
from nnetfix.tools import metrics, process_data
from nnetfix import mlp
from nnetfix import params
import os
import pickle
import h5py

# Model name:
model_name = "model_2_{}.pkl".format(params.label)
filepath = os.path.join(os.path.abspath("models/{}/".format(params.label)),model_name)

nnetmodel = pickle.load(open(filepath, 'rb'))
print("model loaded successfully")

# Load scaler:
scaler_name = "scaler_2_{}.pkl".format(params.label)
sc_filepath = os.path.join(os.path.abspath("models/{}/".format(params.label)),scaler_name)

scaler = pickle.load(open(sc_filepath,'rb'))
print("scaler loaded successfully")


#f = h5py.File(os.path.join(os.path.abspath("datasets/"),"trainingset_{}.hdf5".format(params.label)),'r')

#keys = f.keys()
#Training_dataset = f[keys[0]][:]

#ML_data, scaler = mlp.scale_data(Training_dataset)

# Close hdf5 file
#f.close()
# IMPORTANT: Remove the original Training_dataset to free memory ,expecially for large datasets.
#del Training_dataset

# Prepare X-data from the scaled trainingset.
#Xdata, n_samples = mlp.prepare_X_data(ML_data)
#print("X_data loaded")
#print Xdata

# Prepare y-data (the predicted part) from the scaled trainingset.
#y_glitch = mlp.prepare_Y_data(ML_data)
#print("y-data created")
#print y_glitch

# Delete the ML_data to free memory.
#del ML_data

# Split the entire X- and y- datasets into Training and Testing sets.
#X_train, X_test, X_train_full, X_test_full, y_train, y_test = mlp.split_trainingset(Xdata, y_glitch)
#print("data split.")


# Reconstruct the testing set:
#NNet_prediction =  mlp.NNetfix(nnetmodel,X_test,y_test)
#OriginalData, CutData, PredictData = mlp.reconstruct_testing_set(NNet_prediction, X_test_full, y_test)
# Save it to check results.
#np.savetxt("{}/OriginalData.txt".format(params.outdir),OriginalData)
#np.savetxt("{}/CutData.txt".format(params.outdir),CutData)
#np.savetxt("{}/PredictData.txt".format(params.outdir),PredictData)


GWData = dict()

strain_raw = process_data.load_data(params.IFO)
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

np.savetxt("real_data.txt",np.array(snrlist))

strain_reconstructed.name = "L1:GDS-CALIB_STRAIN"
strain_gated.name = "L1:GDS-CALIB_STRAIN"

strain_reconstructed.write(os.path.join(params.outdir,"{}_{}_2_reconstructed.gwf".format('L1',params.label)))
strain_gated.write(os.path.join(params.outdir,"{}_{}_2_gated.gwf".format('L1',params.label)))
strain_original.write(os.path.join(params.outdir,"{}_{}_2_original.gwf".format('L1',params.label)))

