import numpy as np
from nnetfix import params
import nnetfix.trainingset.templatebank as tbank
import nnetfix.trainingset.trainingset_utils as tsutils
from nnetfix.tools.utils import run_commandline
from nnetfix import mlp
import os 
import datetime
import h5py
import pickle

print(datetime.datetime.now().time())
run_commandline("mkdir {}".format(params.outdir))
run_commandline("mkdir models/{}".format(params.label))
print(params.label)

xml_filename = tbank.generate_tbank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)

txt_filename = tsutils._xml_to_txt(xml_filename,params.label)

n_templates = tsutils.make_hdf5(txt_filename)
tsutils.write_condor_submit_file("generate_trainingset.py",n_templates)
#seg = simulate_single_data_segment(13,7,2)
run_commandline("condor_submit {0}/condor_{1}.sub".format(os.path.abspath('datasets'),params.label))
run_commandline("condor_wait {}/LOG/mainlog.log".format(params.outdir))

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
with open(os.path.join(os.path.abspath('models/{}'.format(params.label)),pkl_filename), 'wb') as file:
     pickle.dump(nnetmodel, file)
     print("model saved successfully")

scaler_filename = "scaler_{}.pkl".format(params.label)
with open(os.path.join(os.path.abspath('models/{}'.format(params.label)),scaler_filename), 'wb') as file:
     pickle.dump(scaler, file)
     print("scaler saved successfully")


NNet_prediction =  mlp.NNetfix(nnetmodel,X_test,y_test)

OriginalData, CutData, PredictData = mlp.reconstruct_testing_set(NNet_prediction, X_test_full, y_test)
print PredictData.shape
print("GREATER SUCCESS")

print(datetime.datetime.now().time())
#run_commandline("./generate_trainingset.py 13")
