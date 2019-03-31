import h5py
import numpy as np
from nnetfix import mlp, params
import pickle
import os 
import datetime

print(datetime.datetime.now().time())
f = h5py.File(os.path.join(os.path.abspath("datasets/"),"trainingset_{}.hdf5".format(params.label)),'r')

keys = f.keys()
Training_dataset = f[keys[0]][:]

ML_data, scaler = mlp.scale_data(Training_dataset)

f.close()

Xdata, n_samples = mlp.prepare_X_data(ML_data)

print("X_data loaded")
print Xdata

y_glitch = mlp.prepare_Y_data(ML_data)
print("y-data created")
print y_glitch

X_train, X_test, X_train_full, X_test_full, y_train, y_test = mlp.split_trainingset(Xdata, y_glitch)

print("data split")

nnetmodel = mlp.NNetfit(X_train, y_train)

print("model trained. GREAT SUCCESS!")

# # SAVE ML_model using pickle

pkl_filename = "model_{}.pkl".format(params.label)
with open(os.path.join(params.outdir,pkl_filename), 'wb') as file:  
     pickle.dump(nnetmodel, file)
     print("model saved successfully")

scaler_filename = "scaler_{}.pkl".format(params.label)
with open(os.path.join(params.outdir,scaler_filename), 'wb') as file:
     pickle.dump(scaler, file)
     print("scaler saved successfully")


NNet_prediction =  mlp.NNetfix(nnetmodel,X_test,y_test)

OriginalData, CutData, PredictData = mlp.reconstruct_testing_set(NNet_prediction, X_test_full, y_test)
print PredictData.shape
print("GREATER SUCCESS")
print(datetime.datetime.now().time())

