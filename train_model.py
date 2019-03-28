import h5py
import numpy as np
from nnetfix import mlp

f = h5py.File("datasets/trainingset_GW170814.hdf5",'r')

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
