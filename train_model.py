import numpy as np
#from nnetfix import params
import nnetfix.trainingset.templatebank as tbank
import nnetfix.trainingset.trainingset_utils as tsutils
from nnetfix.tools.utils import run_commandline
from nnetfix.tools import metrics, process_data
from nnetfix import mlp
import os 
import datetime
import h5py
import pickle

import numpy as np
from nnetfix import params
from nnetfix import mlp
from nnetfix.tools import metrics, utils



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
#
scaler_filename = "scaler_{}.pkl".format(params.label)
with open(os.path.join(os.path.abspath('models/{}'.format(params.label)),scaler_filename), 'wb') as file:
     pickle.dump(scaler, file)
     print("scaler saved successfully")



