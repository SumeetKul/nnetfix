# Runs NNETFIX (Evaluates a saved NNETFIX model) on a series of injections with defined parameters. The parameters of the glitch are the same as the ones used while training the model (check model metadata). 

import numpy as np
import os
import json

from nnetfix import params

# Parameters for Injections:

#n_injections = 50 # Number of injections
n_injections = 15 # Number of injections
#mass1 = (34,36) # Small interval around actual mass value
#mass2 = (28,30)
mass1 = 35
mass2 = 29
snr_range = (15,25)

# Create directory to save data:
base_dir = "simulation_data_output"
data_sub_dir = "tbm_{0}ms_dur_{1}ms".format(int(params.glitch_tbm*1000), int(params.glitch_dur * 1000))
data_dir = os.path.join(base_dir, "injections", params.label, data_sub_dir)
os.makedirs(data_dir)

output_json = {
    "snrs": [],
    "RAs": [],
    "declinations": [],
    "polarizations": [],
    "mass1": mass1,
    "mass2": mass2,
    "snr_range": snr_range,
    "n_injections": n_injections,
}

for i in range(n_injections):
    snr = np.round(np.random.uniform(snr_range[0],snr_range[1]),2)
    output_json["snrs"] += [snr]
    output_json["RAs"] += [np.random.uniform(0,2*np.pi)]
    output_json["declinations"] += [np.random.uniform(-np.pi/2,np.pi/2)]
    output_json["polarizations"] += [np.random.uniform(0,2*np.pi)]

output_filename = os.path.join(data_dir, "snr.json")

with open(output_filename, "w") as outfile:
    json.dump(output_json, outfile)
