#######################################################################################################################################################################################
######################## This module contains all functions required for training the Multi-Layer Perceptron (MLP) regressor Neural Network that forms the heart of NNetfix ##################################################################################################################################################################################################


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import scipy.signal
import params

def scale_data(TrainingData):

    """
     Pre-process the trainingset: This includes scaling the entire dataset using a pre-defined scikit-learn Scaler. For this application, we use a standard scaler which scales the features (here the h(t) sample points) features removing the mean and scaling to unit variance.
    

    From the scikit-learn StandardScaler docstring:
        
        "Centering and scaling happen independently on each feature by computing
        the relevant statistics on the samples in the training set. Mean and
        standard deviation are then stored to be used on later data using the
        `transform` method.

        Standardization of a dataset is a common requirement for many
        machine learning estimators: they might behave badly if the
        individual feature do not more or less look like standard normally
        distributed data (e.g. Gaussian with 0 mean and unit variance)."

    The same scaler object will then be used to scale the real data frames / Injection frames that we will apply NNetfix to.

    Input: 
    TrainingData: A numpy array with individual rows corresponding to sample waveforms used for training.

    Output: 
    A scaled dataset array with the same dimensions.
    """

    scaler = StandardScaler()
    ML_data = scaler.fit_transform(TrainingData)

    return ML_data, scaler


def _get_glitch_parameters(TrainingData, glitch_dur = params.glitch_dur, sample_rate = params.sample_rate):
    """ Keeps a record of the global parameters used for defining the glitch-gating in the data arrays (Derived from the params file and dimensions of the Trainingset) """

    glitch_params = dict()
    
    glitch_params['glitch_dur'] = int(glitch_dur * sample_rate)
    glitch_params['n_samples'] = n_samples = TrainingData.shape[-1] # Number of sample points
    glitch_params['tg'] = None

    return glitch_params



def prepare_X_data(TrainingData,glitch_params):
    
    """ Prepares the samples used as the X-trainingset. These sample waveforms have a gating at the place where the glitch occurs. The size, duration, time of the gating is defined according to the parameters file. The alpha-roll off of the Tukey window used for gating is also defined in params.py 

    """

    tuck = scipy.signal.tukey(glitch_params['glitch_dur'],alpha=alpha)
    gate_y = np.pad(tuck,(tg,(glitch_params['n_samples'] - tg - glitch_params['glitch_dur'])),'constant')
    gate = 1.0-gate_y

   
    X_template = gate*TrainingData


    return X_template


