#######################################################################################################################################################################################
######################## This module contains all functions required for training the Multi-Layer Perceptron (MLP) regressor Neural Network that forms the heart of NNetfix ##################################################################################################################################################################################################


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import scipy.signal
import params
import os



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

glitch_params = dict()
glitch_params['glitch_dur'] = int(params.glitch_dur * params.sample_rate)
glitch_params['tg'] = int(7.0*params.sample_rate) - int((params.gpstime-params.glitch_t)*params.sample_rate)
#glitch_params['glitch_len'] = 


#def _get_glitch_parameters(TrainingData, glitch_dur = params.glitch_dur, sample_rate = params.sample_rate, glitch_t = params.glitch_time):
#    """ Keeps a record of the global parameters used for defining the glitch-gating in the data arrays (Derived from the params file and dimensions of the Trainingset) """
#
#    glitch_params = dict()
#    
#    glitch_params['glitch_dur'] = int(glitch_dur * sample_rate)
#    glitch_params['n_samples'] = TrainingData.shape[-1] # Number of sample points
#    glitch_params['glitch_t'] = glitch_t
#
#    return glitch_params



def prepare_X_data(TrainingData, tg=glitch_params['tg'] , glitch_dur = glitch_params['glitch_dur'], alpha=params.alpha):
    
    """ Prepares the samples used as the X-trainingset. These sample waveforms have a gating at the place where the glitch occurs. The size, duration, time of the gating is defined according to the parameters file. The alpha-roll off of the Tukey window used for gating is also defined in params.py 

    """
    
    n_samples = TrainingData.shape[-1]

    tuck = scipy.signal.tukey(int(params.glitch_dur*params.sample_rate),alpha=alpha)
    gate_y = np.pad(tuck,(tg,(n_samples - tg - glitch_dur)),'constant')
    gate = 1.0-gate_y

   
    X_data = gate*TrainingData


    return X_data, n_samples




def prepare_Y_data(TrainingData, tg = glitch_params['tg'], glitch_dur = glitch_params['glitch_dur'], alpha = params.alpha):

    """ Prepares the samples used as the Y-(or prediction)-trainingset. These represent the data that's supposed to be in the gated portion i.e. the actual part of the signal that is affected by the glitch. The number of sample points for this should be equal to the size of the glitch times the sample rate.
    """
    
    n_samples = TrainingData.shape[-1]
    tuck = scipy.signal.tukey(int(params.glitch_dur*params.sample_rate),alpha=alpha)
    y_glitch = TrainingData[:, tg:(tg+glitch_dur)]
    y_glitch = tuck*y_glitch

    y_glitch[-int(params.noise_fraction*n_samples):] = np.zeros(glitch_dur)
    print(int(params.noise_fraction*n_samples))
    return y_glitch



def split_trainingset(X_data, y_glitch, split_fraction=0.3, sample_rate = params.sample_rate):

    """ 
    This function performs 2 operations:
        (a) Splits the trainingsets into training and testing set based on the given fraction. Defaults to a 0.7-0.3 split.
        (b) Picks out an window of time around the trigger to narrow the size of the trainingset samples. ** Currently this is hard-coded as 2 seconds, which covers most of the BBH           signals over 30 Hz.     
    """

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_data,y_glitch,test_size = split_fraction)

    start_cut_dur = 6.0 #second(s)
    end_cut_dur = 2.5 # second(s)

    X_train = X_train_full[:,int(start_cut_dur*sample_rate):-int(end_cut_dur*sample_rate)]
    X_test = X_test_full[:,int(start_cut_dur*sample_rate):-int(end_cut_dur*sample_rate)]
    y_train = y_train_full
    y_test = y_test_full

    return X_train, X_test, X_train_full, X_test_full, y_train, y_test




def NNetfit(X_train,y_train,hidden_layer_sizes=(200,)):

    """
    """

    nnetfix_model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,verbose=True)
    nnetfix_model.fit(X_train,y_train)

    print(nnetfix_model.score(X_train,y_train))

    return nnetfix_model



def NNetfix(nnetfix_model, X_test, y_test):

    """
    The M.O.A.F.
    """
    print(nnetfix_model.score(X_test,y_test))
    NNet_prediction = nnetfix_model.predict(X_test)


    return NNet_prediction



def reconstruct_testing_set(NNet_prediction, X_test_full, y_test, tg = glitch_params['tg'], glitch_dur = glitch_params['glitch_dur'], alpha=params.alpha):

    """
    """

    invgate = scipy.signal.tukey(glitch_dur,alpha=alpha)
    test_prediction = invgate*NNet_prediction

    # ### 'Fill in the gaps' of the X-data:
    PredictX = np.copy(X_test_full)
    PredictX[:,tg:(tg+glitch_dur)] = PredictX[:, tg:(tg+glitch_dur)] + test_prediction
    PredictData = PredictX
    #PredictData = predict_Lreg 
  
    # # ### FOR REFERENCE: Recover the actual data segments from the testing set:
    ActualX = np.copy(X_test_full)
    ActualX[:,tg:(tg+glitch_dur)] = ActualX[:,tg:(tg+glitch_dur)]+(invgate*y_test)
    OriginalData = ActualX

    ### Finally, we create an array of the cut (X) data set:
    CutData = X_test_full 

    return OriginalData, CutData, PredictData




def process_dataframe(data_array, scaler, n_samples, tg = glitch_params['tg'], glitch_dur = glitch_params['glitch_dur'], alpha = params.alpha):

    """
    """

    data_array = np.load(os.path.join(outdir,frame_name))
    
    data_array_reshape = data_array.reshape(1,n_samples)
    data_array_transform = scaler.transform(data_array_reshape)

    tuck = scipy.signal.tukey(glitch_dur,alpha=alpha)
    gate_y = np.pad(tuck,(tg,(n_samples - tg - glitch_dur)),'constant')
    gate = 1.0-gate_y

    #ML_data_glitch = np.copy(ML_data)
    testdata_glitch = gate*testdata_transform

    X_testdata_full = testdata_glitch
    # Check of the padding has been done right:
    X_testdata_full.shape

    ML_testdata_y = np.copy(testdata_transform)
    y_testglitch_full = ML_testdata_y[:,tg:tg+glitch_dur]
    y_testglitch_full = tuck*y_testglitch_full
    
    start_cut_dur = 6.0 #second(s)
    end_cut_dur = 2.5 # second(s)

    X_testdata = X_testdata_full[:,int(start_cut_dur*sample_rate):-int(end_cut_dur*sample_rate)]

    y_testglitch = y_testglitch_full

    return X_testdata, y_testglitch



def reconstruct_frame(NNet_prediction, X_test_full, y_test, n_samples, tg = glitch_params['tg'], glitch_dur = glitch_params['glitch_dur']):

    """
    """

    invgate = scipy.signal.tukey(glitch_dur,alpha=params.alpha)
    test_prediction = invgate*NNet_prediction

    # ### 'Fill in the gaps' of the X-data:
    PredictX = np.copy(X_test_full)
    PredictX[:,glitch_t:(glitch_t+glitch_dur)] = PredictX[:, glitch_t:(glitch_t+glitch_dur)] + test_prediction
    PredictData = PredictX
    #PredictData = predict_Lreg 

    # # ### FOR REFERENCE: Recover the actual data segments from the testing set:
    ActualX = np.copy(X_test_full)
    ActualX[:,glitch_t:(glitch_t+glitch_dur)] = ActualX[:,glitch_t:(glitch_t+glitch_dur)]+(invgate*y_test)
    OriginalData = ActualX

    ### Finally, we create an array of the cut (X) data set:
    CutData = X_test_full


    gate_y = np.pad(invgate,(glitch_t,(n_samples - glitch_t - glitch_dur)),'constant')
    gate = 1.0-gate_y

    OriginalData = scaler.inverse_transform(testdata_transform.reshape(1,-1))
    OriginalData = OriginalData.reshape(-1)

    CutData = scaler.inverse_transform(CutData.reshape(1,-1))
    CutData = CutData.reshape(-1)
    CutData = gate*CutData

    PredictData = scaler.inverse_transform(PredictData.reshape(1,-1))
    PredictData = PredictData.reshape(-1)


    return OriginalData, CutData, PredictData


