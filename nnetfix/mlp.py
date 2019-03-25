import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


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

