# NNetfix

A Neural Network to 'fix' Gravitational Wave signals coincident with short-duration glitches in LIGO-Virgo data.

Imagine a GW signal appears at the twin LIGO detectors, H1 and L1. But at the exact same instance, one of the detector (say, H1) sees a burst of short-duration transient noise, or a glitch. 
Depending on how much the glitch affects the data, the event may still get recorded as a coincident trigger. If not, it will be a single detector trigger in L1. In any case, we know the following 
details regarding the situation:

*  The gpstime of the GW event trigger and status of the detector (in particular, which detector has the glitch)
*  The gpstime and the duration of the glitch (Knowing the type of glitch is not required as long as it is a short-duration burst.)
*  Initial search parameters of the trigger. Typically, they are the component masses for a CBC signal. Some composite parameters like the chirp mass can be better estimated (?)

NNETFIX uses the above information to simulate copies of fake data segments in which the glitch-affected part has been gated out, and trains a neural network to *reconstruct* the 
portion of data that was affected. This reconstructed data can be used to generate more accurate skymaps and estimate accurate parameters of the binary.
NNETFIX needs the above three parameters to operate. They are to be input in the params.py file. Note that the binary parameters are to be given as a range around the ones included in the trigger.
For instance, if masses m1 and m2 are triggered in the best-matching template, then include a range of masses around these in the parameters file.



### Installation:
1.  Make a clone of the repository. (You'll need your albert.einstein username and password)
2.  Create a virtual environment in the parent nnetfix directory:
    * Python 2.x: "virtualenv nnetfix-pyenv"
    * Python 3.x: "python3 -m venv nnetfix-py3env"
    NOTE: Use Python 2 at present since pycbc is not yet fully compatible with Python 3.
3.  Install all the required dependencies using "pip (python 2.x) / pip3 (python3.x) install -r requirements.txt"
4.  Setup: Run "python setup.py install"

### Steps to run the code:
1.  Edit the params.py configuration file, add an event label (which can correspond to a real GW* event). NNETFIX needs the following three parameters that define the particular scenario in question:
    * The gpstime of the GW trigger.
    * The masses corresponding to the best-matching template. In the params file, include a range of masses in the neighborhood of this. They will be used to generate a templatebank which forms the parameter space for generating waveforms in the training set.
    * The gpstime defining the start time of when the glitch occurs, and the length of the glitch in seconds.
    Apart from this, one can edit several optional parameters which include the approximant used in generating waveforms, Minimal match of the templatebank, Number of independent noise realizations for each template, the noise PSD used in simulating data, the roll-off parameter of the windowinf function used for gating and so on.
2.  Run 'main.py'. This should perform the following steps: 
    (i) Create a templatebank which will be used to generate the trainingset. This is stored as an xml and txt file in the directory with the given label name.
    (ii) Generate a trainingset which is saved as an hdf5 file the datasets directory. 
    (iii) Process the data and Train a multi-layered perceptron neural network on the trainingset.
    (iv) Save the model using pickle for later use.
    (v) [Under Development] use the model to reconstruct the glitch-affected portion of a real GW event if provided in the label.
3.  To test the model on injections, open 'injection_stats.py' and add the desired injection parameters. This will evaluate the model performance on the injections based on metrics such as reconstruction of SNR, the chi-squared test etc.

