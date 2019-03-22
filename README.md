# NNetfix

A Neural Network to 'fix' Gravitational Wave signals coincident with short-duration glitches in LIGO-Virgo data.

Imagine a GW signal appears at the twin LIGO detectors, H1 and L1. But at the exact same instance, one of the detector (say, H1) sees a burst of short-duration transient noise, or a glitch. 
Depending on how much the glitch affects the data, the event may still get recorded as a coincident trigger. If not, it will be a single detector trigger in L1. In any case, we know the following 
details regarding the situation:

*  The gpstime of the GW event trigger and status of the detector (in particular, which detector has the glitch)
*  The gpstime and the duration of the glitch (Knowing the type of glitch is not required as long as it is a short-duration burst.)
*  Initial search parameters of the trigger. Typically, they are the component masses for a CBC signal. Some composite parameters like the chirp mass can be better estimated (?)

NNETFIX needs the above three parameters to operate. They are to be input in the params.py file. Note that the binary parameters are to be given as a range around the ones included in the trigger.
For instance, if masses m1 and m2 are triggered in the best-matching template, then include a range of masses around these in the parameters file.

NNETFIX then generates a templatebank of templates taken in the neighbourhood of the trigger template, with a defined minimal match.

These templates are used to simulate data mimicking data from the affected interferometer (here, H1) with the glitch gated out. 


### Installation:
1.  Make a clone of the repository. (You'll need your albert.einstein username and password)
2.  Create a virtual environment in the parent nnetfix directory:
    * Python 2.x: "virtualenv nnetfix-pyenv"
    * Python 3.x: "python3 -m venv nnetfix-py3env"
3.  Install all the required dependencies using "pip (python 2.x) / pip3 (python3.x) install -r requirements.txt"
4.  Setup: Run "python setup.py install"

### Steps to run the code (in its present state):
1.  Edit the params.py configuration file to add the event label, IFO and gpstime, mass ranges and other parameters to simulate data.
2.  Run 'main.py'. This should generate a trainingset saved as an hdf5 file stored in the datasets directory. Additionally, the templatebank used to generate the trainingset will be stored as an xml and txt file in the directory named as the given label.
3.  
