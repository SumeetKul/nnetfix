#!/usr/bin/env python

import numpy as np
import h5py
import sys,string
import params
import os
# PyCBC
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.psd
from pycbc.detector import Detector
from pycbc.noise.reproduceable import noise_from_string
from pycbc.filter import sigma, resample_to_delta_t, highpass, lowpass_fir
from pycbc.frame import write_frame


###########################################################################################################################################################################
###################################### Define function to simulate data ###################################################################################################

def simulate_single_data_segment(m1,m2,index, end_time = params.gpstime, IFO = params.IFO, apx = params.apx, f_lower = params.f_lower, dur = params.duration, snr_range = params.snr_range, sample_rate = params.sample_rate):
    
    waveform_arr = np.zeros((params.multiplier,int(sample_rate*dur)))
#    for i in range(params.multiplier):

    detector = Detector('{}'.format(IFO))
    coa_phase = np.random.uniform(-np.pi/2,np.pi/2)

    hp, hc = get_td_waveform(approximant=apx,
			 mass1=m1,
			 mass2=m2,
			 coa_phase=coa_phase,
			 delta_t=1.0/sample_rate,
			 f_lower=f_lower)

    hp.start_time += end_time
    hc.start_time += end_time

    for i in range(params.multiplier):

	    snr = np.random.randint(snr_range[0],snr_range[1])
	    toa = np.around(np.random.uniform(6.98,7.02),3)

	    declination = np.random.uniform(-np.pi/2,np.pi/2)
	    right_ascension = np.random.uniform(0,2*np.pi)
	    polarization = np.random.uniform(0,2*np.pi)

	  
	    signal = detector.project_wave(hp, hc, right_ascension, declination, polarization)
	    # Prepend zeros to make the total duration equal to the defined duration:
	    signal.prepend_zeros(int(signal.sample_rate*(dur-signal.duration)))

	    # Add noise:
	    psd = pycbc.psd.aLIGOZeroDetLowPower(dur * sample_rate + 1, 1.0/dur, f_lower)

	    ts = noise_from_string("aLIGOZeroDetLowPower", 0, dur, seed=index, low_frequency_cutoff=30)
	    ts = resample_to_delta_t(ts, 1.0/sample_rate)
	    #print ts.duration
	    ts.start_time = end_time - dur
	    
	    # The data segment = Signal + Noise; add first in the frequency domain:

	    signal = signal.to_frequencyseries()  # Signal in frequency domain
	    fs = ts.to_frequencyseries()          # Time in frequency domain

	    sig = pycbc.filter.sigma(signal,psd=psd, low_frequency_cutoff=f_lower)
	    fs += signal.cyclic_time_shift(toa) / sig * snr

            dataseg = fs.timeseries()

	    dataseg = highpass(dataseg, 35)
	    dataseg = lowpass_fir(dataseg,800,512)
	    # Convert back into time domain:
	    waveform_arr[i] = dataseg
            
            print i 
    return waveform_arr


######################################################################################################################################################################################
#################################### Run as script or through Condor #############################################################################################


index = sys.argv[1]
index = string.atoi(index)
#index2 = sys.argv[2]
#index2 = string.atoi(index2)

templatebank_txtfile_name = "templatebank_{}.txt".format(params.label)
templatebank_file = os.path.join(params.outdir,templatebank_txtfile_name)
masses = np.loadtxt(templatebank_file,delimiter=",")


mass1 = masses[index][0]
mass2 = masses[index][1]


waveform_array = simulate_single_data_segment(mass1,mass2,index)


hdf5_filename = "trainingset_{}.hdf5".format(params.label)
hdf5_file = os.path.join(params.outdir, hdf5_filename)
f = h5py.File(hdf5_file,'a')

keys = f.keys()



f[keys[0]][index*params.multiplier:(index*params.multiplier)+params.multiplier] = waveform_array

print ("Waveforms {} to {}  loaded.".format(str(index*params.multiplier),str((index*params.multiplier)+params.multiplier)))

f.close()

                                                                               
