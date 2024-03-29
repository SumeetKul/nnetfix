#!/usr/bin/env python

import numpy as np
import h5py
import sys,string
from nnetfix import params
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

#def simulate_single_data_segment(m1,m2,index, IFO = params.IFO, apx = params.apx, f_lower = params.f_lower, dur = params.duration, snr_range = params.snr_range, sample_rate = params.sample_rate):
def simulate_single_data_segment(m1, m2, index, IFO, apx, f_lower, dur, snr_range, sample_rate, noise_multiplier, trigger_time, toa_error):
    
    #waveform_arr = np.zeros((params.multiplier,int(sample_rate*dur)))
    waveform_arr = np.zeros((noise_multiplier,int(sample_rate*dur)))

    data_duration = dur + 1
#    for i in range(params.multiplier):

#    detector = Detector('{}'.format(IFO))
#    coa_phase = np.random.uniform(-np.pi/2,np.pi/2)

#    hp, hc = get_td_waveform(approximant=apx,
#             mass1=m1,
#             mass2=m2,
#             coa_phase=coa_phase,
#             delta_t=1.0/sample_rate,
#             f_lower=f_lower)

#    hp.start_time += end_time
#    hc.start_time += end_time

    #for i in range(params.multiplier):
    for i in range(noise_multiplier):

        if m1 == 0.0 and m2 == 0.0:

            # Generate noise from the aLIGO PSD:
            psd = pycbc.psd.aLIGOZeroDetLowPower(data_duration * int(sample_rate)  + 1, 1.0/data_duration, data_duration)

            ts = noise_from_string("aLIGOZeroDetLowPower", 0, data_duration, seed=np.random.randint(20000,50000), low_frequency_cutoff=15)
            ts = resample_to_delta_t(ts, 1.0/sample_rate)
            
            # Whiten and bandpass:
            ts = ts.whiten(1,1)
            #ts = highpass(ts, params.f_lower)
            ts = highpass(ts, f_lower)
            #ts = lowpass_fir(ts,800,512)
            waveform_arr[i] = ts

        else:
            detector = Detector('{}'.format(IFO))
            #coa_phase = np.random.uniform(-np.pi/2,np.pi/2)
            coa_phase = np.random.uniform(0,2*np.pi)

            hp, hc = get_td_waveform(approximant=apx,
                     mass1=m1,
                     mass2=m2,
                     coa_phase=coa_phase,
                     delta_t=1.0/sample_rate,
                     f_lower=f_lower)

            #end_time = params.gpstime + 3.5
            end_time = trigger_time + 3.5

            hp.start_time += end_time
            hc.start_time += end_time

        
            snr = np.random.randint(snr_range[0],snr_range[1])
            #toa = np.around(np.random.uniform(7.7-params.toa_err, 7.7+params.toa_err),3)
            toa = np.around(np.random.uniform(7.7-toa_error, 7.7+toa_error),3)

            #declination = np.random.uniform(-np.pi/2,np.pi/2)
            right_ascension = np.random.uniform(0,2*np.pi)
            polarization = np.random.uniform(0,2*np.pi)

            cos_dec = np.random.uniform(-1, 1)
            dec = np.arccos(cos_dec)
            if dec == np.pi:
                dec = -np.pi/2
            if dec > np.pi/2:
                dec -= np.pi
            declination = dec

 
            signal = detector.project_wave(hp, hc, right_ascension, declination, polarization)
            # Prepend zeros to make the total duration equal to the defined duration:
            signal.prepend_zeros(int(signal.sample_rate*(data_duration-signal.duration)))

            # Add noise:
            psd = pycbc.psd.aLIGOZeroDetLowPower(data_duration * sample_rate + 1, 1.0/data_duration, f_lower)

            ts = noise_from_string("aLIGOZeroDetLowPower", 0, data_duration, seed=index, low_frequency_cutoff=15)
            ts = resample_to_delta_t(ts, 1.0/sample_rate)
            #print ts.duration
            ts.start_time = end_time - data_duration
            
            # The data segment = Signal + Noise; add first in the frequency domain:

            signal = signal.to_frequencyseries()  # Signal in frequency domain
            fs = ts.to_frequencyseries()          # Time in frequency domain

            sig = pycbc.filter.sigma(signal,psd=psd, low_frequency_cutoff=f_lower)
            fs += signal.cyclic_time_shift(toa) / sig * snr

            # Convert back into time-domain:
            dataseg = fs.to_timeseries()

            # Whiten and high-pass:
            dataseg = dataseg.whiten(1,1)
            dataseg = highpass(dataseg, f_lower)
            #dataseg = lowpass_fir(dataseg, 800, 512) 
            waveform_arr[i] = dataseg
             
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


#waveform_array = simulate_single_data_segment(mass1,mass2,index)
waveform_array = simulate_single_data_segment(mass1, mass2,index, IFO = params.IFO, apx = params.apx, f_lower = params.f_lower, dur = params.duration, snr_range = params.snr_range, sample_rate = params.sample_rate, noise_multiplier = params.multiplier, trigger_time = params.gpstime, toa_error = params.toa_err)


hdf5_filename = "trainingset_{}.hdf5".format(params.label)
hdf5_file = os.path.join(os.path.abspath("datasets/"), hdf5_filename)
f = h5py.File(hdf5_file,'a')

keys = f.keys()



f[keys[0]][index*params.multiplier:(index*params.multiplier)+params.multiplier] = waveform_array

print ("Waveforms {} to {}  loaded.".format(str(index*params.multiplier),str((index*params.multiplier)+params.multiplier)))

f.close()

                                                                               
