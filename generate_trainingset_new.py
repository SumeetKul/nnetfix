from __future__ import division

import numpy as np
import h5py
#import sys, string
from nnetfix import params
import os
import sys

#from pycbc.waveform import get_td_waveform, get_td_waveform
from pycbc.waveform import get_td_waveform#, get_td_waveform
import pycbc.psd
from pycbc.detector import Detector
from pycbc.noise.reproduceable import noise_from_string
from pycbc.filter import sigma, resample_to_delta_t, highpass#, lowpass_fir
from pycbc.frame import write_frame

def simulate_single_data_segment(m1, m2, index, ifo, apx, f_lower, dur, snr_range, sample_rate, noise_multiplier, gps_time, toa_error):
    
    waveform_arr = np.zeros((noise_multiplier, int(sample_rate*dur)))

    data_duration = dur + 1

    for i in range(noise_multiplier):

        ts = noise_from_string("aLIGOZeroDetLowPower", 0, data_duration, seed=np.random.randint(20000, 50000), low_frequency_cutoff=15)
        ts = resample_to_delta_t(ts, 1/sample_rate)

        if m1 == 0 and m2 == 0:

            # Generate noise from the aLIGO PSD
            #psd = pycbc.psd.aLIGOZeroDetLowPower()

            #ts = noise_from_string("aLIGOZeroDetLowPower", 0, data_duration, seed=np.random.randint(20000, 50000), low_frequency_cutoff=15)
            #ts = resample_to_delta_t(ts, 1/sample_rate)

            ts = ts.whiten(1, 1)
            ts = highpass(ts, f_lower)
            waveform_arr[i] = ts

        else: 

            detector = Detector(ifo)
            coa_phase = np.random.uniform(0, 2*np.pi)

            hp, hc = get_td_waveform(approximant=apx,
                mass1 = m1,
                mass2 = m2,
                coa_phase = coa_phase,
                delta_t = 1/sample_rate,
                f_lower = f_lower
            )

            end_time = gps_time + 3.5

            hp.start_time += end_time
            hc.start_time += end_time

            snr = np.random.uniform(snr_range[0], snr_range[1])
            toa = np.around(np.random.uniform(7.7 - toa_error, 7.7 + toa_error))

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
            
            signal.prepend_zeros(int(signal.sample_rate*(data_duration - signal.duration)))

            #ts = noise_from_string
            #ts = noise_from_string("aLIGOZeroDetLowPower", 0, data_duration, seed=np.random.randint(20000, 50000), low_frequency_cutoff=15)

            ts.start_time = end_time - data_duration

            signal_f = signal.to_frequencyseries()
            fs = ts.to_frequencyseries()

            psd = pycbc.psd.aLIGOZeroDetLowPower(data_duration * sample_rate + 1, 1.0/data_duration, f_lower)

            sig = pycbc.filter.sigma(signal_f, psd=psd, low_frequency_cutoff = f_lower)

            ratio = snr / sig
            #fs += 
            #low

            fs += ratio * signal_f.cyclic_time_shift(toa)

            ts_data = fs.to_timeseries()

            ts_data = ts_data.whiten(1, 1)
            ts_data = highpass(ts_data, f_lower)

            waveform_arr[i] = ts_data
        
    return waveform_arr

index = sys.argv[1]
index = int(index)

templatebank_txtfile_name = "templatebank_" + params.label + ".txt"
templatebank_file = os.path.join(params.outdir, templatebank_txtfile_name)
masses = np.loadtxt(templatebank_file, delimiter=",")

mass1 = masses[index][0]
mass2 = masses[index][1]

waveform_array = simulate_single_data_segment(mass1, mass2, index, params.IFO, params.apx, params.f_lower, params.duration, params.snr_range, params.sample_rate, params.multiplier, params.gpstime, params.toa_err)

hdf5_filename = "trainingset_" + params.label + ".hdf5"
hdf5_file = os.path.join(os.path.abspath("datasets/"), hdf5_filename)
f = h5py.File(hdf5_file, "a")

keys = f.keys()

start_index = index*params.multiplier
end_index = (index + 1)*params.multiplier
#f[keys[0]][index*params.multiplier:(index + 1)*params.multiplier] = waveform_array
f[keys[0]][start_index:end_index] = waveform_array

print("Waveforms " + str(start_index) + " to " + str(end_index) + " loaded.")

f.close()