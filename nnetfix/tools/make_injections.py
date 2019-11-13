import numpy as np
#PyCBC
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.psd
from pycbc.noise.reproduceable import noise_from_string
from pycbc.filter import sigma, resample_to_delta_t, highpass, lowpass_fir
from pycbc.frame import write_frame
from pycbc.detector import Detector

from gwpy.timeseries import TimeSeries
from nnetfix import params

#def inject_signal(m1, m2, snr, IFO, end_time = params.gpstime, dur = params.duration, sample_rate = params.sample_rate, apx = params.apx, f_lower = params.f_lower):
def inject_signal(m1, m2, snr, IFO, right_ascension, declination, polarization, end_time = params.gpstime, dur = params.duration, sample_rate = params.sample_rate, apx = params.apx, f_lower = params.f_lower):

    """
    Injects a signal into a given interferometer having given component masses using aLIGO coloured noise. The extrinsic parameters, viz. sky localization, phase and polarization            are randomized. The merger time is set at 3.0 seconds before the end of the data segment.
    """
    
    dur += 1

    detector = Detector('{}'.format(IFO))
    coa_phase = np.random.uniform(-np.pi/2,np.pi/2)

    hp, hc = get_td_waveform(approximant=apx,
         mass1=m1,
         mass2=m2,
         coa_phase=coa_phase,
         delta_t=1.0/sample_rate,
         f_lower=f_lower)

    hp.start_time += end_time + 3.5
    hc.start_time += end_time + 3.5

    toa = 7.7

    #declination = np.random.uniform(-np.pi/2,np.pi/2)
    #right_ascension = np.random.uniform(0,2*np.pi)
    #polarization = np.random.uniform(0,2*np.pi)


    signal = detector.project_wave(hp, hc, right_ascension, declination, polarization)
    # Prepend zeros to make the total duration equal to the defined duration:
    signal.prepend_zeros(int(signal.sample_rate*(dur-signal.duration)))

    # Add noise:
    psd = pycbc.psd.aLIGOZeroDetLowPower(dur * int(sample_rate) + 1, 1.0/dur, f_lower)

    ts = noise_from_string("aLIGOZeroDetLowPower", 0, dur, seed=np.random.randint(50000,450000), low_frequency_cutoff=15)
    ts = resample_to_delta_t(ts, 1.0/sample_rate)
    #print ts.duration
    ts.start_time = end_time - dur

    # The data segment = Signal + Noise; add first in the frequency domain:

    signal = signal.to_frequencyseries()  # Signal in frequency domain
    fs = ts.to_frequencyseries()          # Time in frequency domain

    sig = pycbc.filter.sigma(signal,psd=psd, low_frequency_cutoff=f_lower)
    fs += signal.cyclic_time_shift(toa) / sig * snr

    dataseg = fs.to_timeseries()
    
    dataseg = dataseg.whiten(1,1)
    dataseg = highpass(dataseg, params.f_lower)
    #dataseg = lowpass_fir(dataseg,800,512)

    param_list = [right_ascension, declination, polarization, snr]

    return dataseg, param_list



def inject_noise(dur = params.duration, sample_rate = params.sample_rate, trigger_time = params.gpstime):

    """
    Returns a timeseries segment of aLIGO coloured noise of the given duration and sampled at the given rate.
    """
    dur += 1
    # Generate noise from the aLIGO PSD:
    psd = pycbc.psd.aLIGOZeroDetLowPower(dur * int(sample_rate)  + 1, 1.0/dur, dur)

    ts = noise_from_string("aLIGOZeroDetLowPower", 0, dur, seed=np.random.randint(10000), low_frequency_cutoff=10)
    ts = resample_to_delta_t(ts, 1.0/sample_rate)
    #print ts.duration
    ts.start_time = trigger_time - 7.7
    ts = ts.whiten(1,1)
    ts = highpass(ts,params.f_lower)
    noise_seg = ts
    ts1 = highpass(ts, 35)
    #noise_seg = lowpass_fir(ts1,800,512)

    return noise_seg
