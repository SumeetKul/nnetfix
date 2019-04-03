import numpy as np
from gwpy.timeseries import TimeSeries
from nnetfix import params
#PyCBC:
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.noise.reproduceable import noise_from_string
from pycbc.filter import sigma, resample_to_delta_t, highpass, lowpass_fir, notch_fir, highpass_fir
from pycbc.frame import read_frame, write_frame
from pycbc.filter import matched_filter
import pycbc.vetoes

def calculate_snr(timeseries, m1, m2, trigger_time = params.gpstime, apx = params.apx, dur = params.duration, sample_rate = params.sample_rate, f_low = params.f_lower):

	"""
	Calculates the peak snr value and the time corresponding to the peak snr. Returns the SNR time series.
	"""

	# highpass the timeseries:
	strain = highpass(timeseries,f_low)

	# Crop ends to remove effects of the bandpass filter:
        conditioned = strain.crop(1,1)  
		
	# Calculate PSD:
	psd = conditioned.psd(2)
	psd = interpolate(psd, conditioned.delta_f)
	
	psd = inverse_spectrum_truncation(psd, 2 * conditioned.sample_rate,
                                  low_frequency_cutoff=f_low)
	
	# Generate the template for matched filtering:
	hp, _ = get_td_waveform(approximant=apx,
                     mass1=m1,
                     mass2=m2,
                     delta_t=strain.delta_t,
                     f_lower=f_low)


        hp.resize(len(conditioned))
	template = hp.cyclic_time_shift(hp.start_time)
	# Calculate snr time series:
	snr = matched_filter(template, conditioned, psd=psd, low_frequency_cutoff=f_low)
	snr_ts = TimeSeries.from_pycbc(snr).abs()

	# Calculate the peak snr value:
	peak_snr = snr_ts.max(axis=0)

	# Calculate the location of peak snr:
	#peak_loc = (snr_ts.times[(np.where(np.array(snr_ts) == np.amax(np.array(snr_ts),axis=0)))[-1]])[-1]

	return snr_ts, peak_snr



def calculate_chisq(timeseries, m1, m2, f_low = params.f_lower, sample_rate = params.sample_rate):

	"""
	Performs the chi-squared test to match likeness of the data to the template. Returns peak (lowest) chi_squared value and chi_squared time series.
	"""

	num_bins = 20

	# highpass the timeseries:
        strain = timeseries.highpass(20.0)

        # Crop ends to remove effects of the bandpass filter:
        strain = strain.crop(*strain.span.contract(1))
	strain = strain.to_pycbc()

	# aLIGO PSD:
	psd = pycbc.psd.aLIGOZeroDetLowPower(dur * sample_rate / 2 + 1, 1.0/dur, 5)
	
	# Generate the template for matched filtering:
        hp, _ = get_fd_waveform(approximant=apx, mass1=35, mass2=29,
                        f_lower=15, f_final=2048, delta_f=psd.df.value)
	
	# Calculate chi_squared time series:
	chisq = pycbc.vetoes.power_chisq(template, strain, num_bins, psd=psd, low_frequency_cutoff=f_low)
	
	# convert to a reduced chisq
	chisq /= (num_bins * 2) - 2	
	return chisq
