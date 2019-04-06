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
import pycbc.types

def calculate_snr(timeseries, m1, m2, apx = params.apx, dur = params.duration, sample_rate = params.sample_rate, f_low = params.f_lower):

	"""
	Calculates the peak snr value and the time corresponding to the peak snr. Returns the SNR time series.
	"""

	if type(timeseries) != pycbc.types.timeseries.TimeSeries:
		strain = TimeSeries(timeseries, sample_rate = sample_rate)
		timeseries = strain.to_pycbc()
	# highpass the timeseries:
	strain = highpass(timeseries,f_low)

	# Crop ends to remove effects of the bandpass filter:
        conditioned = strain.crop(2,2)  
		
	# Calculate PSD:
	psd = conditioned.psd(4)
	psd = interpolate(psd, conditioned.delta_f)
	
	psd = inverse_spectrum_truncation(psd, 4 * conditioned.sample_rate,
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
	peak_loc = (snr_ts.times[(np.where(np.array(snr_ts) == np.amax(np.array(snr_ts),axis=0)))[-1]])[-1].to_value()

	return snr_ts, peak_snr, peak_loc



def calculate_chisq(timeseries, m1, m2, peak_loc, f_low = params.f_lower, sample_rate = params.sample_rate, trig_time = params.gpstime, apx = params.apx):

	"""
	Performs the chi-squared test to match likeness of the data to the template. Returns peak (lowest) chi_squared value and chi_squared time series.
	"""

	if type(timeseries) != pycbc.types.timeseries.TimeSeries:
                strain = TimeSeries(timeseries, sample_rate = sample_rate)
                timeseries = strain.to_pycbc()
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

	num_bins = 20
	# Calculate chi_squared time series:
	chisq = pycbc.vetoes.power_chisq(template, conditioned, num_bins, psd=psd, low_frequency_cutoff=f_low)
	
	# convert to a reduced chisq
	chisq /= (num_bins * 2) - 2	

	chisq_gwpy = TimeSeries.from_pycbc(chisq)
	# Focus chi_sq timeseries around trigger (peak snr) time:
	chi_focus = chisq_gwpy.crop(peak_loc-0.15,peak_loc+0.15)
	chi_min = chi_focus.min()

	return chi_focus, chi_min
