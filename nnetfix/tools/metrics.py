import numpy as np
from gwpy.timeseries import TimeSeries
from nnetfix import params
#PyCBC:
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.psd
from pycbc.noise.reproduceable import noise_from_string
from pycbc.filter import sigma, resample_to_delta_t, highpass, lowpass_fir, notch_fir, highpass_fir
from pycbc.frame import read_frame, write_frame
from pycbc.filter import matched_filter

def calculate_snr(timeseries, m1, m2, realdata = False, trigger_time = params.gpstime, apx = params.apx, dur = params.duration, sample_rate = params.sample_rate, f_low = params.f_lower):

	"""
	Calculates the peak snr value and the time corresponding to the peak snr. Returns the SNR time series.
	"""

	# highpass the timeseries:
	strain = timeseries.highpass(20.0)

	# Crop ends to remove effects of the bandpass filter:
	strain = strain.crop(*strain.span.contract(1))

	# Generate the template for matched filtering:
	hp, _ = get_fd_waveform(approximant=apx, mass1=35, mass2=29,
                        f_lower=15, f_final=2048, delta_f=psd.df.value)


	if realdata:
		psd = strain.psd(8,2)
		psd = psd.to_pycbc()
		zoom = strain.crop(int(trigger_time)-4,int(trigger_time)+4)
		zoom = zoom.to_pycbc()

	else:
		psd = pycbc.psd.aLIGOZeroDetLowPower(dur * sample_rate / 2 , 1.0/dur, 5)
		zoom = strain.to_pycbc()

	# Calculate snr time series:
	snr = matched_filter(hp, zoom, psd=psd, low_frequency_cutoff=f_low)
	snr_ts = TimeSeries.from_pycbc(snr).abs()

	# Calculate the peak snr value:
	peak_snr = snr_ts.max(axis=0)

	# Calculate the location of peak snr:
	peak_loc = (snrts1.times[(np.where(np.array(snrts1) == np.amax(np.array(snrts1),axis=0)))[-1]])[-1]

	return snr_ts, peak_snr, peak_loc
	
