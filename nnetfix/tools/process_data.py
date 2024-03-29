##### This module includes tools to process real GW d`ata to turn it into a format NNETFIX can work with. This includes bandpassing, cleaning spectral lines and cropping to NNETFIX's default length. ##########

from __future__ import division
import numpy as np
import sys
#GWPy
from gwpy.timeseries import TimeSeries
from gwpy.frequencyseries import FrequencySeries
from gwpy.signal import filter_design
#PyCBC
from pycbc.waveform import get_td_waveform, get_fd_waveform
import pycbc.psd
from pycbc.noise.reproduceable import noise_from_string
from pycbc.filter import sigma, resample_to_delta_t, highpass, lowpass_fir, notch_fir, highpass_fir
from pycbc.frame import read_frame, write_frame
from pycbc.psd import welch, interpolate
from pycbc.types.timeseries import TimeSeries as PycbcTimeSeries

from nnetfix import params
from ligotimegps import LIGOTimeGPS
from scipy.interpolate import interp1d

# GPStime of the merger:
gpstime = params.gpstime

# Dictionary to record data:
spec_lines = dict()

# O1 Spectral lines: (O2 has clean data available through GWOSC. Add tag 'CLN')
spec_lines['L1_lines'] = [33.7,34.7,35.3,60,120,180,307.3,307.5,315.1,333.3,612.5,615.]
spec_lines['H1_lines'] = [35.9,36.7,37.3,60,120,180,299.6,299.4,300.5,300.,302.,302.22,303.31,331.9,504.0,508.5,599.14,599.42,612.5]

def load_data(IFO, tag=params.tag, gpstime=params.gpstime, sample_rate = params.sample_rate, return_whiten_psd = False):  # In future: Add parser for event name.

    """
    Loads and whitens 30s. of data including the event corresponding to the given gpstime.
    """

    #GWdata = TimeSeries.fetch_open_data(IFO, gpstime - 20,  gpstime + 10, sample_rate=sample_rate)
    if params.open_data:
        data = TimeSeries.fetch_open_data('{}'.format(IFO), gpstime - 20, gpstime + 10)
    else:
        data = TimeSeries.get('{}:GDS-CALIB_STRAIN'.format(IFO), gpstime - 20, gpstime + 10)
    #data = TimeSeries.fetch_open_data('{}'.format(IFO), gpstime - 20, gpstime + 10)
    GWdata = data.resample(sample_rate)
    GWdata = GWdata.to_pycbc()
    
    # Calculate the noise spectrum
    # psd = interpolate(welch(GWdata), 1.0 / GWdata.duration)

    # whiten
    # white_strain = (GWdata.to_frequencyseries() / psd ** 0.5).to_timeseries()
    white_strain, whiten_psd = GWdata.whiten(2,2, return_psd = True)
    
    crop_strain = white_strain.crop(2,2)

    # crop_strain = highpass(crop_strain,params.f_lower)
    # crop_strain = lowpass_fir(crop_strain, 800, 512)
    GW_whit_strain = TimeSeries.from_pycbc(crop_strain)
    if return_whiten_psd:
        return GW_whit_strain, whiten_psd
    else:
        return GW_whit_strain

def whiten_data(pycbc_timeseries, crop_length = 2):
    # Calculate the noise spectrum (From the SAME segment)

    psd = interpolate(welch(pycbc_timeseries), 1.0 / pycbc_timeseries.duration)

    # whiten the data using this psd
    white_strain = (pycbc_timeseries.to_frequencyseries() / psd ** 0.5).to_timeseries()

    crop_strain = white_strain.crop(crop_length,crop_length) #crop length in seconds

    return crop_strain 


def recolor_np_array(np_timeseries, pycbc_psd, sample_rate, recolor_option = "gwpy"):
    if recolor_option.lower() == 'gwpy':
        gwpy_timeseries = TimeSeries(np_timeseries, sample_rate = sample_rate)
        recolored_timeseries = recolor_gwpy_timeseries(gwpy_timeseries, pycbc_psd)
    elif recolor_option.lower() == 'pycbc':
        delta_t = 1/sample_rate
        pycbc_timeseries = PycbcTimeSeries(np_timeseries, delta_t = delta_t)
        recolored_timeseries = recolor_pycbc_timeseries(pycbc_timeseries, pycbc_psd)
    return recolored_timeseries

def recolor_gwpy_timeseries(whitened_gwpy_timeseries, pycbc_psd):
    """This method reproduced from code from Kentaro Mogushi with permission"""

    # Make PSD
    whitened_fft = whitened_gwpy_timeseries.fft()

    # interpolate psd
    psd_interpolant = interp1d(pycbc_psd.get_sample_frequencies().data, pycbc_psd.data)
    new_psd = psd_interpolant(whitened_fft.frequencies.value)

    # recolor psd
    recolored_fft = whitened_fft * np.sqrt(new_psd)

    # generate recolored time series
    recolored_timeseries = recolored_fft.ifft()
    return recolored_timeseries

def recolor_pycbc_timeseries(whitened_pycbc_timeseries, pycbc_psd):
    """This method reproduced from code from Kentaro Mogushi with permission"""
    
    # Make PSD
    #whitened_fft = whitened_pycbc_timeseries.fft()
    whitened_fft_pycbc = whitened_pycbc_timeseries.to_frequencyseries()

    # interpolate psd
    psd_interpolant = interp1d(pycbc_psd.get_sample_frequencies().data, pycbc_psd.data)
    new_psd = psd_interpolant(whitened_fft_pycbc.frequencies.value)

    # recolor psd
    recolored_fft = whitened_fft_pycbc * np.sqrt(new_psd)

    # generate recolored time series
    recolored_timeseries = recolored_fft.to_timeseries()
    return recolored_timeseries

#
#
#def clean(timeseries, spec_lines, tag = params.tag, f_low=params.f_lower, f_high=params.f_high):
#
#    """
#    Cleans data by removing spectral lines; bandpasses the data segment.
#    """
#
#    if tag == 'C00':
#
#        bp = filter_design.bandpass(f_low, f_high, 4096.)
##        notches = [filter_design.notch(f, 4096.) for f in spec_lines]
#        zpk = filter_design.concatenate_zpks(bp, *notches)
#
#        clean_timeseries = timeseries.filter(zpk, filtfilt=True)
#
#    elif tag == 'CLN':
#
#        bp_timeseries = timeseries.notch(60).bandpass(f_low,f_high)
#        clean_timeseries = bp_timeseries
#
#    return clean_timeseries
#



def crop_for_nnetfix(timeseries, gpstime =  params.gpstime, sample_rate = params.sample_rate):

    """
    Crops the data into a 10-sec. segment containing the signal that NNETFIX can work on to reconstruct.
    """
    strain_ts = timeseries.to_pycbc()

    #strain_ts = highpass(strain_ts,params.f_lower)
    #strain_ts = lowpass_fir(strain_ts,800,512)

    TOA = gpstime 

    start_time = strain_ts.start_time

    sample_trig_time = float(LIGOTimeGPS(TOA - start_time)) 

    start = int(np.rint(sample_trig_time*sample_rate)) - int(7*sample_rate)
    end = int(np.rint(sample_trig_time*sample_rate)) + int(3*sample_rate)

    inj_segment = strain_ts[start:end]

    return inj_segment, start, end

    
def rejoin_frame(frame_array, raw_timeseries, start, end):
    
    
    filled_timeseries = raw_timeseries.copy()
    filled_timeseries[start:end] = frame_array

    return filled_timeseries


