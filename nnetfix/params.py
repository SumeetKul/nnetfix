#####################################################################################################################################################
########################## A universal configuration file defining various parameters used for running NNETFIX #####################################
####################################################################################################################################################

label = "GW170608"
### 1: Trigger Information:
outdir = "test"
# Interferometer which has a glitch. One of 'H1','L1' or 'V1'.
IFO = 'H1'

# GPSTime of the trigger:
gpstime = 1180922494.49

tag = "CLN"
# Mass Ranges:
mass1_min = 10.0
mass1_max = 15.0

mass2_min = 5.0
mass2_max = 10.0

# Additional Constraints (Optional):
# Mchirp_min =    #chirp mass
# Mchirp_max =    
MTotal_min = 16   #total mass
MTotal_max = 23
#eta_min = 0.25       #mass ratio
#eta_max = 0.30

### 2: Generating the Training set 

apx = "IMRPhenomPv2"
snr_range = (8,45)
minimal_match = 0.97
toa_range = 0.01 #seconds
multiplier = 25
duration = 10
sample_rate = 4096
f_lower = 35
f_high = 500
# Fraction of Trainingset to be comprised of pure noise samples:
noise_fraction = 0.1


### 3: Glitch Information:

# Time before merger: (in sec., defined as duration between the END time of the gating and the trigger time)
glitch_tbm = 0.05

# Duration of the glitch (seconds):
glitch_dur = 0.125

# GPSTime of the glitch: (Currently defined as the START point of the gating we have to do due to the glitch)
glitch_t = gpstime - glitch_tbm - glitch_dur


# Alpha used for the gating:
alpha = 0.1

# Number of pure noise samples to use:
#noise_samples = 2000
