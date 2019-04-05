#####################################################################################################################################################
########################## A universal configuration file defining various parameters used for running NNETFIX #####################################
####################################################################################################################################################

label = "GW150914"
### 1: Trigger Information:
outdir = "test"
# Interferometer which has a glitch. One of 'H1','L1' or 'V1'.
IFO = 'H1'

# GPSTime of the trigger:
gpstime = 1126259462.42

# Mass Ranges:
mass1_min = 28.0
mass1_max = 42.0

mass2_min = 23.0
mass2_max = 37.0

# Additional Constraints (Optional):
# Mchirp_min =    #chirp mass
# Mchirp_max =    
#MTotal_min = 50   #total mass
#MTotal_max = 60
#eta_min = 0.25       #mass ratio
#eta_max = 0.30

### 2: Generating the Training set 

apx = "IMRPhenomPv2"
snr_range = (8,45)
minimal_match = 0.99
toa_range = 0.01 #seconds
multiplier = 50
duration = 10
sample_rate = 4096
f_lower = 30

# Fraction of Trainingset to be comprised of pure noise samples:
noise_fraction = 0.1


### 3: Glitch Information:

# Time before merger: (in sec., defined as duration between the END time of the gating and the trigger time)
glitch_tbm = 0.02

# Duration of the glitch (seconds):
glitch_dur = 0.06

# GPSTime of the glitch: (Currently defined as the START point of the gating we have to do due to the glitch)
glitch_t = gpstime - glitch_tbm - glitch_dur


# Alpha used for the gating:
alpha = 0.1

# Number of pure noise samples to use:
#noise_samples = 2000
