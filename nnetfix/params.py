#####################################################################################################################################################
########################## A universal configuration file defining various parameters used for running NNETFIX #####################################
####################################################################################################################################################

label = "GW150914"
### 1: Trigger Information:
outdir = "examples/GW150914"
# Interferometer which has a glitch. One of 'H1','L1' or 'V1'.
IFO = 'H1'

# GPSTime of the trigger:
gpstime = 1126259462.42

# Mass Ranges:
mass1_min = 30.0
mass1_max = 40.0

mass2_min = 25.0
mass2_max = 35.0

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
toa_range = 0.025 #seconds
multiplier = 25
duration = 10
sample_rate = 4096
f_lower = 35

# Fraction of Trainingset to be comprised of pure noise samples:
noise_fraction = 0.1


### 3: Glitch Information:

# GPSTime of the glitch:
glitch_t = gpstime - 0.08

# Duration of the glitch (seconds):
glitch_dur = 0.04

# Alpha used for the gating:
alpha = 0.1

# Number of pure noise samples to use:
#noise_samples = 2000
