#####################################################################################################################################################
########################## A universal configuration file defining various parameters used for running NNETFIX #####################################
####################################################################################################################################################

label = "GW150914"
### 1: Trigger Information:
outdir = "test"
# Interferometer which has a glitch. One of 'H1','L1' or 'V1'.
IFO = 'H1'

# GPSTime of the trigger:
gpstime = 1186741861.5 

# Mass Ranges:
mass1_min = 35.0
mass1_max = 40.0

mass2_min = 25.0
mass2_max = 30.0

# Additional Constraints (Optional):
# Mchirp_min =    #chirp mass
# Mchirp_max =    
MTotal_min = 50   #total mass
MTotal_max = 60
#eta_min = 0.25       #mass ratio
#eta_max = 0.30

### 2: Generating the Training set 

apx = "IMRPhenomPv2"
snr_range = (5,45)
minimal_match = 0.99
toa_range = 0.03 #seconds
multiplier = 30
duration = 10
sample_rate = 4096
f_lower = 35
