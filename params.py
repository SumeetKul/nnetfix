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
mass1_min = 25.0
mass1_max = 45.0

mass2_min = 20.0
mass2_max = 40.0

# Additional Constraints (Optional):
# Mchirp_min =    #chirp mass
# Mchirp_max =    
# MTotal_min =    #total mass
# MTotal_max = 
# Eta_min =       #mass ratio
# Eta_max = 

### 2: Generating the Training set 

apx = "IMRPhenomPv2"
snr_range = (5,45)
minimal_match = 0.99
toa_range = 0.03 #seconds
multiplier = 30
duration = 10
sample_rate = 4096
f_lower = 35
