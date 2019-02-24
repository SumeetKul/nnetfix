import numpy as np
import params
from generate_templatebank import generate_templatebank

print(params.label)

generate_templatebank(params.label,params.mass1_min,params.mass1_max,params.mass2_min,params.mass2_max,params.minimal_match)
