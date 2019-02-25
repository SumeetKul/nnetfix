import numpy as np
#from .. import params 
#from ..core import utils
import subprocess
#import logger
import os
import params
from core.utils import run_commandline
def generate_tbank(label,m1_min,m1_max,m2_min,m2_max,match):

    """ Generates a templatebank corresponding to given parameter ranges using pycbc_geom_nonspinbank

    Parameters
    ----------
    label: str

   
   m1_min:
                        Minimum mass1: must be >= min-mass2. REQUIRED.
                        UNITS=Solar mass (default: None)
   m1_max:
                        Maximum mass1: must be >= max-mass2. REQUIRED.
                        UNITS=Solar mass (default: None)
   m2_min:
                        Minimum mass2. REQUIRED. UNITS=Solar mass (default:
                        None)
   m2_max:
                        Maximum mass2. REQUIRED. UNITS=Solar mass (default:
                        None)
   match: 
                        Minimal Match between neighbouring templates.

  Other OPTIONAL parameters from pycbc_grom_nonspinbank: (To be incorporated as keyword arguments in future)
  --max-total-mass MAX_TOTAL_MASS
                        Maximum total mass. OPTIONAL, if not provided the max
                        total mass is determined by the component masses.
                        UNITS=Solar mass (default: None)
  --min-total-mass MIN_TOTAL_MASS
                        Minimum total mass. OPTIONAL, if not provided the min
                        total mass is determined by the component masses.
                        UNITS=Solar mass (default: None)
  --max-chirp-mass MAX_CHIRP_MASS
                        Maximum chirp mass. OPTIONAL, if not provided the max
                        chirp mass is determined by the component masses.
                        UNITS=Solar mass (default: None)
  --min-chirp-mass MIN_CHIRP_MASS
                        Minimum total mass. OPTIONAL, if not provided the min
                        chirp mass is determined by the component masses.
                        UNITS=Solar mass (default: None)
  --max-eta MAX_ETA     Maximum symmetric mass ratio. OPTIONAL, no upper bound
                        on eta will be imposed if not provided. UNITS=Solar
                        mass. (default: 0.25)
  --min-eta MIN_ETA     Minimum symmetric mass ratio. OPTIONAL, no lower bound
                        on eta will be imposed if not provided. UNITS=Solar
                        mass. (default: 0.0)


    Returns
    -------
    output_cache_file: str
        Path to the generated output xml table

    """
#    logger.info('Building gw_data_find command line')

    outdir = params.outdir

    cache_file = 'templatebank_{}.xml'.format(label)
    output_cache_file = os.path.join(outdir, cache_file)


    cl_list = ['pycbc_geom_nonspinbank ']
    cl_list.append('--pn-order threePointFivePN')
    cl_list.append('--f0 50')
    cl_list.append('--f-low 15')
    cl_list.append('--f-upper 4096')
    cl_list.append('--delta-f 0.1')
    cl_list.append('--min-match {}'.format(match))
    cl_list.append('--min-mass1 {}'.format(m1_min))
    cl_list.append('--max-mass1 {}'.format(m1_max))
    cl_list.append('--min-mass2 {}'.format(m2_min))
    cl_list.append('--max-mass2 {}'.format(m2_max))
    cl_list.append('--verbose')
    cl_list.append('--psd-model aLIGOZeroDetHighPower')
    cl_list.append('--output-file {}'.format(output_cache_file))

    cl = ' '.join(cl_list)
    print("Generating templatebank as |  "+cl)
    run_commandline(cl)
    print("Templatebank xml file saved at "+output_cache_file)
    return output_cache_file



def _xml_to_txt(filename, label):
    
    # Convert xml into txt:
    xmlfile = filename
    #xmlfile = os.path.join(params.outdir,xml_filename)
    cache_file = "templatebank_{}.txt".format(label)
    output_cache_file = os.path.join(params.outdir,cache_file)

    cl_list = ['ligolw_print ']
    cl_list.append('{}'.format(xmlfile))
    cl_list.append('-t sngl_inspiral')
    cl_list.append('-c mass1')
    cl_list.append('-c mass2')
    
    cl_list.append('> {}'.format(output_cache_file))
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_cache_file
