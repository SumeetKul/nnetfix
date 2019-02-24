import numpy as np
import params 
import subprocess

def generate_templatebank(label,m1_min,m1_max,m2_min,m2_max,match):

    """ Builds a gw_data_find call and process output

    Parameters
    ----------
    observatory: str, {H1, L1, V1}
        Observatory description
    gps_start_time: float
        The start time in gps to look for data
    duration: int
        The duration (integer) in s
    calibrartion: int {1, 2}
        Use C01 or C02 calibration
    outdir: string
        A path to the directory where output is stored
    query_type: string
        The LDRDataFind query type

    Returns
    -------
    output_cache_file: str
        Path to the output cache file

    """
    logger.info('Building gw_data_find command line')

#    observatory_lookup = dict(H1='H', L1='L', V1='V')
#    observatory_code = observatory_lookup[observatory]

 #   if query_type is None:
 #       logger.warning('No query type provided. This may prevent data from being read.')
 #       if observatory_code == 'V':
 #           query_type = 'V1Online'
 #       else:
 #           query_type = '{}_HOFT_C0{}'.format(observatory, calibration)

 #   logger.info('Using LDRDataFind query type {}'.format(query_type))

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
    cl_list.append('--url-type file')
    cl_list.append('--lal-cache')
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_cache_file



def _xml_to_txt(label):
    
    # Convert xml into txt:
    xml_filename = "templatebank_{}.xml".format(label)
    xmlfile = os.path.join(outdir,xml_filename)
    output_file = "templatebank_{}.txt".format(label)

    cl_list = ['ligolw_print {}']
    cl_list.append('{}'.format(xmlfile))
    cl_list.append('-t sngl_inspiral')
    cl_list.append('-c mass1')
    cl_list.append('-c mass2')
    
    cl_list.append('> {}'.format(output_file)
    cl = ' '.join(cl_list)
    run_commandline(cl)
    return output_file
