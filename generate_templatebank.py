import numpy as np
import params 
import subprocess


def generate_templatebank(label,m1_min,m1_max,m2_min,m2_max,match):
    

    
    Tmpltbankgen_command = """ pycbc_geom_nonspinbank --pn-order threePointFivePN --f0 50 --f-low 15 --f-upper 4096 --delta-f 0.1 --min-match {0} --min-mass1 {1} --max-mass1 {2} --min-mass2 {3} --max-mass2 {4} --verbose --psd-model aLIGOZeroDetHighPower --output-file "bank_{5}.xml" """.format(match,m1_min,m1_max,m2_min,m2_max,label)

    tmpltbank_dir = "/home/sumeet.kulkarni/nnetfix/"
    subprocess.call(Tmpltbankgen_command,shell=True)
    
    # Convert xml into txt:
    subprocess.call("ligolw_print {1}bank_{0}.xml -t sngl_inspiral -c mass1 -c mass2 > bank_{0}.txt".format(label,tmpltbank_dir)) 
