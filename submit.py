#!/usr/bin/python
import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("inputfile", help="python file to submit")
parser.add_argument("--args", default="", help="arguments for python file to submit")
a = parser.parse_args()
a.args = a.args.replace('"', "")

if a.args != "":
    jobname = a.args.split()[0] + ".job"
    outfile = a.args.split()[0] + ".out"
else:
    jobname = os.path.basename(a.inputfile)[:-3] + ".job"
    outfile = a.inputfile[:-3] + ".out"

with open(jobname, "w") as f:
    f.write("#!/bin/bash\n")
    f.write("#$ -N {}\n".format(jobname))
    f.write("#$ -cwd\n")
    f.write("#$ -q UI-GPU\n")
    f.write("#$ -pe smp 80\n")
    f.write("#$ -l ngpus=4\n")
    f.write("#$ -l h_rt=72:00:00\n")
    f.write("conda activate molgnn\n")
    f.write("cd $SGE_O_WORKDIR\n")
    f.write("python {} {} > {}\n".format(a.inputfile, a.args, outfile))
os.system("qsub {}".format(jobname))
os.system("sleep 1")
