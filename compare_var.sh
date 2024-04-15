#!/usr/bin/env bash

#SBATCH --job-name=compare_var                     # job name
#SBATCH --output=logs/compare_var-%j.out           # standard output and error log
#SBATCH --mail-user=jhackney@middlebury.edu        # where to send mail
#SBATCH --mail-type=ALL                            # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=24gb                                 # job memory request 
#SBATCH --cpus-per-task=25                         # number of CPUs to allocate                         
#SBATCH --partition=standard                       # partition (queue) 
#SBATCH --time=12:00:00                            # time limit hrs:min:sec 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 

# run script 
python3 compare_var.py

# end of job info 
echo "Ending: "`date +"%D %T"`