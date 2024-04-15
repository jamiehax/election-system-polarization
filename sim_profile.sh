#!/usr/bin/env bash

#SBATCH --job-name=sim_profile                    # job name
#SBATCH --output=logs/sim_profile-%j.out           # standard output and error log
#SBATCH --mail-user=jhackney@middlebury.edu     # where to send mail
#SBATCH --mail-type=ALL                         # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=24gb                              # job memory request 
#SBATCH --cpus-per-task=20                       # number of CPUs to allocate
#SBATCH --partition=standard                    # partition (queue) 
#SBATCH --time=48:00:00                         # time limit hrs:min:sec 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 

# run script 
python3 sim_profile.py

# end of job info 
echo "Ending: "`date +"%D %T"`