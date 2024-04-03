#!/usr/bin/env bash
# slurm template for serial jobs
# Set SLURM options

#SBATCH --job-name=election_simulation # job name
#SBATCH --output=election_simulation-%j.out # standard output and error log
#SBATCH --mail-user=jhackney@middlebury.edu # where to send mail
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=12gb # job memory request 
#SBATCH --partition=standard # partition (queue) 
#SBATCH --time=02:00:00 # time limit hrs:min:sec 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 

# run script 
python3 run.py

# end of job info 
echo "Ending: "`date +"%D %T"`