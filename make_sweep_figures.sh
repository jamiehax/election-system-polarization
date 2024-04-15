#!/usr/bin/env bash

#SBATCH --job-name=sweep_figs # job name
#SBATCH --output=logs/sweep_figs-%j.out # standard output and error log
#SBATCH --mail-user=jhackney@middlebury.edu # where to send mail
#SBATCH --mail-type=ALL # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mem=8gb # job memory request 
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard # partition (queue) 
#SBATCH --time=01:00:00 # time limit hrs:min:sec 

# print SLURM envirionment variables
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}" echo "Starting: "`date +"%D %T"` 

# run script 
python3 make_sweep_figures.py

# end of job info 
echo "Ending: "`date +"%D %T"`