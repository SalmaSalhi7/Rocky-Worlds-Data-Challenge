#!/bin/bash

#SBATCH --tasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem_per_cpu=7G
#SBATCH --time=2:00:00
#SBATCH --account=def-rdoyon
#SBATCH --job-name=indiv_fit
#SBATCH --output=indiv_fit_%j.out
#SBATCH --error=indiv_fit_%j.err
#SBATCH --mail-user="salma.salhi@umontreal.ca"
#SBATCH --mail-type=ALL

source $HOME/rocky-worlds/bin/activate

python /home/ssalhi/Rocky-Worlds/step2_individual_fit/run_step2.py \
  --config "/home/ssalhi/Rocky-Worlds/configs/individual_fit_template.yaml" \
