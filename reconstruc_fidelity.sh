#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00 # Runtime in D-HH:MM:SS
#SBATCH -o ./Logs/reco%A_%a.out # File to which STDOUT will be written
#SBATCH -e ./Logs/reco%A_%a.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=filippo.orazi2@unibo.it # Email to which notifications will be sent
#SBATCH --account=IscrC_DRST
#SBATCH --partition=dcgp_usr_prod

module load python/3.10.8--gcc--8.5.0
source ~/dimred_venv/bin/activate

echo "Job runs on node:"
echo $SLURM_NODELIST
cd $SLURM_SUBMIT_DIR

python reconstruction_fidelity.py
echo "job finished"
wait
