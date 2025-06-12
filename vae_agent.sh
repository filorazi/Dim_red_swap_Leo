#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00 # Runtime in D-HH:MM:SS
#SBATCH -o ./Logs/VAE_06_earth_mover_%A_%a.out # File to which STDOUT will be written
#SBATCH -e ./Logs/VAE_06_earth_mover_%A_%a.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=filippo.orazi2@unibo.it # Email to which notifications will be sent
#SBATCH --account=IscrC_DRST
#SBATCH --partition=dcgp_usr_prod

module load python/3.10.8--gcc--8.5.0
source ~/dimred_venv/bin/activate

echo "Job runs on node:"
echo $SLURM_NODELIST
cd $SLURM_SUBMIT_DIR

export TASK_ID=$SLURM_ARRAY_TASK_ID
export JOB_ID=$SLURM_ARRAY_JOB_ID
export JID=$SLURM_JOB_ID

BATCHSIZE=10
EPOCHS=150
LISTOPSUPPORT="1 2 3"
LISTOPSUPPORTPROBS="1. 1. 1."
LISTOPSUPPORTMAXRANGE="1 5 3"
VALSPLIT=0.2
OPTSTEP=0.2
NUMINPUTQUBITS=8

echo COSTTYPE=earth-mover
echo NUMINPUTQUBITS=$NUMINPUTQUBITS
echo NUMMIDDLEQUBIT=$NUMMIDDLEQUBIT
echo EPOCHS=$EPOCHS
echo OPTSTEP=$OPTSTEP
echo BATCHSIZE=$BATCHSIZE
echo VALSPLIT=$VALSPLIT
echo LISTOPSUPPORT=$LISTOPSUPPORT
echo LISTOPSUPPORTPROBS=$LISTOPSUPPORTPROBS
echo LISTOPSUPPORTMAXRANGE=$LISTOPSUPPORTMAXRANGE
echo 'NO JAX' 

OUT_BASE=/leonardo/home/userexternal/forazi00/Dim_red_swap_Leo/out
OUT="${OUT_BASE}/out_8to${NUMMIDDLEQUBIT}"  # Unique output file for each job

python single_run.py -ni ${NUMINPUTQUBITS} -nt ${NUMMIDDLEQUBIT} -b ${BATCHSIZE} -e ${EPOCHS} -v ${VALSPLIT} -sz ${OPTSTEP} -of ${OUT} -ls ${LISTOPSUPPORT}  -lp ${LISTOPSUPPORTPROBS}  -lr ${LISTOPSUPPORTMAXRANGE} -njx

echo "job finished"
wait
