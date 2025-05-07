#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# #SBATCH --begin=2021-07-10T02:00:00 # Defer start to this much time
#SBATCH --time=7-00:00:00 # Runtime in D-HH:MM:SS
#SBATCH -o ./Logs/VAE_06_earth_mover_%A_%a.out # File to which STDOUT will be written
#SBATCH -e ./Logs/VAE_06_earth_mover_%A_%a.err # File to which STDERR will be written
#SBATCH --mail-type=FAIL # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=filippo.orazi2@unibo.it # Email to which notifications will be sent


module purge
module load python/3.8.12--gcc--8.4.1
source dimred_venv/bin/activate

echo "Job runs on node:"
echo $SLURM_NODELIST
cd $SLURM_SUBMIT_DIR

export TASK_ID=$SLURM_ARRAY_TASK_ID
export JOB_ID=$SLURM_ARRAY_JOB_ID
export JID=$SLURM_JOB_ID

BATCHSIZES=(100)
BATCHSIZE=${BATCHSIZES[$(( ${TASK_ID} % 1 ))]}
EPOCHS=2500
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

OUT_BASE=/g100/home/userexternal/forazi00/cinequbo/FILIPPO/outputs/out
OUT="${OUT_BASE}/out_8to${NUMMIDDLEQUBIT}/out_${ITERATION}.scan"  # Unique output file for each job

python rest.py $OUT
# srun -N 1 -n 1 python single_run.py -ni ${NUMINPUTQUBITS} -nt ${NUMMIDDLEQUBIT} -b ${BATCHSIZE} -e ${EPOCHS} -v ${VALSPLIT} -sz ${OPTSTEP} -of 'runs/ee' -ls ${LISTOPSUPPORT}  -lp ${LISTOPSUPPORTPROBS}  -lr ${LISTOPSUPPORTMAXRANGE} -njx

echo "job finished"
wait
