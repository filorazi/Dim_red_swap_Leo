#!/bin/sh
for i in $(seq 1 1); do
for nmq in $(seq 1 2); do
    sbatch --export=NUMMIDDLEQUBIT=$nmq,ITERATION=$i vae_agent.sh

done
done
