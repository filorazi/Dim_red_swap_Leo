#!/bin/sh
for i in $(seq 1 2); do
for nmq in $(seq 4 7); do
    sbatch --export=NUMMIDDLEQUBIT=$nmq,ITERATION=$i vae_agent.sh

done
done
