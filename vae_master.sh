#!/bin/sh
for i in $(seq 1 10)
for nmq in $(seq 1 7); do
    sbatch --export=NUMMIDDLEQUBIT=$nmq --export=ITERATION=$i vae_agent.sh 

done
