#!/bin/sh
for i in $(seq 1 4); do
for nmq in $(seq 1 4); do
    sbatch --export=NUMMIDDLEQUBIT=$nmq,ITERATION=$i vae_agent.sh

done
done
