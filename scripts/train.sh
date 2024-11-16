#!/bin/sh
# FILENAME: train.sh

experiment_name="baseline"

# Train
./job.sh $experiment_name train

for training_id in {1..29}
do
    ./job.sh $experiment_name train $training_id
done

# Test
./job.sh $experiment_name test

# Evaluate
./job.sh $experiment_name eval