#!/bin/sh
# FILENAME: job.sh

experiment_name=$1
mode=$2
training_id=$3

base_command=(data=multi3drefer experiment_name=${experiment_name} +detector_path=checkpoints/PointGroup_ScanNet.ckpt)

if [ "$mode" = "train" ]; then
    if [ -z "$training_id" ]; then
        python train.py "${base_command[@]}" scheduled_job=True
    else
        python train.py "${base_command[@]}" scheduled_job=True resume=True
    fi
elif [ "$mode" = "test" ]; then
    python test.py data.inference.split=val ckpt_path=output/Multi3DRefer/${experiment_name}/training/best.ckpt "${base_command[@]}"
elif [ "$mode" = "eval" ]; then
    python evaluate.py data.evaluation.split=val "${base_command[@]}"
fi