#!/usr/bin/env bash
MVS_TRAINING="C:/Users/USER/Desktop/synthdataset10"
EXP_NAME="d384_synth10_fringe_depthmin200_vanila"
LOG_DIR="./checkpoints/$EXP_NAME"

if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

dirAndName="$LOG_DIR/$EXP_NAME.log"
if [ ! -d $dirAndName ]; then
    touch $dirAndName
fi

python -u train_phase.py --dataset=dtu_yao_phase --batch_size=1 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt \
    --wandb --numdepth=384 --experiment_name $EXP_NAME --logdir $LOG_DIR