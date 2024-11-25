#!/usr/bin/env bash
DTU_TESTING="C:/Users/USER/Desktop/synthdataset10"
CKPT_FILE="./checkpoints/d256_synth10_fringe_baye_depthmin200_normalize/model_10.ckpt"
OUT_PATH="C:/Dev/MVSNet/outputs/uph/256_synth10_fringe_baye_depthmin200_normalize"

OUT_PATH_UPH="$OUT_PATH/depth_est"
OUT_PATH_CONF="$OUT_PATH/sigma"
OUT_PATH_PROB="$OUT_PATH/prob_vol"

if [ ! -d $OUT_PATH ]; then
    mkdir -p $OUT_PATH
    mkdir -p $OUT_PATH_UPH
    mkdir -p $OUT_PATH_CONF
    mkdir -p $OUT_PATH_PROB
fi

python eval_phase.py --dataset=dtu_yao_phase --batch_size=1 --numdepth=256 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt \
     --loadckpt $CKPT_FILE --outdir $OUT_PATH --bayesian_mode
