#!/bin/bash

# custom config
DATA=/data
TRAINER=zsrobust
DATASET=$1
SHOTS=$2
CFG=vit_b32_ep10_batch4
SEED=$3

EPS=1
ALPHA=1
TRAIN_ITER=2
TEST_ITER=100


OUTPUT_DIR=/output_dir


    # Training
TRAIN_DIR=${OUTPUT_DIR}/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

if [ -d "$TRAIN_DIR" ]; then
    echo "Oops! The results exist at ${TRAIN_DIR} (so skip this job)"
else
    echo "Starting training for ${DATASET}..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${TRAIN_DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    ATTACK.PGD.TRAIN_ITER ${TRAIN_ITER} \
    ATTACK.PGD.TEST_ITER ${TEST_ITER} \
    ATTACK.PGD.EPS ${EPS} \
    ATTACK.PGD.ALPHA ${ALPHA} \
    
fi