#!/bin/bash

# custom config
DATA=/data
TRAINER=AdvZeroshotCLIP
OUTPUT_DIR=/output_dir

CFG=vit_b32_ep5_batch4_2ctx_notransform
for DATASET in dtd; do
    for SEED in 1 2 3 4 5 6 7 8 9 10; do
        python train.py \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/advMaPLe/${CFG}.yaml \
        --output-dir ${OUTPUT_DIR}/${TRAINER}/${CFG}/${DATASET}/${SEED} \
        --eval-only \
        --seed ${SEED} \
        DATASET.NUM_SHOTS 2 \
        DATASET.SUBSAMPLE_CLASSES base \

    done
done