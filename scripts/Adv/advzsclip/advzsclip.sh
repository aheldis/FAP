#!/bin/bash

# custom config
DATA=/data
TRAINER=AdvZeroshotCLIP
OUTPUT_DIR=/output_dir

CFG=vit_b32_ep5_batch4_2ctx_notransform
for DATASET in imagenet caltech101 dtd eurosat oxford_pets oxford_flowers stanford_cars food101 fgvc_aircraft sun397 ucf101; do
    for SEED in 1 2 3; do
        python train.py \
        --root ${DATA} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/advMaPLe/${CFG}.yaml \
        --output-dir ${OUTPUT_DIR}/${TRAINER}/${CFG}/${DATASET}/${SEED} \
        --eval-only \
        --seed ${SEED} \

    done
done