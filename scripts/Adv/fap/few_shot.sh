#!/bin/bash

# running command:
# CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/few_shot.sh caltech101 16 0

ADV_TERM=cos
EPS=1
ALPHA=1
TRAIN_ITER=2
TEST_ITER=100
LAMBDA_1=1.5

ATTACK_RELATED=adv_term-${ADV_TERM}_eps-${EPS}_alpha-${ALPHA}_train_iter-${TRAIN_ITER}_test_iter-${TEST_ITER}_lambda1-${LAMBDA_1}

#2. dataset_related
TRAIN_DATASET=$1
SHOTS=$2

# 3. model_related 
CFG=vit_b32_ep10_batch4_2ctx_notransform
TRAINER=FAP

MODEL_RELATED=${TRAINER}_${CFG}

#4. seed
SEED=$3

SEED_RELATED=seed${SEED}

# 4. path_related
#modify this
OUTPUT_DIR=/output_dir

TRAIN_DIR=${OUTPUT_DIR}/few_shot/${ATTACK_RELATED}/${MODEL_RELATED}/${TRAIN_DATASET}_shot${SHOTS}/${SEED_RELATED}

DATA=/data



        if [ -d "$TRAIN_DIR" ]; then
            echo "Oops! The training results exist at ${TRAIN_DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${TRAIN_DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${TRAIN_DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            ATTACK.PGD.ADV_TERM ${ADV_TERM}\
            ATTACK.PGD.TRAIN_ITER ${TRAIN_ITER}\
            ATTACK.PGD.TEST_ITER ${TEST_ITER}\
            ATTACK.PGD.EPS ${EPS}\
            ATTACK.PGD.LAMBDA_1 ${LAMBDA_1}\
            ATTACK.PGD.ALPHA ${ALPHA}\

        fi