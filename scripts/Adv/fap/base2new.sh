#!/bin/bash

# running command:
# CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/fap/base2new.sh caltech101 0

# 1. attack_related
ADV_TERM=cos
EPS=2.0 #TODO
ALPHA=0.25 #TODO
TRAIN_ITER=2
TEST_ITER=20 #TODO
LAMBDA_1=1.5


ATTACK_RELATED=adv_term-${ADV_TERM}/eps-${EPS}_alpha-${ALPHA}_train_iter-${TRAIN_ITER}_test_iter-${TEST_ITER}_lambda1-${LAMBDA_1}
#2. dataset_related
DATASET=$1
SHOTS=16

DATASET_RELATED=${DATASET}_shots_${SHOTS}

# 3. model_related 
CFG=vit_b32_ep10_batch4_2ctx_notransform
TRAINER=FAP

MODEL_RELATED=${TRAINER}_${CFG}

#4. seed
SEED=${2}

SEED_RELATED=seed${SEED}

# 4. path_related
#modify this
OUTPUT_DIR=/output_dir

DATA=/data

TRAIN_DIR=${OUTPUT_DIR}/base2new/train_base/${ATTACK_RELATED}/${DATASET_RELATED}/${MODEL_RELATED}/${SEED_RELATED}
TEST_DIR=${OUTPUT_DIR}/base2new/test_new/${ATTACK_RELATED}/${DATASET_RELATED}/${MODEL_RELATED}/${SEED_RELATED}

        if [ -d "$TRAIN_DIR" ]; then
            echo "Oops! The training results exist at ${TRAIN_DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${TRAIN_DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base \
            ATTACK.PGD.ADV_TERM ${ADV_TERM}\
            ATTACK.PGD.TRAIN_ITER ${TRAIN_ITER}\
            ATTACK.PGD.TEST_ITER ${TEST_ITER}\
            ATTACK.PGD.EPS ${EPS}\
            ATTACK.PGD.LAMBDA_1 ${LAMBDA_1}\
            ATTACK.PGD.ALPHA ${ALPHA}\

        fi

        # Testing section
        LOADEP=10
        SUB=new

        if [ -d "$TEST_DIR" ]; then
            echo "Oops! The testing results exist at ${TEST_DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${TEST_DIR} \
            --model-dir ${TRAIN_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}\
            ATTACK.PGD.ADV_TERM ${ADV_TERM}\
            ATTACK.PGD.TRAIN_ITER ${TRAIN_ITER}\
            ATTACK.PGD.TEST_ITER ${TEST_ITER}\
            ATTACK.PGD.EPS ${EPS}\
            ATTACK.PGD.LAMBDA_1 ${LAMBDA_1}\
            ATTACK.PGD.ALPHA ${ALPHA}\

        fi
