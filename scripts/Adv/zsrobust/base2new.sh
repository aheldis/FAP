#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 bash scripts/Adv/zsrobust/base2new.sh

# custom config
DATA=/data
TRAINER=zsrobust

for SEED in 1 2 3; do
    CFG=vit_b32_ep10_batch4
    SHOTS=16
    LOADEP=10
    SUB=new

    OUTPUT_DIR=/output_dir


    DATASETS=(imagenet caltech101 dtd eurosat oxford_pets oxford_flowers stanford_cars food101 fgvc_aircraft sun397 ucf101)

    for DATASET in "${DATASETS[@]}"; do
        echo "Processing dataset: ${DATASET}"

        # Training
        TRAIN_DIR=${OUTPUT_DIR}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
            DATASET.SUBSAMPLE_CLASSES base
        fi

        # Validation
        MODEL_DIR=${OUTPUT_DIR}/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        VALID_DIR=${OUTPUT_DIR}/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

        if [ -d "$VALID_DIR" ]; then
            echo "Evaluating model for ${DATASET}..."
            echo "Results are available in ${VALID_DIR}. So skip the job"
        else
            echo "Evaluating model for ${DATASET}..."
            echo "Running the evaluation and saving the output to ${VALID_DIR}"
            python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${VALID_DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
        fi

        
    done
done