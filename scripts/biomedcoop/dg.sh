#!/bin/bash

# custom config
DATA=$1
DATASET=$2      # source dataset, e.g. busi or btmri
MODEL=$3

SHOTS=16
NCTX=4
CSC=False
CTP=end
LOADEP=100

METHOD=BiomedCoOp
TRAINER=BiomedCoOp_${MODEL}

# -----------------------------
# Target datasets for DG eval
# -----------------------------
if [ "$DATASET" = "busi" ]; then
    TARGETS="buid busbra udiat"
elif [ "$DATASET" = "btmri" ]; then
    TARGETS="btmri_p btmri_s brisc"
else
    echo "Unknown source dataset: ${DATASET}"
    echo "Please add target datasets for ${DATASET} in dg.sh"
    exit 1
fi

for SEED in 1 2 3
do
    COMMON_DIR=shots_${SHOTS}/${TRAINER}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}

    SOURCE_DIR=output/dg/${DATASET}/${COMMON_DIR}

    # -----------------------------
    # Train on source dataset
    # -----------------------------
    if [ -d "$SOURCE_DIR" ]; then
        echo "Oops! The source results exist at ${SOURCE_DIR} (so skip training)"
    else
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
            --output-dir ${SOURCE_DIR} \
            TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
            TRAINER.BIOMEDCOOP.CSC ${CSC} \
            TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
    fi

    # -----------------------------
    # Evaluate on target datasets
    # -----------------------------
    for TARGET in ${TARGETS}
    do
        TARGET_DIR=output/dg/${TARGET}/${COMMON_DIR}

        if [ -d "$TARGET_DIR" ]; then
            echo "Oops! The DG results exist at ${TARGET_DIR} (so skip evaluation)"
        else
            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${TARGET}.yaml \
                --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
                --output-dir ${TARGET_DIR} \
                --model-dir ${SOURCE_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
                TRAINER.BIOMEDCOOP.CSC ${CSC} \
                TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
done