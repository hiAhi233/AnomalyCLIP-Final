#!/bin/bash

# custom config
DATA=$1
DATASET=$2      # source dataset, e.g. busi or btmri
MODEL=$3

SHOTS=16
NCTX=4
CSC=False
CTP=end
CFG=vit_b16
LOADEP=100

METHOD=CoCoOp
TRAINER=CoCoOp_${MODEL}

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
            --config-file configs/trainers/${METHOD}/${CFG}.yaml \
            --output-dir ${SOURCE_DIR} \
            TRAINER.COCOOP.N_CTX ${NCTX} \
            TRAINER.COCOOP.CSC ${CSC} \
            TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
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
                --config-file configs/trainers/${METHOD}/${CFG}.yaml \
                --output-dir ${TARGET_DIR} \
                --model-dir ${SOURCE_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                TRAINER.COCOOP.N_CTX ${NCTX} \
                TRAINER.COCOOP.CSC ${CSC} \
                TRAINER.COCOOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}
        fi
    done
done