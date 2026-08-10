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

    MODEL_DIR=few_shot/${DATASET}/${COMMON_DIR}

    # -----------------------------
    # Download source checkpoint
    # -----------------------------
    if [ -d "$MODEL_DIR" ]; then
        echo "The checkpoint exists at ${MODEL_DIR} (skipping download)"
    else
        python download_ckpts.py \
            --task few_shot \
            --dataset ${DATASET} \
            --shots ${SHOTS} \
            --trainer ${TRAINER}

        echo "Downloaded the checkpoint for ${MODEL_DIR}"
    fi

    # -----------------------------
    # Evaluate on target datasets
    # -----------------------------
    for TARGET in ${TARGETS}
    do
        DIR=output_eval/dg/${TARGET}/${COMMON_DIR}

        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${TARGET}.yaml \
                --config-file configs/trainers/${METHOD}/few_shot/${DATASET}.yaml \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --output-dir ${DIR} \
                --eval-only \
                TRAINER.BIOMEDCOOP.N_CTX ${NCTX} \
                TRAINER.BIOMEDCOOP.CSC ${CSC} \
                TRAINER.BIOMEDCOOP.CLASS_TOKEN_POSITION ${CTP}
        fi
    done
done