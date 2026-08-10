#!/bin/bash

# custom config
DATA=$1
DATASET=$2      # source dataset, e.g. busi or btmri
MODEL=$3

CFG=vit_b16

METHOD=Zeroshot
TRAINER=Zeroshot${MODEL}

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

for TARGET in ${TARGETS}
do
    DIR=output/dg/${TARGET}/${TRAINER}/${CFG}

    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
            --root ${DATA} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${TARGET}.yaml \
            --config-file configs/trainers/${METHOD}/${CFG}.yaml \
            --output-dir ${DIR} \
            --eval-only
    fi
done