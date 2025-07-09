#!/bin/bash

# source activate your_env_name

EMBED_PATH="./embedding_data_saprot.npy"
TEST_INDEX="fixed_test_index.npy"
OUT_DIR="./gat_results"

# Residue contact network files under different distance thresholds
FILE_NUMS=(4 6 8 9 10 12)


for NUM in "${FILE_NUMS[@]}"
do
    DATA_FILE="/mnt/disk1/mdyuan/PGAT-ABPp-main/T3SE/T3SE_data_${NUM}.csv"

    echo "starting threshold of the RCN used for training: $NUM"

    python train.py \
        --data_path "$DATA_FILE" \
        --embedding_path "$EMBED_PATH" \
        --file_number "$NUM" \
        --output_dir "$OUT_DIR" \
        --hidden_units 10 \
        --num_heads 6 \
        --num_layers 1 \
        --batch_size 32 \
        --epochs 500 \
        --lr 0.0001 \
        --fixed_test_index "$TEST_INDEX"

    echo "finished threshold of the RCN used for training: $NUM"
done
