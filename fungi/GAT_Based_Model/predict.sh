#!/bin/bash

# 使用示例: bash run_effector_prediction.sh /path/to/protein.pdb

# 获取输入参数
if [ -z "$1" ]
then
    pdb_file="../fung_eff.pdb"
    protein_name=$(basename "$pdb_file" .pdb)
    MODEL_DIR="../saprot_GAT_model"   # 自己放置的模型目录，确保里面有 .h5 模型文件
    odir="./outdir"
fi

# 创建输出目录

echo "Step 1: Extracting structure and generating residue contact matrix"
(
source /home/cplei/miniconda3/bin/activate /storage/home/cplei/micromamba/envs/PGAT
python ./fungi/GAT_Based_Model/script/PDBProcess.py \
 --pdb_file "$pdb_file" --camp_thresh 9 --output_path "$odir/matrix_9.csv"
python ./fungi/GAT_Based_Model/script/PDBProcess.py \
 --pdb_file "$pdb_file" --camp_thresh 10 --output_path "$odir/matrix_10.csv"
)

echo "step2: Generating ESM-1b embeddings..."
(
source /home/cplei/miniconda3/bin/activate /home/cplei/miniconda3/envs/esm2
python ./fungi/GAT_Based_Model/script/esm_embs.py \
 --csv_file_path "$odir/matrix_9.csv" --output_path "$odir/esm_9.npy"
)

echo "step3: Generating SaProt embeddings..."
(
source /home/cplei/miniconda3/bin/activate /storage/home/cplei/micromamba/envs/SaProt
python ./fungi/GAT_Based_Model/script/get_saprot_embedding.py \
  --pdb_file "$pdb_file" --output_path "$odir/saprot_10.npy"
)

echo "step4: Running ESM-1b_GAT model"
(
source /home/cplei/miniconda3/bin/activate /storage/home/cplei/micromamba/envs/PGAT
python ./fungi/GAT_Based_Model/script/ESM_GAT_model_Predict.py \
 --csv_file "$odir/matrix_9.csv" --npy_file "$odir/esm_9.npy" --output_file "$odir/esm_9_prediction.csv"
)

echo "step5: Running SaProt_GAT model"
(
source /home/cplei/miniconda3/bin/activate /storage/home/cplei/micromamba/envs/PGAT
python ./fungi/GAT_Based_Model/script/SaProt_GAT_model_Predict.py \
 --csv_file "$odir/matrix_9.csv" --npy_file "$odir/saprot_10.npy" --output_file "$odir/saprot_10_prediction.csv"    
)
