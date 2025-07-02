#!/bin/bash

# 使用示例: bash run_effector_prediction.sh /path/to/protein.pdb

# 获取输入参数
if [ -z "$1" ]
then
    pdb_file="../oomysetes_eff.pdb"
    protein_name=$(basename "$pdb_file" .pdb)
    MODEL_DIR="../saprot_GAT_model"   # 自己放置的模型目录，确保里面有 .h5 模型文件
    odir="./outdir"
fi

# 创建输出目录

echo "Step 1: Extracting structure and generating residue contact matrix"
(
source /home/mdyuan/miniconda3/bin/activate /storage/home/mdyuan/micromamba/envs/PGAT
python ./oomycetes/GAT_Based_Model/script/PDBProcess.py \
 --pdb_file "$pdb_file" --camp_thresh 9 --output_path "$odir/matrix_9.csv"
python ./oomycetes/GAT_Based_Model/script/PDBProcess.py \
 --pdb_file "$pdb_file" --camp_thresh 6 --output_path "$odir/matrix_6.csv"
)

echo "step2: Generating ESM-2 embeddings..."
(
source /home/mdyuan/miniconda3/bin/activate /home/mdyuan/miniconda3/envs/esm2
python ./oomycetes/GAT_Based_Model/script/esm_embs.py \
 --csv_file_path "$odir/matrix_9.csv" --output_path "$odir/esm_9.npy"
)

echo "step3: Generating SaProt embeddings..."
(
source /home/mdyuan/miniconda3/bin/activate /storage/home/mdyuan/micromamba/envs/SaProt
python ./oomycetes/GAT_Based_Model/script/get_saprot_embedding.py \
  --pdb_file "$pdb_file" --output_path "$odir/saprot_6.npy"
)

echo "step4: Running ESM-2_GAT model"
(
source /home/mdyuan/miniconda3/bin/activate /storage/home/mdyuan/micromamba/envs/PGAT
python ./oomycetes/GAT_Based_Model/script/ESM_GAT_model_Predict.py \
 --csv_file "$odir/matrix_9.csv" --npy_file "$odir/esm_9.npy" --output_file "$odir/esm_9_prediction.csv"
)

echo "step5: Running SaProt_GAT model"
(
source /home/mdyuan/miniconda3/bin/activate /storage/home/mdyuan/micromamba/envs/PGAT
python ./oomycetes/GAT_Based_Model/script/SaProt_GAT_model_Predict.py \
 --csv_file "$odir/matrix_9.csv" --npy_file "$odir/saprot_6.npy" --output_file "$odir/saprot_6_prediction.csv"    
)
