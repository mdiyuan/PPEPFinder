odir="./outdir"
seq_file="./seq.fasta"
python ./script/predict.py \
            --fasta_path $seq_file \
            --model_location ./fungi/Seq_Based_Model/esm2_transformer_model/checkpoint.pt \
            --secretion_systems eff \
            --out_dir $odir \
            --no_cuda 
