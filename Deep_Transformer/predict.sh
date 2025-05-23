fasta_path=$(realpath ../fung_data.fasta)
model_location=$(realpath ../fung/checkpoint.pt)
out_dir=$(realpath output)

#Activate the environment
source /home/cplei/miniconda3/bin/activate  /home/cplei/miniconda3/envs/DeepTransformer
#echo
python predict.py --fasta_path $fasta_path \
		--model_location $model_location \
		--secretion_systems eff \
		--out_dir $out_dir  
