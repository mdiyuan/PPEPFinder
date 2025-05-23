import os
import sys
sys.path.append("./SaProt-main/")
import numpy as np
import torch
from utils.foldseek_util import get_struc_seq
from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer

def get_saprot_embedding(pdb_file, output_path):
    foldseek_path = "./SaProt-main/bin/foldseek"
    config = {
        "task": "base",
        "config_path": "./SaProt-main/SaProt_650M_AF2",
        "load_pretrained": True,
    }


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SaprotBaseModel(**config).to(device)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])

    chain_ids = ["A"]  


    protein_id = os.path.splitext(os.path.basename(pdb_file))[0]

    try:
        parsed_seqs = get_struc_seq(foldseek_path, pdb_file, chain_ids, plddt_mask=False)
        for chain_id in chain_ids:
            if chain_id not in parsed_seqs:
                print(f"Chain {chain_id} not found in {protein_id}. Skipping...")
                continue

            _, _, combined_seq = parsed_seqs[chain_id]

  
            inputs = tokenizer(combined_seq, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}


            all_hidden_states = model.get_hidden_states(inputs)
            last_layer_embedding = all_hidden_states[-1] 
            embedding_array = last_layer_embedding.detach().cpu().numpy()

            np.save(output_path, embedding_array)
            print(f"Saved embedding for {protein_id} to {output_path}")

    except Exception as e:
        print(f"Error processing {protein_id}: {e}")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    get_saprot_embedding(args.pdb_file, args.output_path)
