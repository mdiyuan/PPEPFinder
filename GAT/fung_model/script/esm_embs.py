import numpy as np
import pandas as pd
import torch
import esm

def get_esm_embeddings(csv_file_path,output_path):
    df = pd.read_csv(csv_file_path)
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    model.eval()  
    repr_layer = 33  

    def get_protein_embeddings(sequence):
        tokens = alphabet.encode(sequence)
        tokens_tensor = torch.tensor(tokens) 
        with torch.no_grad():
            out = model(tokens_tensor.unsqueeze(0), repr_layers=[repr_layer], return_contacts=False)
        embeddings = out["representations"][repr_layer] 
        return embeddings.squeeze(0).cpu().numpy()
    embeddings_list = []
    total_length = 0
    embedding_dim = 1280  
    for seq in df['seq']:
        print(f"Processing sequence of length {len(seq)}")  
        embeddings = get_protein_embeddings(seq)
        print(f"Generated embeddings shape: {embeddings.shape}")
        embeddings_list.append(embeddings)
        total_length += embeddings.shape[0] 
    final_embeddings = np.zeros((total_length, embedding_dim))
    current_index = 0
    for emb in embeddings_list:
        seq_len = emb.shape[0]
        final_embeddings[current_index:current_index + seq_len, :] = emb
        current_index += seq_len

    np.save(output_path, final_embeddings)

    print(f"Total concatenated sequence length: {total_length}")
    print(f"Concatenated embeddings saved to {output_path}. Shape: {final_embeddings.shape}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    get_esm_embeddings(args.csv_file_path,args.output_path)
