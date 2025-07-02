from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os

def load_pdb(pdbfile):
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('.')[0], pdbfile)
    residues = [r for r in structure.get_residues()]
    
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]
    
    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)
            
    return distances, seqs[0]

def load_cmap(pdbfile, cmap_thresh):
    D, seq = load_pdb(pdbfile)
    A = np.zeros(D.shape, dtype=np.float32)
    A[D < int(cmap_thresh)] = 1
    return A, seq

def process_pdb(pdb_path, cmap_thresh,output_path,label):
    A, seq = load_cmap(pdb_path, cmap_thresh)
    A_str = str(A.tolist())
    
    pdb_entry = {
        'filename': os.path.basename(pdb_path), 
        'A': A_str,
        'seq': seq,
        'label': label
    }
    
    df = pd.DataFrame([pdb_entry])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed PDB file saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", type=str, required=True)
    parser.add_argument("--camp_thresh", type=str, required=True,default=9)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--label", type=str,default='0')
    args = parser.parse_args()
    process_pdb(args.pdb_file, args.camp_thresh,args.output_path,args.label)