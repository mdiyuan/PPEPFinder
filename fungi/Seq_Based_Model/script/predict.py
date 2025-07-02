import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from esm import Alphabet, FastaBatchedDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from DeepSecE.model import EffectorTransformer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def predict(model, fasta, batch_size, device, outdir, pos_labels, save_attn=False):
    predicted_labels = ['Non-', 'Eff']
    print(f'Loading FASTA Dataset from {fasta}')

    dataset = FastaBatchedDataset.from_file(fasta)
    alphabet = Alphabet.from_architecture("roberta_large")
    loader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), num_workers=0, batch_size=batch_size)

    model.eval()
    probs = []
    preds = []
    names = []
    lengths = []
    systems = []
    scores = []
    seq_records = []

    if save_attn:
        attn_dict = {}

    with torch.no_grad():
        for labels, strs, toks in tqdm(loader):
            toks = toks.to(device)
            if save_attn:
                out, attn = model(strs, toks)
                attn = attn.cpu().numpy()
            else:
                out = model(strs, toks)
            prob = torch.softmax(out, dim=1)
            _, pred = torch.max(prob, 1)

            probs.append(prob.detach().cpu().numpy())
            preds.append(pred.detach().cpu().numpy())

            for i, str_seq in enumerate(strs):
                name = labels[i].split()[0]
                pred_label = predicted_labels[pred[i].cpu().numpy()]

                systems.append(pred_label)
                scores.append(prob[i, pred[i]].item())

                if pred_label in pos_labels:
                    if save_attn:
                        seq = str_seq[:1020]
                        avg_attn = attn[i, :, :len(seq), :len(seq)].sum(0).mean(0)
                        attn_dict[name] = avg_attn
                    record = SeqRecord(Seq(str_seq), id=name, description=f'putative {pred_label} protein')
                    seq_records.append(record)

                names.append(name)
                lengths.append(len(str_seq))

    probs = np.concatenate(probs)
    preds = np.concatenate(preds)

    probs_non_effector = probs[:, 0]
    probs_eff = probs[:, 1]

    result = pd.DataFrame({
        'name': names,
        'score': scores,
        'Non-Effectoe.prob': probs_non_effector,
        'Effector.prob': probs_eff,
        'length': lengths
    })
    result = result.round(4)

    result_file = os.path.join(outdir, 'predictions_with_all_proteins.csv')
    print(f"Writing prediction results in {result_file}")
    result.to_csv(result_file, index=False)

    effector = result[result['system'].isin(pos_labels)]
    effector_file = os.path.join(outdir, 'results.csv')
    print(f"Writing putative secreted proteins in {effector_file}")
    effector.to_csv(effector_file, index=False)

    fasta_file = os.path.join(outdir, 'secreted-proteins.fasta')
    print(f"Writing putative secreted proteins in {fasta_file}")
    SeqIO.write(seq_records, fasta_file, 'fasta')

    if save_attn:
        attn_file = os.path.join(outdir, 'attn.npz')
        print(f"Saving secreted protein attention in {attn_file}")
        np.savez(attn_file, **attn_dict)

def main(args):
    set_seed(42)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Using device {device} for prediction')
    start_time = time.time()

    model = EffectorTransformer(1280, 33, hid_dim=256, num_layers=1, heads=4,
                                 dropout_rate=0.4, num_classes=2, return_attn=args.save_attn)
    model.to(device)

    print(f'Loading model from {args.model_location}')
    if args.no_cuda:
        model.load_state_dict(torch.load(args.model_location, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(args.model_location))

    predict(model, args.fasta_path, args.batch_size, device,
            args.out_dir, args.secretion_systems, args.save_attn)

    end_time = time.time()
    secs = end_time - start_time

    print(f'It took {secs:.1f}s to finish the prediction')

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Predict secreted substrate proteins from protein sequences in a FASTA file.")

    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size used in prediction. (default: 1)')
    parser.add_argument('--fasta_path', required=True, type=str,
                        help='input ordered protein sequences.')
    parser.add_argument('--model_location', required=True, type=str,
                        help='path to the model weights.')
    parser.add_argument('--secretion_systems', nargs='+', default=['eff'],
                        help="types of secreted proteins requiring prediction. (default: eff)")
    parser.add_argument('--out_dir', default='./', type=str,
                        help='output directory of prediction results.')
    parser.add_argument('--save_attn', action='store_true',
                        help='save the sequence attention of secreted proteins.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='add when CUDA is not available.')

    args = parser.parse_args()
    main(args)
