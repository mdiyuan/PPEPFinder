# train_gat.py
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, average_precision_score)
from GraphFromPDB import graph_from_pdb, Dataset
from Model import GraphAttentionNetwork

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAT on protein data")
    parser.add_argument('--data_path', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--embedding_path', type=str, required=True, help='Path to .npy embedding file')
    parser.add_argument('--file_number', type=int, required=True, help='Data file number identifier')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--hidden_units', type=int, default=10)
    parser.add_argument('--num_heads', type=int, default=6)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--fixed_test_index', type=str, default='fixed_test_index.npy')
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.data_path)
    seq_length = [len(i) for i in df.seq.values]

    embeddings = np.load(args.embedding_path)
    embeddings_list = []
    start_idx = 0
    for length in seq_length:
        end_idx = start_idx + length 
        embeddings_list.append(embeddings[start_idx:end_idx, :])
        start_idx = end_idx

    if not os.path.exists(args.fixed_test_index):
        labels = df['label']
        pos_idx = labels[labels == 1].index.tolist()
        neg_idx = labels[labels == 0].index.tolist()
        n_pos = int(len(pos_idx) * 0.2)
        n_neg = int(len(neg_idx) * 0.2)
        test_index_fixed = (
            np.random.choice(pos_idx, n_pos, replace=False).tolist() +
            np.random.choice(neg_idx, n_neg, replace=False).tolist()
        )
        np.save(args.fixed_test_index, test_index_fixed)
    else:
        test_index_fixed = np.load(args.fixed_test_index)

    x_test = graph_from_pdb(df.iloc[test_index_fixed], [embeddings_list[i] for i in test_index_fixed])
    y_test = df.iloc[test_index_fixed].label
    test_dataset = Dataset(x_test, y_test)

    train_valid_index = np.setdiff1d(np.arange(df.shape[0]), test_index_fixed)
    df_train_valid = df.iloc[train_valid_index]
    embeddings_train_valid = [embeddings_list[i] for i in train_valid_index]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(skf.split(df_train_valid, df_train_valid['label'])):
        print(f"Training fold {fold + 1}...")

        x_train = graph_from_pdb(df_train_valid.iloc[train_idx], [embeddings_train_valid[i] for i in train_idx])
        y_train = df_train_valid.iloc[train_idx].label

        x_valid = graph_from_pdb(df_train_valid.iloc[valid_idx], [embeddings_train_valid[i] for i in valid_idx])
        y_valid = df_train_valid.iloc[valid_idx].label

        train_dataset = Dataset(x_train, y_train)
        valid_dataset = Dataset(x_valid, y_valid)

        model = GraphAttentionNetwork(1280, args.hidden_units, args.num_heads, args.num_layers, args.batch_size)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

        model_path = os.path.join(args.output_dir, f'model_{args.file_number}_fold_{fold + 1}.h5')
        checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_binary_accuracy', verbose=1)
        early_stopping = EarlyStopping(monitor='val_binary_accuracy', min_delta=1e-2, patience=50, restore_best_weights=True)

        model.fit(train_dataset,
                  validation_data=valid_dataset,
                  epochs=args.epochs,
                  callbacks=[early_stopping, checkpoint],
                  verbose=2)

        model.load_weights(model_path)
        y_val_pred_prob = model.predict(valid_dataset)
        y_val_pred = (y_val_pred_prob > 0.5).astype(int)

        y_test_pred_prob = model.predict(test_dataset)
        y_test_pred = (y_test_pred_prob > 0.5).astype(int)

        fold_results.append({
            'fold': fold + 1,
            'val_accuracy': accuracy_score(y_valid, y_val_pred),
            'val_precision': precision_score(y_valid, y_val_pred),
            'val_recall': recall_score(y_valid, y_val_pred),
            'val_f1': f1_score(y_valid, y_val_pred),
            'val_auc': roc_auc_score(y_valid, y_val_pred_prob),
            'val_auprc': average_precision_score(y_valid, y_val_pred_prob),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred),
            'test_auc': roc_auc_score(y_test, y_test_pred_prob),
            'test_auprc': average_precision_score(y_test, y_test_pred_prob)
        })

    result_df = pd.DataFrame(fold_results)
    result_df.insert(0, 'file_number', args.file_number)
    result_df.to_csv(os.path.join(args.output_dir, f'results_{args.file_number}.csv'), index=False)

if __name__ == '__main__':
    main()
