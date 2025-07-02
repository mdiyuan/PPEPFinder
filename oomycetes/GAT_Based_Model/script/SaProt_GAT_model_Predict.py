import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
from GraphFromPDB import graph_from_pdb, Dataset
from Model import GraphAttentionNetwork

def predict_saprot_gat(csv_file, npy_file, output_file):
    warnings.filterwarnings("ignore")
    np.random.seed(42)
    tf.random.set_seed(42)

    model_paths = [
        "./oomycetes/GAT_Based_Model/saprot_GAT_kf_model/SaProt_GAT_distance_6_fold_1.h5",
        "./oomycetes/GAT_Based_Model/saprot_GAT_kf_model/SaProt_GAT_distance_6_fold_2.h5",
        "./oomycetes/GAT_Based_Model/saprot_GAT_kf_model/SaProt_GAT_distance_6_fold_3.h5",
        "./oomycetes/GAT_Based_Model/saprot_GAT_kf_model/SaProt_GAT_distance_6_fold_4.h5",
        "./oomycetes/GAT_Based_Model/saprot_GAT_kf_model/SaProt_GAT_distance_6_fold_5.h5"
    ]


    df = pd.read_csv(csv_file)
    embeddings = np.load(npy_file)  

    x_pred = graph_from_pdb(df, [embeddings])  
    y_pred = df.label
    pred_dataset = Dataset(x_pred, y_pred)

    HIDDEN_UNITS = 10
    NUM_HEADS = 6
    NUM_LAYERS = 1
    BATCH_SIZE = 1  

    
    all_predictions = []

    for model_path in model_paths:
        gat_model = GraphAttentionNetwork(1280, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, BATCH_SIZE)
        gat_model.load_weights(model_path)
        predictions = gat_model.predict(pred_dataset)
        all_predictions.append(predictions)

    final_prediction = np.mean(all_predictions, axis=0).squeeze()

    protein_ids = df.iloc[:, 0]  

    is_effector = np.where(final_prediction > 0.5, "eff", "Non")  

    effector_prob = np.round(final_prediction, 2)
    non_effector_prob = np.round(1 - final_prediction, 2)

    df_results = pd.DataFrame({
        "Name": protein_ids,
        "Effector.prob": effector_prob,
        "Is Effector(prob>0.5)": is_effector  
    })

    df_results.to_csv(output_file, index=False)

    print(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--npy_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    predict_saprot_gat(args.csv_file, args.npy_file, args.output_file)