import argparse
import os

from tqdm import tqdm
import pandas as pd

from model import affinity_predictor
from utils.io import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument('-m', '--model', type=str, default="TransformerCPI2", help='Model to use for prediction.')
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help='Number of molecules predicted per batch')
    parser.add_argument('-d', '--device', type=str, default="cuda", help='Device to use.')
    args = parser.parse_args()
    paras = vars(args)
    about("Affinity Prediction", paras)

    with open(os.path.join(args.input, "target_seq.txt"), "r") as f:
        target_seq = f.read().strip()

    predict_model = getattr(affinity_predictor, args.model)(device=args.device)

    report_file_path = os.path.join(args.input, "ligand_report.csv")
    report_df = pd.read_csv(report_file_path)
    smiles_list = report_df["SMILES"].tolist()
    scores = []

    for i in tqdm(range(0, len(smiles_list), args.batch_size), desc="Predicting affinity"):
        score = predict_model.predict(target_seq, smiles_list[i : i+args.batch_size])
        scores.extend(score)

    report_df["PredictedAffinity"] = scores

    report_df.to_csv(report_file_path, index=False)

if __name__ == "__main__":
    main()