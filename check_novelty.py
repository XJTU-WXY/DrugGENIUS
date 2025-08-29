import os

import argparse
from multiprocessing import cpu_count

from tqdm import tqdm
from FPSim2 import FPSim2Engine
import numpy as np
import pandas as pd

from utils.io import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Tanimoto similarity threshold.")
    parser.add_argument("-f", "--fp_database", type=str, default="fpsim2_fingerprints.h5", help="Path of FPSim2 reference database.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Path of FPSim2 reference database.")
    parser.add_argument("--threads", type=int, default=cpu_count(), help="Number of threads.")
    args = parser.parse_args()
    paras = vars(args)
    about("Novelty Checking", paras)

    report_file_path = os.path.join(args.input, "ligand_report.csv")
    report_df = pd.read_csv(report_file_path)
    result_dir = os.path.join(args.input, "novelty_checking")
    os.makedirs(result_dir, exist_ok=True)

    print("Loading patented molecules fingerprint database ...")
    if args.device == "cpu":
        fpe = FPSim2Engine(args.fp_database)
    elif args.device == "cuda":
        try:
            from FPSim2 import FPSim2CudaEngine
            fpe = FPSim2CudaEngine(args.fp_database)
        except ImportError:
            raise Exception("FPSim2CudaEngine not available, maybe CuPy is not installed.")
    elif args.device.startswith("cuda:"):
        try:
            from FPSim2 import FPSim2CudaEngine
            import cupy as cp
            cuda_id = int(args.device.split(":")[1])
            cp.cuda.Device(cuda_id).use()
            fpe = FPSim2CudaEngine(args.fp_database)
        except ImportError:
            raise Exception("FPSim2CudaEngine not available, maybe CuPy is not installed.")
    else:
        raise Exception("Unknown device.")

    report_df["MaxSimilarity"] = None
    report_df["MostSimilarMolID"] = None

    for idx, row in tqdm(list(report_df.iterrows()), desc="Checking similarities"):
        if args.device == "cpu":
            sims = fpe.similarity(row["SMILES"], threshold=args.threshold, metric='tanimoto', n_workers=args.threads)
        else:
            sims = fpe.similarity(row["SMILES"], threshold=args.threshold)
        if sims.size > 0:
            sims_2d = np.array(sims.tolist(), dtype=float)
            max_idx = np.argmax(sims_2d[:, 1].astype(float))
            most_similar_id = sims_2d[max_idx, 0]
            max_similarity = float(sims_2d[max_idx, 1])

            report_df.at[idx, "MostSimilarMolID"] = most_similar_id
            report_df.at[idx, "MaxSimilarity"] = max_similarity
        np.savetxt(os.path.join(result_dir, f"{row['Hash']}.csv"), sims, delimiter=",", header="ID,Similarity", comments='', fmt = ['%s', '%.8f'])

    report_df.to_csv(report_file_path, index=False)

if __name__ == "__main__":
    main()