import argparse
import os
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from utils.io import *

def read_single_file(args):
    return load_json(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input', type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument('--threads', type=int, default=cpu_count(), help='Number of reading threads.')
    args = parser.parse_args()
    paras = vars(args)
    about("Report", paras)

    input_dir = os.path.join(args.input, "ligands")
    if not os.path.exists(input_dir):
        raise FileNotFoundError("No such input directory: '{}'".format(input_dir))

    tasks = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".json")]

    print(f"Found {len(tasks)} ligands.")

    with Pool(args.threads) as pool:
        results = list(tqdm(pool.imap(read_single_file, tasks), total=len(tasks), desc="Reading properties of ligands"))

    report_file_path = os.path.join(args.input, "ligand_report.csv")
    pd.DataFrame(results).to_csv(report_file_path, index=False)

    print(f"Report saved to '{report_file_path}'")

if __name__ == "__main__":
    main()