from typing import List, Union, Tuple

import os
import subprocess
import argparse
import warnings
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem

from utils.io import *
from utils.postprocess import passes_lipinski

warnings.filterwarnings('ignore')


def filter_report_df(df, filter) -> pd.DataFrame:
    df.fillna({"MaxSimilarity": 0}, inplace=True)

    for prop, bounds in filter.get("Properties", {}).items():
        min_val = bounds.get("Min", None)
        max_val = bounds.get("Max", None)
        if min_val is not None:
            df = df[df[prop] >= min_val]
        if max_val is not None:
            df = df[df[prop] <= max_val]

    for field in ["GenerationFrequency", "MaxSimilarity", "PredictedAffinity"]:
        bounds = filter.get(field, {})
        if not bounds:
            continue
        min_val = bounds.get("Min", None)
        max_val = bounds.get("Max", None)
        if min_val is not None:
            df = df[df[field] >= min_val]
        if max_val is not None:
            df = df[df[field] <= max_val]

    lipinski_flag = filter.get("Lipinski", False)
    if lipinski_flag:
        df = df[df.apply(lambda row: passes_lipinski(row), axis=1)]

    cluster_val = filter.get("Cluster", None)
    if cluster_val is not None:
        if isinstance(cluster_val, list):
            df = df[df["Cluster"].isin(cluster_val)]
        else:
            df = df[df["Cluster"] == cluster_val]
    return df.reset_index(drop=True)


def sdf_to_pdbqt(args):
    # import numpy as np
    # np.seterr(all="ignore")
    idx, sdf_path, pdbqt_path = args
    if os.path.exists(pdbqt_path):
        return idx, pdbqt_path
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mk_prep = MoleculePreparation()
    molsetup = mk_prep(mol)[0]
    pdbqt_string = PDBQTWriterLegacy.write_string(molsetup)
    if pdbqt_string[1]:
        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string[0])
        return idx, pdbqt_path
    else:
        return idx, None

def batch_sdf_to_pdbqt(sdf_list, pdbqt_dir: str, num_processes: int):
    with Pool(processes=num_processes) as pool:
        args = [(sdf[0], sdf[1], os.path.join(pdbqt_dir, os.path.basename(sdf[1]).replace(".sdf", ".pdbqt"))) for sdf in sdf_list]
        results = list(tqdm(pool.imap_unordered(sdf_to_pdbqt, args), total=len(sdf_list), desc="Preparing ligands"))
    return [_ for _ in results if _[1] is not None]

def dock(args):
    idx, receptor, ligand, config, exhaustiveness, out = args
    if not os.path.exists(out):
        command = [
            "vina",
            "--receptor", receptor,
            "--ligand", ligand,
            "--config", config,
            "--exhaustiveness", exhaustiveness,
            "--out", out
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(out):
                return idx, read_docking_result(out)
            else:
                return idx, None
        except:
            return idx, None
    else:
        return idx, read_docking_result(out)

def batch_dock(receptor, pdbqt_list, config, exhaustiveness, result_dir, num_processes: int):
    with Pool(processes=num_processes) as pool:
        args = [(pdbqt[0], receptor, pdbqt[1], config, str(exhaustiveness), os.path.join(result_dir, os.path.basename(pdbqt[1]))) for pdbqt in pdbqt_list]
        results = list(tqdm(pool.imap_unordered(dock, args), total=len(pdbqt_list), desc="Docking"))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--protein', type=str, help='Path of the pdbqt file of target protein.', required=True)
    parser.add_argument("-b", '--box', type=str, help='Path of the txt file of grid box.', required=True)
    parser.add_argument('-i', '--input', type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument('-f', '--filter', type=str, default=os.path.join(os.getcwd(), "filter_docking.yaml"), help='Path of filter config file.')
    parser.add_argument("-e", '--exhaustiveness', type=int, default=8, help='Exhaustiveness for Autodock Vina.')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads.')
    args = parser.parse_args()
    paras = vars(args)
    about("Docking", paras)

    report_file_path = os.path.join(args.input, "ligand_report.csv")
    report_df = pd.read_csv(report_file_path)
    print(f"Found {len(report_df)} ligands.")
    report_df_filtered = filter_report_df(report_df, load_yaml(args.filter))
    print(f"After filtering, {len(report_df_filtered)} ligands retained.")

    ligand_dir = os.path.join(args.input, "ligands")
    pdbqt_dir = os.path.join(args.input, "docking_pdbqt_cache")
    result_dir = os.path.join(args.input, "docking_result")

    os.makedirs(pdbqt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    sdf_list = [(idx, os.path.join(ligand_dir, row["Hash"] + ".sdf")) for idx, row in report_df_filtered.iterrows()]

    pdbqt_list = batch_sdf_to_pdbqt(sdf_list, pdbqt_dir, args.threads)

    docking_results = batch_dock(args.protein, pdbqt_list, args.box, args.exhaustiveness, result_dir, args.threads)

    report_df_filtered["VinaScore"] = None

    for idx, docking_score in docking_results:
        report_df_filtered.at[idx, "VinaScore"] = docking_score

    report_df_filtered.to_csv(os.path.join(args.input, "ligand_report_docked.csv"), index=False)
    print(f"Docking report saved to {os.path.join(args.input, 'ligand_report_docked.csv')}")

if __name__ == "__main__":
    mp.freeze_support()
    main()











