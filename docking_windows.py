from typing import List, Union

import os
import subprocess
import argparse
import warnings
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from meeko import MoleculePreparation
from meeko import PDBQTWriterLegacy
from rdkit import Chem

from utils.io import *

warnings.filterwarnings('ignore')

def sdf_to_pdbqt(args):
    sdf_path, pdbqt_path, filter = args
    if os.path.exists(pdbqt_path):
        return pdbqt_path
    mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
    mk_prep = MoleculePreparation()
    molsetup = mk_prep(mol)[0]
    pdbqt_string = PDBQTWriterLegacy.write_string(molsetup)
    if pdbqt_string[1]:
        with open(pdbqt_path, "w") as f:
            f.write(pdbqt_string[0])
        return pdbqt_path
    else:
        return None

def batch_sdf_to_pdbqt(sdf_list: List[str], pdbqt_dir: str, filter_dict: Union[dict, None], num_processes: int):
    with Pool(processes=num_processes) as pool:
        args = [(sdf, os.path.join(pdbqt_dir, os.path.basename(sdf).replace(".sdf", ".pdbqt")), None) for sdf in sdf_list]
        results = list(tqdm(pool.imap_unordered(sdf_to_pdbqt, args), total=len(sdf_list), desc="Preparing ligands"))
    return [_ for _ in results if _ is not None]

def dock(args):
    receptor, ligand, config, exhaustiveness, out = args
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
        except:
            pass

def batch_dock(receptor, pdbqt_list, config, exhaustiveness, result_dir, num_processes: int):
    with Pool(processes=num_processes) as pool:
        args = [(receptor, pdbqt, config, str(exhaustiveness), os.path.join(result_dir, os.path.basename(pdbqt))) for pdbqt in pdbqt_list]
        list(tqdm(pool.imap_unordered(dock, args), total=len(pdbqt_list), desc="Docking"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", '--protein', type=str, help='Path of the pdbqt file of target protein.', required=True)
    parser.add_argument("-b", '--box', type=str, help='Path of the txt file of grid box.', required=True)
    parser.add_argument('-i', '--input', type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument("-e", '--exhaustiveness', type=int, default=8, help='Exhaustiveness for Autodock Vina.')
    parser.add_argument('-l', '--list', type=str, default=os.path.join(os.getcwd(), "filter_docking.yaml"), help='Path of a txt file containing a list of hashes.')
    parser.add_argument('--threads', type=int, default=cpu_count(), help='Number of threads.')
    args = parser.parse_args()
    paras = vars(args)
    about("Docking", paras)

    ligand_dir = os.path.join(args.input, "ligands")
    pdbqt_dir = os.path.join(args.input, "docking_pdbqt_cache")
    result_dir = os.path.join(args.input, "docking_result")

    os.makedirs(pdbqt_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    # if args.filter.endswith(".txt"):
    #     with open(args.filter, "r") as f:
    #         hash_list = [_.strip() for _ in f.readlines()]
    #         sdf_files = [os.path.join(ligand_dir, f) for f in os.listdir(ligand_dir) if f.endswith(".sdf")]

    sdf_list = [os.path.join(ligand_dir, f) for f in os.listdir(ligand_dir) if f.endswith(".sdf")]

    pdbqt_list = batch_sdf_to_pdbqt(sdf_list, pdbqt_dir, None, args.threads)

    batch_dock(args.protein, pdbqt_list, args.box, args.exhaustiveness, result_dir, args.threads)

if __name__ == "__main__":
    mp.freeze_support()
    main()











