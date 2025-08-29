from typing import List

import argparse
import warnings
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from utils.postprocess import *
from utils.io import *

warnings.filterwarnings('ignore')

def _smiles_to_sdf_worker(args):
    smiles, output_dir, filter_dict, em_iters = args
    return smiles_to_sdf(smiles, output_dir, filter_dict, em_iters)

def batch_smiles_to_sdf(smiles_list: List[str], output_dir: str, filter_dict: Union[dict, None], em_iters: int, num_pp_processes: int):
    with Pool(processes=num_pp_processes) as pool:
        args = [(smi, output_dir, filter_dict, em_iters) for smi in smiles_list]
        results = list(tqdm(pool.imap_unordered(_smiles_to_sdf_worker, args), total=len(smiles_list), desc="Post-processing"))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input', type=str, help='Path of a txt file containing existed SMILES.', required=True)
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument('-f', '--filter', type=str, default=os.path.join(os.getcwd(), "filter_generate.yaml"), help='Path of filter config file.')
    parser.add_argument('--em_iters', type=int, default=10000, help='Max number of iterations for energy minimization.')
    parser.add_argument('--threads', type=int, default=cpu_count(), help='Number of post-processing threads.')
    args = parser.parse_args()
    paras = vars(args)
    about("Post-process Existed SMILES", paras)

    with open(args.input, 'r') as f:
        smiles_list = [_.strip() for _ in f.readlines() if _]

    output_dir = os.path.join(args.output, "ligands")
    os.makedirs(output_dir, exist_ok=True)

    results = batch_smiles_to_sdf(smiles_list=smiles_list,
                                  output_dir=output_dir,
                                  filter_dict=load_yaml(args.filter),
                                  em_iters=args.em_iters,
                                  num_pp_processes=args.threads)

    print(f"{sum(1 for r in results if r == 0)}/{len(results)} SMILES were retained after post-processing.")

if __name__ == "__main__":
    mp.freeze_support()
    main()