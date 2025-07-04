import os

import argparse
from multiprocessing import cpu_count

from tqdm import tqdm
from FPSim2 import FPSim2Engine
import numpy as np

from utils.io import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(os.getcwd(), "result", "generated_ligands"), help="Path of input directory containing *_prop.json files.")
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), "result", "novelty_checking"), help='Path of output directory for result files.')
    parser.add_argument("-t", "--threshold", type=float, default=0.6, help="Tanimoto similarity threshold.")
    parser.add_argument("-f", "--fp_database", type=str, default="fpsim2_fingerprints.h5", help="Path of FPSim2 reference database.")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Path of FPSim2 reference database.")
    parser.add_argument("--threads", type=int, default=cpu_count(), help="Number of threads.")
    args = parser.parse_args()
    paras = vars(args)
    about("Novelty Checking", paras)

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

    files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith(".json")]
    for f in tqdm(files, desc="Checking similarities"):
        prop = load_json(f)
        if args.device == "cpu":
            sims = fpe.similarity(prop["SMILES"], threshold=args.threshold, metric='tanimoto', n_workers=args.threads)
        else:
            sims = fpe.similarity(prop["SMILES"], threshold=args.threshold)
        np.savetxt(os.path.join(args.output, f"{prop['Hash']}.csv"), sims, delimiter=",", header="ID,Similarity", comments='', fmt = ['%s', '%.8f'])

if __name__ == "__main__":
    main()