from typing import List

import argparse
import time
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import torch
from tqdm import tqdm

from utils import model
from utils.postprocess import *
from utils.io import *


def _smiles_to_sdf_worker(args):
    smiles, output_dir, filter_dict, em_iters = args
    return smiles_to_sdf(smiles, output_dir, filter_dict, em_iters)

def batch_smiles_to_sdf(smiles_list: List[str], output_dir: str, filter_dict: Union[dict, None], em_iters: int, num_pp_processes: int):
    with Pool(processes=num_pp_processes) as pool:
        args = [(smi, output_dir, filter_dict, em_iters) for smi in smiles_list]
        results = pool.map(_smiles_to_sdf_worker, args)
    return results

def generator_process(queue: mp.Queue, generate_model, target_seq, max_queue_len: int, model_kwargs):
    while True:
        if queue.qsize() < max_queue_len:
            smiles_batch = generate_model.generate(target_seq, **model_kwargs)
            queue.put(smiles_batch)
        else:
            time.sleep(0.5)

def pp_process(queue: mp.Queue, output_dir: str, filter_dict: Union[dict, None], em_iters: int, total_num: int,
               counter: mp.Value, num_pp_processes: int):
    pbar = tqdm(total=total_num, desc="Generating ligands")
    while True:
        queue_len = queue.qsize()
        pbar.set_description(f"Generating ligands | Queue size: {queue_len}")
        if not queue.empty():
            smiles_batch = queue.get()
            results = batch_smiles_to_sdf(
                smiles_batch,
                output_dir=output_dir,
                filter_dict=filter_dict,
                em_iters=em_iters,
                num_pp_processes=num_pp_processes
            )
            success_count = sum([1 for r in results if r is not None])
            with counter.get_lock():
                counter.value += success_count
                pbar.update(success_count)

            if counter.value >= total_num:
                break
        else:
            time.sleep(0.5)
    pbar.close()

def run_pipeline(generate_model, target_seq: str, model_kwargs: dict,
                 output_dir: str, total_num: int,
                 filter_dict: Union[dict, None], em_iters: int,
                 max_queue_len: int, num_pp_proc: int):
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(maxsize=max_queue_len)
    counter = mp.Value('i', 0)

    gen_proc = mp.Process(target=generator_process, args=(queue, generate_model, target_seq, max_queue_len, model_kwargs))
    pp_proc = mp.Process(target=pp_process, args=(queue, output_dir, filter_dict, em_iters, total_num, counter, num_pp_proc))

    gen_proc.start()
    pp_proc.start()

    pp_proc.join()
    gen_proc.terminate()
    torch.cuda.empty_cache()

def about(args):
    print("""

    ____                   _____________   ________  _______
   / __ \_______  ______ _/ ____/ ____/ | / /  _/ / / / ___/
  / / / / ___/ / / / __ `/ / __/ __/ /  |/ // // / / /\__ \ 
 / /_/ / /  / /_/ / /_/ / /_/ / /___/ /|  // // /_/ /___/ / 
/_____/_/   \__,_/\__, /\____/_____/_/ |_/___/\____//____/  
                 /____/                                     
 An All-in-One Framework for Sequence-based Ligand Design
   Licensed under GNU GPLv3     XJTU-WXY @ FISSION Lab
======================= Parameters =========================
    """)
    print(args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input', type=str, help='Path of FASTA file or amino acid string of target protein.', required=True)
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), "result", "generated_ligands"), help='Path of directory for generated sdf files of ligands.')
    parser.add_argument('-m', '--model', type=str, default="DrugGPT", help='Model to use for generation.')
    parser.add_argument('-n', '--total_num', type=int, default=1000, help='Total number of ligands to generate.')
    parser.add_argument('-f', '--filter', type=str, default=os.path.join(os.getcwd(), "filter_generate.yaml"), help='Path of filter config file.')
    parser.add_argument('-d', '--device', type=str, default="cuda", help='Device to use.')
    parser.add_argument('--pp_proc', type=int, default=cpu_count(), help='Number of post-processing parallel processes.')
    parser.add_argument('--em_iters', type=int, default=10000, help='Max number of iterations for energy minimization.')
    parser.add_argument('--queue_len', type=int, default=100, help='Maximum length of the cache queue.')
    args, unknown_args = parser.parse_known_args()
    model_kwargs = parse_extra_args(unknown_args)
    paras = vars(args)
    paras.update(model_kwargs)
    about(paras)

    if os.path.exists(args.input):
        target_seq = read_fasta_file(args.input)
    else:
        target_seq = args.input

    generate_model = getattr(model, args.model)(device=args.device)

    os.makedirs(args.output, exist_ok=True)

    run_pipeline(
        generate_model=generate_model,
        target_seq=target_seq,
        model_kwargs=model_kwargs,
        output_dir=args.output,
        total_num=args.total_num,
        filter_dict=load_yaml(args.filter),
        em_iters=args.em_iters,
        max_queue_len=args.queue_len,
        num_pp_proc=args.pp_proc
    )

if __name__ == "__main__":
    main()