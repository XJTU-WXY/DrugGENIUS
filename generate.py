from typing import List

import warnings
import argparse
import time
import multiprocessing as mp
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from tqdm import tqdm
from rdkit.rdBase import BlockLogs

from model import ligand_generator
from utils.postprocess import *
from utils.io import *

warnings.filterwarnings('ignore')

def _smiles_to_sdf_worker(args):
    smiles, output_dir, filter_dict, em_iters, record_raw_output = args
    return smiles_to_sdf(smiles, output_dir, filter_dict, em_iters, record_raw_output)

def batch_smiles_to_sdf(smiles_list: List[str], output_dir: str, filter_dict: Union[dict, None], em_iters: int, num_pp_threads: int, record_raw_output: bool):
    args_list = [(smi, output_dir, filter_dict, em_iters, record_raw_output) for smi in smiles_list]
    results = []

    with ThreadPoolExecutor(max_workers=num_pp_threads) as executor:
        futures = [executor.submit(_smiles_to_sdf_worker, args) for args in args_list]
        for future in as_completed(futures):
            results.append(future.result())

    return results

def generator_process(queue: mp.Queue, generate_model, device, target_seq, max_queue_len: int, model_kwargs: dict, init_seed: Union[int, None]):
    current_seed = init_seed
    generate_model = getattr(ligand_generator, generate_model)(device=device)
    while True:
        if queue.qsize() < max_queue_len:
            if current_seed is not None:
                current_seed += 1
            smiles_batch = generate_model.generate(target_seq, current_seed, **model_kwargs)
            queue.put(smiles_batch)
        else:
            time.sleep(0.5)

def pp_process(queue: mp.Queue, output_dir: str, filter_dict: Union[dict, None], em_iters: int, total_num: int,
               counter: mp.Value, num_pp_threads: int, record_raw_output: bool):
    current_batch = 1
    pbar = tqdm(total=total_num, desc="Generating ligands", position=0)
    while True:
        queue_len = queue.qsize()
        pbar.set_postfix({"Queue": queue_len, "Current batch": current_batch})
        if not queue.empty():
            smiles_batch = queue.get()
            results = batch_smiles_to_sdf(
                smiles_batch,
                output_dir=output_dir,
                filter_dict=filter_dict,
                em_iters=em_iters,
                num_pp_threads=num_pp_threads,
                record_raw_output=record_raw_output
            )
            success_count = sum([1 for r in results if r == 0])
            duplicate = [r for r in results if r]
            for m in duplicate:
                meta_data = load_json(m)
                meta_data["GenerationFrequency"] += 1
                save_json(meta_data, m)

            with counter.get_lock():
                counter.value += success_count
                pbar.update(success_count)

            if counter.value >= total_num:
                break
            current_batch += 1
        else:
            time.sleep(0.5)
    pbar.close()

def run_pipeline(generate_model, device: str, target_seq: str, model_kwargs: dict,
                 output_dir: str, total_num: int,
                 filter_dict: Union[dict, None], em_iters: int,
                 max_queue_len: int, num_pp_threads: int, init_seed: Union[int, None],
                 record_raw_output: bool):
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue(maxsize=max_queue_len)
    counter = mp.Value('i', 0)

    gen_proc = mp.Process(target=generator_process, args=(queue, generate_model, device, target_seq, max_queue_len, model_kwargs, init_seed))
    pp_proc = mp.Process(target=pp_process, args=(queue, output_dir, filter_dict, em_iters, total_num, counter, num_pp_threads, record_raw_output))

    gen_proc.start()
    pp_proc.start()

    try:
        pp_proc.join()
        gen_proc.terminate()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Terminating processes...")
        gen_proc.terminate()
        pp_proc.terminate()
    finally:
        queue.close()
        queue.join_thread()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--input', type=str, help='Path of FASTA file or amino acid string of target protein.', required=True)
    parser.add_argument('-o', '--output', type=str, default=os.path.join(os.getcwd(), "result"), help='Path of the project directory.')
    parser.add_argument('-m', '--model', type=str, default="DrugGPT", help='Model to use for generation.')
    parser.add_argument('-n', '--total_num', type=int, default=10000, help='Total number of ligands to generate.')
    parser.add_argument('-f', '--filter', type=str, default=os.path.join(os.getcwd(), "filter_generate.yaml"), help='Path of filter config file.')
    parser.add_argument('-d', '--device', type=str, default="cuda", help='Device to use.')
    parser.add_argument('--threads', type=int, default=cpu_count(), help='Number of post-processing threads.')
    parser.add_argument('--em_iters', type=int, default=10000, help='Max number of iterations for energy minimization.')
    parser.add_argument('--queue_len', type=int, default=20, help='Maximum length of the cache queue.')
    parser.add_argument('--init_seed', type=int, default=None, help='The initial random seed for result reproducibility. Each generated batch will increase the seed by one. If not specified, current timestamp will be used as random seed for each batch.')
    parser.add_argument('--record_raw_output', action='store_true', default=False, help='For research purposes, record the raw SMILES string output by the ligand generation model in the json file')
    args, unknown_args = parser.parse_known_args()
    model_kwargs = parse_extra_args(unknown_args)
    paras = vars(args)
    paras.update(model_kwargs)
    about("Generation", paras)

    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.input):
        target_seq = read_fasta_file(args.input)
    else:
        target_seq = args.input
    print(f"Target: {target_seq}")

    with open(os.path.join(args.output, "target_seq.txt"), "w") as f:
        f.write(target_seq)

    output_dir = os.path.join(args.output, "ligands")
    os.makedirs(output_dir, exist_ok=True)

    save_json(paras, os.path.join(args.output, "generation_parameters.json"))

    run_pipeline(
        generate_model=args.model,
        device=args.device,
        target_seq=target_seq,
        model_kwargs=model_kwargs,
        output_dir=output_dir,
        total_num=args.total_num,
        filter_dict=load_yaml(args.filter),
        em_iters=args.em_iters,
        max_queue_len=args.queue_len,
        num_pp_threads=args.threads,
        init_seed=args.init_seed,
        record_raw_output=args.record_raw_output
    )

if __name__ == "__main__":
    mp.freeze_support()
    main()