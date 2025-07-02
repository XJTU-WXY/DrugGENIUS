import os
import warnings
import argparse
import multiprocessing as mp
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.lines as mlines
from sklearn.manifold import TSNE
import networkx as nx
import community as community_louvain  # aka python-louvain
from sklearn.neighbors import NearestNeighbors

from utils.io import *
from utils.postprocess import filter_mol_by_prop

morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_morgan_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    fp = morgan_generator.GetFingerprint(mol)
    arr = np.zeros((1,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def process_single_file(args):
    file_path, output_dir, filter_dict, no_cache = args

    data = load_json(file_path)

    if not filter_mol_by_prop(data, filter_dict):
        return None

    ligand_hash = data["Hash"]
    smiles = data["SMILES"]
    fp_path = os.path.join(output_dir, f"{ligand_hash}_fp.npy")
    sim_path = os.path.join(output_dir, f"{ligand_hash}_sim.csv")

    # If cached fingerprint exists, load it
    if os.path.exists(fp_path) and not no_cache:
        fp = np.load(fp_path)
    else:
        fp = compute_morgan_fp(smiles)
        np.save(fp_path, fp)

    # Similarity check
    if os.path.exists(sim_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sims = np.loadtxt(sim_path, delimiter=",", skiprows=1)
        is_novel = True if sims.size == 0 else False
    else:
        is_novel = None

    return {
        "prop": data,
        "fp": fp,
        "is_novel": is_novel,
    }

def louvain_clustering(embeddings, k=10, metric='euclidean', seed=42, n_jobs=-1):
    print("Building kNN graph ...")
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    G = nx.Graph()
    for i in range(len(embeddings)):
        for j_idx in range(1, k + 1):
            j = indices[i][j_idx]
            dist = distances[i][j_idx]
            sim = 1 / (1 + dist)
            G.add_edge(i, j, weight=sim)

    print("Running Louvain community detection ...")
    partition = community_louvain.best_partition(G, random_state=seed)
    labels = [partition.get(i, -1) for i in range(len(embeddings))]
    return labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(os.getcwd(), "result", "generated_ligands"), help="Path of input directory containing *_prop.json files.")
    parser.add_argument("-o", "--output", type=str, default=os.path.join(os.getcwd(), "result", "cluster_report"), help="Path of output directory for report files.")
    parser.add_argument("-k", "--k_neighbors", type=float, default=10, help="Number of nearest neighbors to use in clustering.")
    parser.add_argument('-f', '--filter', type=str, default=os.path.join(os.getcwd(), "filter_cluster.yaml"), help='Path of filter config file.')
    parser.add_argument("--no_cache", action='store_true', default=False, help="Force refreshing the cache of fps and t-SNE results.")
    parser.add_argument("--proc", type=int, default=cpu_count(), help="Number of parallel processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for t-SNE.")
    args = parser.parse_args()
    paras = vars(args)
    about("Clustering", paras)

    os.makedirs(args.output, exist_ok=True)

    filter_dict = load_yaml(args.filter)

    files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.endswith("_prop.json")]
    tasks = [(f, args.input, filter_dict, args.no_cache) for f in files]

    print(f"Found {len(tasks)} ligands.")

    with mp.Pool(args.proc) as pool:
        results = list(tqdm(pool.imap(process_single_file, tasks), total=len(tasks), desc="Extracting fingerprints"))

    results = [_ for _ in results if _ is not None]

    print(f"{len(results)} ligands remain after filtering.")

    fps = np.array([r["fp"] for r in results])
    props = [r["prop"] for r in results]
    novelty_flags = [r["is_novel"] for r in results]

    # Clustering (t-SNE + Louvain)
    embedding_path = os.path.join(args.output, "tsne_embeddings.npy")
    if os.path.exists(embedding_path) and not args.no_cache:
        print("Loading cached t-SNE embeddings ...")
        embeddings = np.load(embedding_path)
    else:
        print("Running t-SNE dimensionality reduction ...")
        tsne = TSNE(n_components=2, random_state=args.seed, n_jobs=args.proc)
        embeddings = tsne.fit_transform(fps)
        np.save(embedding_path, embeddings)

    print("Running clustering ...")

    labels = louvain_clustering(embeddings, k=args.resolution, metric='euclidean', seed=args.seed, n_jobs=args.proc)

    # Save cluster info per molecule
    print("Saving cluster report ...")
    cluster_report = []
    for i, prop in enumerate(props):
        prop.update({"Novelty": novelty_flags[i], "Cluster": int(labels[i])})
        cluster_report.append(prop)
    pd.DataFrame(cluster_report).to_csv(os.path.join(args.output, "cluster_report.csv"), index=False)

    # Plot 1: colored by cluster
    print("Saving cluster plots ...")
    plt.figure(figsize=(10,8), dpi=1000)
    unique_labels = set(labels)
    for label in unique_labels:
        idxs = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(embeddings[idxs, 0], embeddings[idxs, 1], label=f"Cluster {label}", s=0.05)
        x_center = embeddings[idxs, 0].mean()
        y_center = embeddings[idxs, 1].mean()
        text = plt.text(
            x_center, y_center, str(label),
            color='black', fontsize=14, ha='center', va='center'
        )
        text.set_path_effects([
            PathEffects.Stroke(linewidth=2, foreground='white'),
            PathEffects.Normal()
        ])

    plt.title("Clusters of Ligands", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "clusters.pdf"))
    plt.close()

    # Plot 2: colored by novelty
    plt.figure(figsize=(10,8), dpi=1000)
    novel_idxs = [i for i, flag in enumerate(novelty_flags) if flag == True]
    similar_idxs = [i for i, flag in enumerate(novelty_flags) if flag == False]
    unknown_idxs = [i for i, flag in enumerate(novelty_flags) if flag is None]
    # Novel group
    plt.scatter(embeddings[novel_idxs, 0], embeddings[novel_idxs, 1], color="indianred", label="Novel", s=0.05, alpha=0.7)
    # Similar group
    plt.scatter(embeddings[similar_idxs, 0], embeddings[similar_idxs, 1], color="royalblue", label="Similar to patented", s=0.05, alpha=0.7)

    plt.scatter(embeddings[unknown_idxs, 0], embeddings[unknown_idxs, 1], color="grey", label="Unknown", s=0.05, alpha=0.7)

    legend_handles = [
        mlines.Line2D([], [], color='indianred', marker='o', linestyle='None',
                      markersize=6, label='Novel'),
        mlines.Line2D([], [], color='royalblue', marker='o', linestyle='None',
                      markersize=6, label='Similar to patented'),
        mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                      markersize=6, label='Unknown'),
    ]

    plt.title("Novelty of Ligands", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(handles=legend_handles, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "novelty.pdf"))
    plt.close()


if __name__ == "__main__":
    main()