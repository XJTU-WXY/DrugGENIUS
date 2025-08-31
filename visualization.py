import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

from utils.io import *

value_columns = [
    "GenerationFrequency", "MolWt", "HeavyAtomCount", "LogP", "QED", "SA",
    "TPSA", "HBD", "HBA", "RotatableBonds", "NumAromaticRings",
    "FractionCSP3", "FormalCharge", "NumAliphaticRings"
]

def get_point_size(n_samples, min_size=0.5, max_size=5):
    size = 2000 / n_samples
    size = max(min_size, min(size, max_size))
    return size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default=os.path.join(os.getcwd(), "result"), help="Path of input directory containing *_prop.json files.")
    parser.add_argument("-c", "--cmap", type=str, default="plasma", help="Colormap name for the plots.")
    parser.add_argument("-f", "--format", type=str, default="png", help="Format for the plots.")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for the plots.")
    args = parser.parse_args()
    paras = vars(args)
    about("Visualization", paras)

    report_df = pd.read_csv(os.path.join(args.input, "ligand_report.csv"))
    embeddings = np.load(os.path.join(args.input, "tsne_embeddings.npy"))
    result_dir = os.path.join(args.input, "figures")
    os.makedirs(result_dir, exist_ok=True)

    point_size = get_point_size(len(report_df))

    for col in value_columns:
        if col not in report_df.columns:
            print(f"Property '{col}' not found. Skipping...")
            continue

        print(f"Drawing plot for '{col}' ...")
        values = report_df[col].values
        plt.figure(figsize=(10, 8), dpi=args.dpi)
        sc = plt.scatter(
            embeddings[:, 0], embeddings[:, 1],
            c=values, cmap=args.cmap, s=point_size, alpha=0.8
        )
        cbar = plt.colorbar(sc)
        cbar.set_label(col, fontsize=14)

        if "Cluster" in report_df.columns:
            unique_labels = set(report_df["Cluster"])
            for label in unique_labels:
                idxs = report_df.index[report_df["Cluster"] == label].tolist()
                x_center = embeddings[idxs, 0].mean()
                y_center = embeddings[idxs, 1].mean()
                text = plt.text(
                    x_center, y_center, str(label),
                    color="black", fontsize=14, ha="center", va="center"
                )
                text.set_path_effects([
                    PathEffects.Stroke(linewidth=2, foreground="white"),
                    PathEffects.Normal()
                ])

        plt.xlabel("t-SNE Dimension 1", fontsize=14)
        plt.ylabel("t-SNE Dimension 2", fontsize=14)
        plt.tight_layout()

        plot_file_path = os.path.join(result_dir, f"figure_{col}.{args.format}")
        plt.savefig(plot_file_path)
        plt.close()

    if "MaxSimilarity" in report_df.columns:
        plt.figure(figsize=(10, 8), dpi=args.dpi)

        colors = []
        for val in report_df["MaxSimilarity"]:
            if pd.isna(val) or val == "":
                colors.append("orange")  # Novel
            else:
                colors.append("blue")  # Similar to patented

        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=point_size, alpha=0.8)

        if "Cluster" in report_df.columns:
            unique_labels = set(report_df["Cluster"])
            for label in unique_labels:
                idxs = report_df.index[report_df["Cluster"] == label].tolist()
                x_center = embeddings[idxs, 0].mean()
                y_center = embeddings[idxs, 1].mean()
                text = plt.text(
                    x_center, y_center, str(label),
                    color="black", fontsize=14, ha="center", va="center"
                )
                text.set_path_effects([
                    PathEffects.Stroke(linewidth=2, foreground="white"),
                    PathEffects.Normal()
                ])

        plt.xlabel("t-SNE Dimension 1", fontsize=14)
        plt.ylabel("t-SNE Dimension 2", fontsize=14)
        plt.tight_layout()

        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Novel',
                       markerfacecolor='orange', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Similar to patented',
                       markerfacecolor='blue', markersize=10)
        ]
        plt.legend(handles=legend_elements, loc='best', fontsize=12, frameon=True)

        plot_file_path = os.path.join(result_dir, f"figure_Novelty.{args.format}")
        plt.savefig(plot_file_path)
        plt.close()
    else:
        print("Column 'MaxSimilarity' not found. Skipping...")

if __name__ == "__main__":
    main()