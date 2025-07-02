<div class="title" align=center>
    <img src="https://github.com/XJTU-WXY/DrugGENIUS/blob/master/doc/logo.png">
    <br><br>
    <p>
        <img src="https://img.shields.io/github/license/XJTU-WXY/DrugGENIUS">
    	<img src="https://img.shields.io/badge/python-≥3.9-blue">
        <img src="https://img.shields.io/github/stars/XJTU-WXY/DrugGENIUS?style=social">
    </p>
    <div>An All-in-One Framework for Sequence-based Ligand Design</div>
</div>

## 🛠 THIS PROJECT IS STILL UNDER DEVELOPMENT
Currently only limited functions are implemented, including generation, molecular properties calculation, filtering, novelty checking, and clustering. It may contain bugs and should be used at your own risk.
## 🚩 Introduction
Sequence-based deep learning models for ligand design have recently garnered increasing interests in the research community. These end-to-end approaches, which generate ligand SMILES directly from protein amino acid sequences, not only enable access to a broader and more readily available training dataset but also allow the model to move beyond the constraints of binding pocket structures and leverage the richer information embedded in the primary sequence.

This project aims to provide an all-in-one, cross-platform, and efficient framework for sequence-based ligand design, streamlining the complex process into a series of simple and user-friendly steps—such as ligand molecule generation, property evaluation, filtering, novelty checking, and docking.

Currently, the project includes an optimized implementation of the DrugGPT model. By decoupling the generation and post-processing stages and eliminating the inefficient single-threaded design of the original version, the generation speed has been improved by 7–10×. Additionally, the framework introduces a set of practical tools for molecule filtering based on physicochemical properties. Support for more models is planned in future updates.

## 📥 Deployment
### Clone
```shell
git clone https://github.com/XJTU-WXY/DrugGENIUS
cd DrugGENIUS
```
### Create virtual environment (Optional, take Conda as example)
This project is developed and tested in Python 3.11, and should theoretically support all versions ≥ 3.9.
```shell
conda create -n druggenius python=3.11
conda activate druggenius
```
### Install requirements
```shell
pip install -r requirements.txt
```
🔔 Please also install the appropriate version of [PyTorch](https://pytorch.org/) according to your platform.
## 🗝 Usage
### ⚗ Ligand generation
Unlike the original implementation of DrugGPT, DrugGENIUS can automatically detect whether the input text is a raw sequence or a path to a FASTA file. It decouples molecule generation and post-processing into two parallel processes, using a shared cache queue to transfer generated molecules. Post-processing is also optimized with multi-processing, enabling simultaneous molecular property calculation, filtering, and energy minimization—significantly improving overall performance and throughput.

This step will generate ligand .sdf files and corresponding _prop.json files containing molecular physicochemical properties in output directory. Both files are named using the hash value of the SMILES string.

On the first run, the required model will be automatically downloaded from HuggingFace. Please ensure a stable internet connection and be patient during the download process.

Use `generate.py`
- Common arguments
  - `-i` | `--input`: Path of FASTA file or amino acid string of target protein. 
  - `-o` | `--output`: Path of directory for generated sdf files of ligands. Default: `./result/generated_ligands/`
  - `-m` | `--model`: Model to use for generation. Currently only supports: `DrugGPT`
  - `-n` | `--total_num`: Total number of ligands to generate. Default: `1000`
  - `-f` | `--filter`: Path of filter config file. Default: `./filter_generate.yaml`. 
    > Filtering can be applied directly during generation process so that all generated ligands meet the criteria. To customize the filtering criteria, modify the `filter_generate.yaml` file
  - `-d` | `--device`: Device to use. Default: `cuda`. 
    > In a multi-GPU environment, specify the order of GPU to be used, such as `cuda:1`. Multi-GPU distributed inference will be supported in future versions.
  - `--pp_proc`: Number of post-processing parallel processes. Default value is the total number of CPU cores on the device.
  - `--em_iters`: Max number of iterations for energy minimization. Default: `10000`
  - `--queue_len`: Maximum length of the cache queue. Default: `100`. 
  
- Arguments for `DrugGPT` model
  - `--batch_size`: The number of molecules to try to generate in each batch. Default: `16`
    > DrugGPT supports parallelization to speed up the generation, but this means more RAM usage. Adjust this value according to the actual total amount of RAM.
  - `--ligand_prompt`: The SMILES of basic scaffold. Default: `""`
    > DrugGPT supports using a predefined scaffold as the basic structure for ligand generation. See its paper for details.
  - `--top_k`: The number of highest probability tokens to be considered for top-k sampling. Default: `40`.
  - `--top_p`: The cumulative probability threshold (0.0 - 1.0) for top-p (nucleus) sampling. It defines the minimum subset of tokens to consider for random sampling. Default: `0.9`.
  - `--temp`: Adjusts the randomness of text generation. Higher values produce more diverse outputs. Default: `1.0`.

### ✨ Novelty checking
DrugGENIUS enables comparison of generated ligands against a patent molecule database using Morgan fingerprint-based Tanimoto similarity. For each ligand, a CSV report is generated, listing patent molecule IDs with similarity scores above a specified threshold.

Before using, please download reference database supported by FPSim2 to the root directory of this project. Here we use [SureChEMBL](https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data/latest/fpsim2_fingerprints.h5).

Use `check_novelty.py`
- Common arguments
  - `-i` | `--input`: Path of input directory containing *_prop.json files. Default: `./result/generated_ligands/`  
  - `-t` | `--threshold`: Tanimoto similarity threshold. Default: `0.6`
    > The lower the threshold is, the more patent molecules will be searched, and the slower the search speed will be. 
  - `-f` | `--fp_database`: Path of FPSim2 reference database. Default: `./fpsim2_fingerprints.h5`. 
  - `-d` | `--device`: Device to use. Default: `cpu`. 
    > GPU is also supported, just set to `cuda`. In a multi-GPU environment, specify the order of GPU to be used, such as `cuda:1`.
  - `--proc`: Number of parallel processes. Default value is the total number of CPU cores on the device. Not avaliable when `--device` is set to `cuda`.

### 🗃 Clustering
When working with a large number of generated ligands, clustering based on molecular fingerprints is a good strategy to identify candidates for further study. DrugGENIUS supports filtering molecules based on specified property criteria, followed by dimensionality reduction and clustering using t-SNE and the Louvain algorithm. The process produces both a detailed report table and a visualized clustering plot.

Use `clutser.py`
- Common arguments
  - `-i` | `--input`: Path of input directory containing *.sdf and *_prop.json files. Default: `./result/generated_ligands/`
  - `-o` | `--output`: Path of output directory for report files. Default: `./result/cluster_report/`
  - `-k` | `--k_neighbors`: Number of nearest neighbors to use in clustering. Default: `10`. 
  - `-f` | `--filter`: Path of filter config file. Default: `./filter_clustering.yaml`. 
    > Filtering can be applied during clustering process to reduce calculation. To customize the filtering criteria, modify the `filter_clustering.yaml` file.
  - `--no_cache`: Force refreshing the cache of fps and dimensionality reduction results.
    > Calculating fingerprints and dimensionality reduction may take a long time. By default, already calculated fingerprints and t-SNE embeddings will not be recalculated in a new run. To force a cache refresh, enable this argument.
  - `--proc`: Number of dimensionality reduction and clustering processes. Default value is the total number of CPU cores on the device.
  - `--seed`: Random seed for dimensionality reduction and clustering. Default: `42`

### 🔩 Docking
After above steps, users can identify molecules of interest according to their research goals and filtering criteria. DrugGENIUS also supports batch molecular docking and provides docking scores for each ligand. To perform docking, users need to prepare a hash list of selected molecules—this is a plain text file with one hash per line. Alternatively, candidate molecules can be selected directly from the clustering report.

**TODO**

### 📇 About the filter
| Property             | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| MolWt                | Molecular weight (in Daltons). Filters molecules based on their total mass. |
| HeavyAtomCount       | Number of heavy (non-hydrogen) atoms. Controls molecular size and complexity. |
| LogP                 | Partition coefficient (logP), indicating hydrophobicity.                    |
| QED                  | Quantitative Estimate of Drug-likeness. Higher values suggest better drug-like properties. |
| SA                   | Synthetic Accessibility score. Lower values indicate easier synthesis.      |
| TPSA                 | Topological Polar Surface Area. Related to molecule's ability to permeate cells. |
| HBD                  | Number of hydrogen bond donors. Affects solubility and binding.             |
| HBA                  | Number of hydrogen bond acceptors. Influences molecule's polarity and interaction. |
| RotatableBonds       | Number of rotatable bonds. Controls molecular flexibility.                  |
| NumAromaticRings     | Number of aromatic rings in the molecule. Often associated with drug-like features. |
| FractionCSP3         | Fraction of sp3-hybridized carbon atoms. Indicates 3D character of molecules. |
| FormalCharge         | Net formal charge of the molecule. Used to filter highly charged compounds. |
| NumAliphaticRings    | Number of non-aromatic (aliphatic) rings.                                   |
| Lipinski             | Whether to apply Lipinski's Rule of Five for drug-likeness.                 |

## 📑 Cite this work
**TODO**

## ⚖ License
This project is licensed under [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.html).

The birth of DrugGENIUS is inseparable from the following excellent open-source projects:
- [PyTorch](https://pytorch.org/)
- [transformers](https://github.com/huggingface/transformers)
- [RDKit](https://www.rdkit.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [tqdm](https://tqdm.github.io/)
- [networkx](https://networkx.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [python-louvain](https://github.com/taynaud/python-louvain)
- [FPSim2](https://github.com/chembl/FPSim2)
  
*Open source leads the world to a brighter future!*

## 📝 Reference
[1] Y. Li, C. Gao, X. Song, X. Wang, Y. Xu, and S. Han, “DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins,” Jun. 30, 2023, bioRxiv. doi: 10.1101/2023.06.29.543848.