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
Currently only limited functions are implemented, It may contain bugs and should be used at your own risk.
## 🚩 Introduction
Sequence-based deep learning models for ligand design have recently garnered increasing interests in the research community. These end-to-end approaches, which generate ligand SMILES directly from protein amino acid sequences, not only enable access to a broader and more readily available training dataset but also allow the model to move beyond the constraints of binding pocket structures and leverage the richer information embedded in the primary sequence.

This project aims to provide an all-in-one, cross-platform, and efficient framework for sequence-based ligand design, streamlining the complex process into a series of simple and user-friendly steps—such as ligand molecule generation, property evaluation, filtering, novelty checking, and docking.

Currently, the project includes an optimized implementation of the [DrugGPT](https://github.com/LIYUESEN/druggpt) [1] model. By decoupling the generation and post-processing stages and eliminating the inefficient single-threaded design of the original version, the generation speed has been improved by 7–10×. Additionally, the framework introduces a set of practical tools for molecule filtering based on physicochemical properties. An efficient implementation of the sequence-based affinity prediction model [transformerCPI2.0](https://github.com/lifanchen-simm/transformerCPI2.0) [2] is also included to efficiently and accurately predict affinity. Support for more models is planned in future updates.

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
### Install PyTorch
Please also install the appropriate version of [PyTorch](https://pytorch.org/) according to your platform.

**Example:**
```shell
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```
## 🗝 Usage
### ⚗ Generate new ligands
Unlike the original implementation of DrugGPT, DrugGENIUS can automatically detect whether the input text is a raw sequence or a path to a FASTA file. It decouples molecule generation and post-processing into two parallel processes, using a shared cache queue to transfer generated molecules. Post-processing is also optimized with multi-threading, enabling simultaneous molecular property calculation, filtering, and energy minimization—significantly improving overall performance and throughput.

This step will generate ligand *.sdf files and corresponding *.json files containing molecular physicochemical properties in output directory. Both files are named using the hash value of the SMILES string.

On the first run, the required model will be automatically downloaded from HuggingFace. Please ensure a stable internet connection and be patient during the download process.

**Example:**
```shell
python generate.py -i BCL2L11.fasta
```
- Common arguments
  - `-i` | `--input`: Path of FASTA file or amino acid string of target protein. 
  - `-o` | `--output`: Path of directory for the project. Default: `./result/`
  - `-m` | `--model`: Model to use for generation. Currently only supports: `DrugGPT`
  - `-n` | `--total_num`: Total number of ligands to generate. Default: `10000`
  - `-f` | `--filter`: Path of filter config file. Default: `./filter_generate.yaml`. 
    > Filtering can be applied directly during generation process so that all generated ligands meet the criteria. To customize the filtering criteria, modify the `filter_generate.yaml` file
  - `-d` | `--device`: Device to use. Default: `cuda`. 
    > In a multi-GPU environment, specify the order of GPU to be used, such as `cuda:1`. Multi-GPU distributed inference will be supported in future versions.
  - `--threads`: Number of post-processing threads. Default value is the total number of CPU cores on the device.
  - `--em_iters`: Max number of iterations for energy minimization. Default: `10000`
  - `--queue_len`: Maximum length of the cache queue. Default: `20`. 
  - `--init_seed`: The initial random seed for result reproducibility. Each generated batch will increase the seed by one. If not specified, current timestamp will be used as random seed for each batch.
  - `--record_raw_output`: For research purposes, record the raw SMILES string output by the ligand generation model in the json file.
  
- Arguments for `DrugGPT` model
  - `--batch_size`: The number of molecules to try to generate in each batch. Default: `16`
    > DrugGPT supports parallelization to speed up the generation, but this means more RAM usage. Adjust this value according to the actual total amount of RAM.
  - `--ligand_prompt`: The SMILES of basic scaffold. Default: `""`
    > DrugGPT supports using a predefined scaffold as the basic structure for ligand generation. See its paper for details.
  - `--top_k`: The number of highest probability tokens to be considered for top-k sampling. Default: `40`.
  - `--top_p`: The cumulative probability threshold (0.0 - 1.0) for top-p (nucleus) sampling. It defines the minimum subset of tokens to consider for random sampling. Default: `0.9`.
  - `--temp`: Adjusts the randomness of text generation. Higher values produce more diverse outputs. Default: `1.0`.

### 🗳 Start from existed ligands
Sometimes, we already have a list of candidate ligands and are only interested in subsequent screening, affinity prediction, docking, etc. In this case, we need to prepare a text file with each line containing the SMILES of a ligand, and use `from_existed.py` to extract properties and perform energy minimization for subsequent processing by DrugGENIUS.

**Example:**
```shell
python from_existed.py -i example_exsited_ligands.txt
```
- `-i` | `--input`: Path of a txt file containing existed SMILES, one per line.
- `-o` | `--output`: Path of directory for the project. Default: `./result/`
- `-f` | `--filter`: Path of filter config file. Default: `./filter_generate.yaml`. 
  > Similar to the generation process, filtering can also be applied when starting from an exsited ligand list. To customize the filtering criteria, modify the `filter_generate.yaml` file
- `--threads`: Number of post-processing threads. Default value is the total number of CPU cores on the device.
- `--em_iters`: Max number of iterations for energy minimization. Default: `10000`

### 📇 Merge report file
Before subsequent steps, a report file needs to be generated from all candidate ligands. The file `ligand_report.csv` stores the metadata for each ligand. Subsequent steps will be based on this file, and the results will be stored as a new column.

**Example:**
```shell
python report.py
```
- `-i` | `--input`: Path of directory for the project. Default: `./result/`
- `--threads`: Number of reading threads. Default value is the total number of CPU cores on the device.

### ✨ Novelty checking
DrugGENIUS enables comparison of generated ligands against a patent molecule database using Morgan fingerprint-based Tanimoto similarity. For each ligand, a CSV report is generated, listing patent molecule IDs with similarity scores above a specified threshold.

Before using, please download reference database supported by FPSim2 to the root directory of this project. Here we use [SureChEMBL](https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data/latest/fpsim2_fingerprints.h5). Download the `fpsim2_fingerprints.h5` file and put it into root directory of DrugGENIUS.

**Example:**
```shell
python check_novelty.py
```
- `-i` | `--input`: Path of directory for the project. Default: `./result/`
- `-t` | `--threshold`: Tanimoto similarity threshold. Default: `0.6`
  > The lower the threshold is, the more patent molecules will be searched, and the slower the search speed will be. 
- `-f` | `--fp_database`: Path of FPSim2 reference database. Default: `./fpsim2_fingerprints.h5`. 
- `-d` | `--device`: Device to use. Default: `cpu`. 
  > GPU is also supported, just set to `cuda`. In a multi-GPU environment, specify the order of GPU to be used, such as `cuda:1`.
- `--threads`: Number of threads. Default value is the total number of CPU cores on the device. Not avaliable when `--device` is set to `cuda`.

When finished, the `ligand_report.csv` file will have a new column `MaxSimilarity` indicating the maximum similarity between the ligands and the patented molecule database, and a column `MostSimilarMolID` indicating the ID of the most similar patented molecule.

### 📊 Affinity Prediction
Compared with traditional molecular docking, a number of affinity prediction methods based on deep learning have emerged, which can efficiently and accurately predict affinity scores. DrugGENIUS currently includes an efficient implementation of transformerCPI2.0 and will support more models in the future.

**Example:**
```shell
python predict_affinity.py
```
- `-i` | `--input`: Path of directory for the project. Default: `./result/`
- `-m` | `--model`: Model to use for prediction. Currently only supports: `TransformerCPI2`
- `-b` | `--batch_size`: Number of molecules predicted per batch. Default: 1024. 
  > A larger batch size results in a larger memory usage.
- `-d` | `--device`: Device to use. Default: `cuda`. 
  > In a multi-GPU environment, specify the order of GPU to be used, such as `cuda:1`.

When finished, the `ligand_report.csv` file will have a new column `PredictedAffinity` indicating the predicted affinity score.

### 🗃 Clustering
When working with a large number of generated ligands, clustering based on molecular fingerprints is a good strategy to identify candidates for further study. DrugGENIUS supports filtering molecules based on specified property criteria, followed by dimensionality reduction and clustering using t-SNE and the Louvain algorithm.

**Example:**
```shell
python cluster.py
```
- `-i` | `--input`: Path of directory for the project. Default: `./result/`
- `-k` | `--k_neighbors`: Number of nearest neighbors to use in clustering. Default: `10`. 
- `--no_cache`: Force refreshing the cache of fps and dimensionality reduction results.
  > Calculating fingerprints and dimensionality reduction may take a long time. By default, already calculated fingerprints and t-SNE embeddings will not be recalculated in a new run. To force a cache refresh, enable this argument.
- `--threads`: Number of threads for dimensionality reduction and clustering. Default value is the total number of CPU cores on the device.
- `--seed`: Random seed for dimensionality reduction and clustering. Default: `42`

When finished, the `ligand_report.csv` file will have a new column `Cluster` indicating the ID of clusters, and a `clustering.pdf` figure will be generated to show the result. 

### 📈 Visualization
DrugGENIUS provides simple visualization tools to show the results of the above steps.

**Example:**
```shell
python visualization.py
```
- `-i` | `--input`: Path of directory for the project. Default: `./result/`
- `-c` | `--cmap`: Colormap name for the plots. Default: `plasma`. 
- `-f` | `--format`: Format for the plots. Default: `png`. 
- `--dpi`: DPI for the plots. Default: `300`. 

This will generate a series of t-SNE plots with color mapping to show the distribution of various properties in the molecules.

### 🔩 Docking
DrugGENIUS supports preprocessing ligands and running AutoDock Vina in a multiprocess mode to fully leverage multi-core CPUs. Before getting started, make sure [AutoDock Vina](https://vina.scripps.edu/downloads/) is properly installed and that the `vina` command can be executed from the command line.

Molecular docking requires the 3D structure of the target protein and a docking box. Please prepare the target protein in PDB format along with either docking box parameters or a docking box file. Then, use the [mk_prepare_receptor.py](https://meeko.readthedocs.io/en/release-doc/rec_cli_options.html) command from the meeko library to preprocess the target protein as needed.

**Example:**
```shell
mk_prepare_receptor -i your_target.pdb -o your_target --box_enveloping docking_box.sdf -p -v
```

This step will generate the required files for docking: `your_target.pdbqt` and `your_target.box.txt`.

**Example:**
```shell
python docking.py -p your_target.pdbqt -b your_target.box.txt
```
- `-p` | `--protein`: Path of the pdbqt file of target protein.
- `-b` | `--box`: Path of the txt file of grid box.
- `-i` | `--input`: Path of directory for the project. Default: `./result/`
- `-f` | `--filter`: Path of filter config file. Default: `./filter_docking.yaml`. 
  > Molecular docking is time-consuming. Usually, we only dock molecules with good properties. Pre-docking filter can be defined in the `filter_docking.yaml` file.
- `-e` | `--exhaustiveness`: Exhaustiveness for Autodock Vina. Default: `8`
- `--threads`: Number of preprocessing and docking threads. Default: `4`.
  > AutoDock Vina itself has limited multi-core optimization. If too many processes are opened, performance will be reduced and a large amount of RAM will be consumed. Please adjust this value so that the CPU usage is just full.

Once docking is complete, pose results will be generated in the `docking_result` subdirectory of your project, with each file named according to the ligand’s hash value. A new `ligand_report_docking.csv` file will be created, and the `VinaScore` column records the best Vina score for each ligand.

### 🎛 About the filter
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
| *GenerationFrequency | The number of times a molecule is repeatedly generated. A higher value indicates that the model favors this molecule.|
| *MaxSimilarity       | The maximum similarity to molecules in the patented database. During the novelty checking, only values above threshold are recorded; values below the threshold are treated as 0.|
| *PredictedAffinity   | The predicted affinity score.|
| *Cluster             | Cluster IDs. If set to null, no filtering will be applied. If set to a list enclosed in square brackets and separated by commas (e.g. `[0,1,3]`), only the specified clusters will be selected.|

*Fields marked with * are only available during the docking step.*

## 📑 Cite this work
If you use DrugGENIUS in your work, please cite: 

[![DOI](https://zenodo.org/badge/1012404941.svg)](https://doi.org/10.5281/zenodo.16992818)

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
- [meeko](https://github.com/forlilab/Meeko)
  
*Open source leads the world to a brighter future!*

## 📝 Reference
[1] Y. Li, C. Gao, X. Song, X. Wang, Y. Xu, S. Han, DrugGPT: A GPT-based Strategy for Designing Potential Ligands Targeting Specific Proteins, (2023) 2023.06.29.543848. https://doi.org/10.1101/2023.06.29.543848.

[2] L. Chen, Z. Fan, J. Chang, R. Yang, H. Hou, H. Guo, Y. Zhang, T. Yang, C. Zhou, Q. Sui, Z. Chen, C. Zheng, X. Hao, K. Zhang, R. Cui, Z. Zhang, H. Ma, Y. Ding, N. Zhang, X. Lu, X. Luo, H. Jiang, S. Zhang, M. Zheng, Sequence-based drug design as a concept in computational drug design, Nat Commun 14 (2023) 4217. https://doi.org/10.1038/s41467-023-39856-w.
