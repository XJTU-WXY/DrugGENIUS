from typing import List, Tuple

import torch
import numpy as np
from rdkit import Chem
from tape import TAPETokenizer

num_atom_feat = 34
SYMBOLS = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']
DEGREES = [0, 1, 2, 3, 4, 5, 6]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    'other'
]
NUM_HS = [0, 1, 2, 3, 4]
CHIRALITY = ['R', 'S']

tokenizer = TAPETokenizer(vocab='iupac')

def one_hot_index(x, allowable_set, allow_unknown=True):
    if x not in allowable_set:
        if allow_unknown:
            x = allowable_set[-1]
        else:
            raise ValueError(f"input {x} not in allowable set {allowable_set}")
    idx = allowable_set.index(x)
    return np.eye(len(allowable_set), dtype=bool)[idx]


def atom_features(atom,explicit_H=False,use_chirality=True):
    """
    Generate atom-level feature vector (length 34).
    Features include symbol, degree, charge, hybridization, aromaticity, hydrogen count, chirality.
    """
    features = [
        one_hot_index(atom.GetSymbol(), SYMBOLS),  # 10
        one_hot_index(atom.GetDegree(), DEGREES, allow_unknown=False),  # 7
        np.array([atom.GetFormalCharge(), atom.GetNumRadicalElectrons()], dtype=np.float32),  # 2
        one_hot_index(atom.GetHybridization(), HYBRIDIZATIONS),  # 6
        np.array([atom.GetIsAromatic()], dtype=np.float32)  # 1
    ]

    if not explicit_H:
        features.append(one_hot_index(atom.GetTotalNumHs(), NUM_HS))  # 5

    if use_chirality:
        try:
            chirality_feat = one_hot_index(atom.GetProp('_CIPCode'), CHIRALITY)
        except KeyError:
            chirality_feat = np.array([0., 0.], dtype=np.float32)

        features.append(chirality_feat)  # 2
        features.append(np.array([atom.HasProp('_ChiralityPossible')], dtype=np.float32))  # 1

    return np.concatenate(features)

def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"SMILES cannot be parsed: {smiles}")
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    atom_feats = np.array(atom_feats, dtype=np.float32)

    adj_matrix = Chem.GetAdjacencyMatrix(mol)
    return atom_feats, adj_matrix

def get_featurizer(smiles_list: List[str], sequence: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    compounds, adjacencies, proteins = [], [], []

    encoded_seq = tokenizer.encode(sequence)
    protein_tensor = torch.tensor(encoded_seq, dtype=torch.long).numpy()

    for smiles in smiles_list:
        atom_feat, adj = mol_features(smiles)
        compounds.append(atom_feat.astype(np.float32))
        adjacencies.append(adj.astype(np.float32))
        proteins.append(protein_tensor.astype(np.int64))

    return compounds,adjacencies,proteins

if __name__ == "__main__":
    f = get_featurizer(["CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O"], ["MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHTLRYLVLHLA"])
    print(f)
