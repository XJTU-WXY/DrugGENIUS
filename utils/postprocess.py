from typing import Union

import os
import hashlib

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Contrib.SA_Score import sascorer
from rdkit.rdBase import BlockLogs

from utils.io import *

block = BlockLogs()

def calc_mol_properties(mol: Chem.Mol):
    mol = Chem.AddHs(mol)

    return {
        "MolWt": Descriptors.MolWt(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
        "LogP": Descriptors.MolLogP(mol),
        "QED": QED.qed(mol),
        "SA": sascorer.calculateScore(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumAromaticRings": Lipinski.NumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "FormalCharge": Chem.GetFormalCharge(mol),
        "NumAliphaticRings": Lipinski.NumAliphaticRings(mol)
    }


def filter_mol_by_prop(mol_prop: dict, filter_dict: dict):
    if filter_dict.get("Properties", None) is not None:
        for key, value in filter_dict["Properties"].items():
            if key in mol_prop:
                min_value = value["Min"]
                max_value = value["Max"]
                if min_value is not None and mol_prop[key] < min_value:
                    return False
                if max_value is not None and mol_prop[key] > max_value:
                    return False
    if filter_dict.get("Lipinski", False) and not passes_lipinski(mol_prop):
        return False
    return True

def passes_lipinski(mol_prop):
    conditions = [
        mol_prop["MolWt"] <= 500,
        mol_prop["LogP"] <= 5,
        mol_prop["HBD"] <= 5,
        mol_prop["HBA"] <= 10
    ]
    return sum(conditions) >= 3

def smiles_to_mol(smiles: str) -> Union[Chem.Mol, None]:
    block = BlockLogs()
    mol = Chem.MolFromSmiles(smiles)
    return mol

def mol_to_minimized_sdf(mol: Chem.Mol, output_path: str, em_iters: int=10000) -> bool:
    try:
        mol = Chem.AddHs(mol)

        result = AllChem.EmbedMolecule(mol)
        if result != 0:
            return False

        if not AllChem.MMFFHasAllMoleculeParams(mol):
            return False
        result = AllChem.MMFFOptimizeMolecule(mol, maxIters=em_iters)

        if result != 0:
            return False

        writer = Chem.SDWriter(output_path)
        writer.write(mol)
        writer.close()

        return True
    except:
        return False

def smiles_to_sdf(smiles: str,
                  output_dir: str,
                  filter_dict: dict,
                  em_iters: int=10000):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

    smiles_hash = hashlib.sha1(canonical_smiles.encode()).hexdigest()
    sdf_path = os.path.join(output_dir, f"{smiles_hash}.sdf")
    json_path = os.path.join(output_dir, f"{smiles_hash}.json")

    if not os.path.exists(sdf_path):
        mol_prop = calc_mol_properties(mol)
        if not filter_mol_by_prop(mol_prop, filter_dict):
            return None
        if mol_to_minimized_sdf(mol, sdf_path, em_iters=em_iters):
            meta_data = {"Hash": smiles_hash, "SMILES": canonical_smiles, "GenerationFrequency": 1}
            meta_data.update(mol_prop)
            save_json(meta_data, json_path)
            return 0
        else:
            return None
    else:
        return json_path