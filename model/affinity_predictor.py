from typing import Union, List
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from model.transformerCPI import network, featurizer


class AbstractAffinityPredictorModel(ABC):
    @abstractmethod
    def __init__(self, device: str, **kwargs):
        pass

    @abstractmethod
    def predict(self, protein_list, smiles_list, **kwargs) -> List[float]:
        pass

class TransformerCPI2(AbstractAffinityPredictorModel):
    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = network.Predictor(device=device)
        self.model.eval()

    def pack(self, atoms, adjs, proteins):
        batch_size = len(atoms)
        atom_dim = atoms[0].shape[1]

        atom_lens = [a.shape[0] + 1 for a in atoms]  # +1 for padding node
        protein_lens = [p.shape[0] for p in proteins]
        max_atom_len = max(atom_lens)
        max_protein_len = max(protein_lens)

        atoms_padded = torch.zeros((batch_size, max_atom_len, atom_dim), device=self.device)
        adjs_padded = torch.zeros((batch_size, max_atom_len, max_atom_len), device=self.device)
        proteins_padded = torch.zeros((batch_size, max_protein_len), dtype=torch.int64, device=self.device)

        for i, (atom, adj, protein) in enumerate(zip(atoms, adjs, proteins)):
            a_len = atom.shape[0]
            atoms_padded[i, 1:a_len+1, :] = torch.FloatTensor(atom).to(self.device)
            adj = torch.FloatTensor(adj).to(self.device) + torch.eye(a_len, device=self.device)
            adjs_padded[i, 1:a_len+1, 1:a_len+1] = adj
            adjs_padded[i, 0, :] = 1
            adjs_padded[i, :, 0] = 1
            proteins_padded[i, :protein.shape[0]] = torch.LongTensor(protein).to(self.device)

        return atoms_padded, adjs_padded, proteins_padded, atom_lens, protein_lens

    def predict(self, protein_list, smiles_list, batch_size=32):
        scores = []
        with torch.no_grad():
            for i in range(0, len(smiles_list), batch_size):
                batch_smiles = smiles_list[i:i+batch_size]
                batch_proteins = protein_list[i:i+batch_size]

                compounds, adjacencies, proteins = featurizer.get_featurizer(batch_smiles, batch_proteins)
                atoms_pad, adjs_pad, prots_pad, atom_lens, prot_lens = self.pack(compounds, adjacencies, proteins)

                logits = self.model.forward(atoms_pad, adjs_pad, prots_pad, atom_lens, prot_lens)
                probs = F.softmax(logits, dim=1)[:, 1]
                scores.extend(probs.cpu().numpy().tolist())

        return scores

if __name__ == "__main__":
    protein_list = [
        "MPHSSLHPSIPCPRGHGAQKAALVLLSACLVTLWGLGEPPEHTLRYLVLHLA",
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYN",
        "MSDSEVNQEAKPEVKPEVKPETHINLKPVKFLE"
    ]

    smiles_list = [
        "CS(=O)(C1=NN=C(S1)CN2C3CCC2C=C(C4=CC=CC=C4)C3)=O",
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CC1=C(C(=O)NC(=O)N1)N"
    ]

    tester = TransformerCPI2(device="cuda:0")
    scores = tester.predict(protein_list, smiles_list)

    for i, (sm, pr, score) in enumerate(zip(smiles_list, protein_list, scores)):
        print(f"[{i}] SMILES: {sm[:30]}...  | Protein: {pr[:15]}...  =>  Score: {score:.4f}")