from typing import Union, List
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel


class AbstractModel(ABC):
    @abstractmethod
    def __init__(self, device: str, **kwargs):
        pass

    @abstractmethod
    def generate(self, target_seq, **kwargs) -> List[str]:
        pass

class DrugGPT(AbstractModel):
    def __init__(self, device: str="cpu", local_dir: Union[None, str]=None):
        self.tokenizer = AutoTokenizer.from_pretrained('liyuesen/druggpt', cache_dir=local_dir)
        self.model = GPT2LMHeadModel.from_pretrained("liyuesen/druggpt", cache_dir=local_dir)

        self.device = torch.device(device)
        self.model.to(self.device)

    def generate(self,
                 target_seq: str,
                 ligand_prompt: str="",
                 batch_size: int=16,
                 top_k: int=40,
                 top_p: float=0.9,
                 temp: float=1.0) -> List[str]:
        prompt = f"<|startoftext|><P>{target_seq}<L>{ligand_prompt}"
        prompt = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.device)
        attention_mask = prompt.ne(self.tokenizer.pad_token_id).float()
        outputs = self.model.generate(
            prompt,
            do_sample=True,
            max_length=1024,
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temp),
            num_return_sequences=int(batch_size),
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ligands = [self.tokenizer.decode(_, skip_special_tokens=True).split('<L>')[1] for _ in outputs]
        return generated_ligands


if __name__ == "__main__":
    druggpt = DrugGPT(device="cuda:0")
    ligands = druggpt.generate("MAKQPSDVSSECDREGRQLQPAERPPQLRPGAPTSLQTEPQGNPEGNHGGEGDSCPHGSPQGPLAPPASPGPFATRSPLFIFMRRSSLLSRSSSGYFSFDTDRSPAPMSCDKSTQTPSPPCQAFNHYLSAMASMRQAEPADMRPEIWIAQELRRIGDEFNAYYARRVFLNNYQAAEDHPRMVILRLLRYIVRLVWRMH")
    print(ligands)