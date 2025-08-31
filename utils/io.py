import json
import yaml

def save_json(j: dict, path: str):
    with open(path, 'w') as f:
        json.dump(j, f, indent=4)

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def load_yaml(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def parse_extra_args(extra_args):
    extra_dict = {}
    for i in range(0, len(extra_args), 2):
        key = extra_args[i].lstrip('-')
        value = extra_args[i + 1] if i + 1 < len(extra_args) else None
        extra_dict[key] = value
    return extra_dict

def read_fasta_file(file_path):
    with open(file_path, 'r') as f:
        sequence = []

        for line in f:
            line = line.strip()
            if not line.startswith('>'):
                sequence.append(line)

        protein_sequence = ''.join(sequence)
    return protein_sequence

def read_docking_result(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                vina_score = float(line.split()[3])
                break
    return vina_score

def about(mode, args):
    print(f"""
    ____                   _____________   ________  _______
   / __ \_______  ______ _/ ____/ ____/ | / /  _/ / / / ___/
  / / / / ___/ / / / __ `/ / __/ __/ /  |/ // // / / /\__ \ 
 / /_/ / /  / /_/ / /_/ / /_/ / /___/ /|  // // /_/ /___/ / 
/_____/_/   \__,_/\__, /\____/_____/_/ |_/___/\____//____/  
                 /____/                                     
 An All-in-One Framework for Sequence-based Ligand Design
   Licensed under GNU GPLv3     XJTU-WXY @ FISSION Lab
{mode:=^60}""")
    print(f"Parameters: {args}")