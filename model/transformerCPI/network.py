import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from tape.models.modeling_bert import ProteinBertModel
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*")

class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, n_layers=3):
        super().__init__()
        self.pretrain = ProteinBertModel.from_pretrained('bert-base')
        self.hid_dim = 768
        self.n_layers = n_layers
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward=self.hid_dim*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.n_layers)

    def forward(self, protein, mask):
        # protein = [batch, seq len]
        # mask = [batch, seq len]  0 for true positions, 1 for mask positions
        input_mask = (mask > 0).float()
        with torch.no_grad():
            protein = self.pretrain(protein,input_mask)[0]
        mask = (mask == 1)
        protein = self.encoder(protein,src_key_padding_mask=mask)
        # protein = [seq len, batch, 768]
        return protein, mask


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self,n_layers=3, dropout=0.1):
        super().__init__()
        self.hid_dim = 768
        self.n_layers = n_layers
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.hid_dim, nhead=8, dim_feedforward=self.hid_dim * 4, dropout=0.1, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.n_layers)
        self.fc_1 = nn.Linear(768, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, hid_dim]
        # src = [protein len, batch, hid_dim] # encoder output
        # trg = [compound len, batch, hid_dim]
        trg_mask = (trg_mask == 1)
        trg = self.decoder(trg, src, tgt_key_padding_mask=trg_mask, memory_key_padding_mask=src_mask)
        # trg = [compound len,batch size, hid dim]
        # trg = [batch, compound len, hid dim]
        x = trg[:,0,:]
        label = F.relu(self.fc_1(x))
        label = self.fc_2(label)
        return label


class Predictor(nn.Module):
    def __init__(self, device, atom_dim=34):
        super().__init__()
        self.device = device
        self.encoder = Encoder().to(self.device)
        self.decoder = Decoder().to(self.device)
        self.fc_1 = nn.Linear(atom_dim, atom_dim)
        self.fc_2 = nn.Linear(atom_dim, 768)

        model_path = hf_hub_download(repo_id="XJTU-WXY/transformerCPI2_repack", filename="transformerCPI2.pth")
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        self.to(device)

    def gcn(self, input, adj):
        # input =[batch,num_node, atom_dim]
        # adj = [batch,num_node, num_node]
        support = self.fc_1(input)
        # support =[batch,num_node,atom_dim]
        output = torch.bmm(adj, support)
        # output = [batch,num_node,atom_dim]
        return output

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.ones((N, compound_max_len),device=self.device)
        protein_mask = torch.ones((N, protein_max_len),device=self.device)
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 0
            protein_mask[i, :protein_num[i]] = 0
        return compound_mask, protein_mask


    def forward(self, compound, adj,  protein, atom_num, protein_num):
        # compound = [batch,atom_num, atom_dim]
        # adj = [batch,atom_num, atom_num]
        # protein = [batch,protein len, 768]
        compound_max_len = compound.shape[1]
        protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)
        compound = self.gcn(compound, adj)
        # compound = [batch size,atom_num, atom_dim]
        compound = F.relu(self.fc_2(compound))
        # compound = [batch, compound len, 768]
        enc_src, src_mask = self.encoder(protein, protein_mask)
        # enc_src = [protein len,batch , hid dim]
        out = self.decoder(compound, enc_src, compound_mask, src_mask)
        # out = [batch size, 2]
        return out
