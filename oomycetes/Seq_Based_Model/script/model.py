import torch
import torch.nn as nn
from einops import rearrange

import esm
from DeepSecE.module import TransformerLayer, MLPLayer

class EffectorTransformer(nn.Module):

    def __init__(self, emb_dim, repr_layer, num_layers, heads,
                 hid_dim=256, dropout_rate=0.4, num_classes=2, attn_dropout=0.05, return_embedding=False, return_attn=False):

        super().__init__()
        self.pretrained_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S() 
        self.padding_idx = alphabet.padding_idx 
        self.dim = hid_dim
        self.repr_layer = repr_layer
        self.num_layers = num_layers

        self.conv = nn.Conv1d(emb_dim, hid_dim, 1, 1, bias=False) 
        self.layers = nn.ModuleList(
            [
                TransformerLayer(hid_dim, heads, dropout_rate, attn_dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.clf = nn.Linear(hid_dim, num_classes)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.return_embedding = return_embedding
        self.return_attn = return_attn

    def forward(self, strs, toks):

        toks = toks[:, :1022]
        padding_mask = (toks != self.padding_idx)[:, 1:-1] 

        out = self.pretrained_model(
            toks, repr_layers=[self.repr_layer], return_contacts=False) 
        x = out["representations"][self.repr_layer][:, 1:-1, :]  
        x = x * padding_mask.unsqueeze(-1).type_as(x)

        x = rearrange(x, 'b n d -> b d n')
        x = self.conv(x)  
        x = rearrange(x, 'b d n -> b n d')

        batch = toks.shape[0]
        
        for layer in self.layers:
            x, attn = layer(
                x, mask=padding_mask.unsqueeze(1).unsqueeze(2)
            )

        out = torch.cat([x[i, :len(strs[i]) + 1].mean(0).unsqueeze(0)
                        for i in range(batch)], dim=0) 
        if self.return_embedding:
            return out
        else:
            logits = self.clf(out)
            if self.return_attn:
                return logits, attn
            else:
                return logits
