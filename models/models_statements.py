import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Dict
from .gnn_encoder import StarEEncoder, StarEBase
from utils.utils_gcn import get_param
from models.time_encoder import TimeEncode



class StarE_Transformer(StarEEncoder):
    model_name = 'StarE_Transformer_Statement'

    def __init__(self, kg_graph_repr: Dict[str, np.ndarray], config: dict, id2e: tuple = None):
        if id2e is not None:
            super(self.__class__, self).__init__(kg_graph_repr, config, id2e[1])
        else:
            super(self.__class__, self).__init__(kg_graph_repr, config)

        self.model_name = 'StarE_Transformer_Statement'

        self.hid_drop2 = config['STAREARGS']['HID_DROP2']
        self.feat_drop = config['STAREARGS']['FEAT_DROP']
        self.num_transformer_layers = config['STAREARGS']['T_LAYERS']
        self.num_heads = config['STAREARGS']['T_N_HEADS']
        self.num_hidden = config['STAREARGS']['T_HIDDEN']
        self.d_model = config['EMBEDDING_DIM']
        self.positional = config['STAREARGS']['POSITIONAL']
        self.p_option = config['STAREARGS']['POS_OPTION']
        self.pooling = config['STAREARGS']['POOLING']  # min / avg / concat
        self.embed_statics = config['SAMPLER_W_STATICS']
        self.embed_qualifiers = config['SAMPLER_W_QUALIFIERS']

        self.hidden_drop = torch.nn.Dropout(self.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.feat_drop)

        encoder_layers = TransformerEncoderLayer(self.d_model, self.num_heads, self.num_hidden, config['STAREARGS']['HID_DROP2'])
        self.encoder = TransformerEncoder(encoder_layers, config['STAREARGS']['T_LAYERS'])
        self.position_embeddings = nn.Embedding(config['MAX_QPAIRS'] - 1, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.emb_dim)

        if self.pooling == "concat":
            self.flat_sz = self.emb_dim * (config['MAX_QPAIRS'] - 1)
            self.fc = torch.nn.Linear(self.flat_sz, self.emb_dim)
        else:
            self.fc = torch.nn.Linear(self.emb_dim, self.emb_dim)

    def concat(self, e1_embed, rel_embed, quals=None, statics=None):
        e1_embed = e1_embed.view(-1, 1, self.emb_dim)
        rel_embed = rel_embed.view(-1, 1, self.emb_dim)
        """
            arrange quals in the conve format with shape [bs, num_qual_pairs, emb_dim]
            num_qual_pairs is 2 * (any qual tensor shape[1])
            for each datum in bs the order will be 
                rel1, emb
                en1, emb
                rel2, emb
                en2, emb
        """
        if statics is not None:
            if quals is not None:
                stack_inp = torch.cat([e1_embed, rel_embed, quals, statics], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
            else:
                stack_inp = torch.cat([e1_embed, rel_embed, statics], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        else:
            if quals is not None:
                stack_inp = torch.cat([e1_embed, rel_embed, quals], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
            else:
                stack_inp = torch.cat([e1_embed, rel_embed], 1).transpose(1, 0)  # [2 + num_qual_pairs, bs, emb_dim]
        return stack_inp

    def forward(self, sub, rel, time, quals=None):
        '''


        :param sub: bs
        :param rel: bs
        :param quals: bs*(sl-2) # bs*14
        :return:


        '''
        sub_emb, rel_emb, qual_emb, static_emb, all_ent, mask = \
            self.forward_base(sub, rel, time, self.hidden_drop, self.feature_drop, quals, True)

        # bs*emb_dim , ......, bs*6*emb_dim

        stk_inp = self.concat(sub_emb, rel_emb, qual_emb, static_emb)

        if self.positional:
            positions = torch.arange(stk_inp.shape[0], dtype=torch.long, device=self.device).repeat(stk_inp.shape[1], 1)
            pos_embeddings = self.position_embeddings(positions).transpose(1, 0)
            stk_inp = stk_inp + pos_embeddings

        x = self.encoder(stk_inp, src_key_padding_mask=mask)

        if self.pooling == 'concat':
            x = x.transpose(1, 0).reshape(-1, self.flat_sz)
        elif self.pooling == "avg":
            x = torch.mean(x, dim=0)
        elif self.pooling == "min":
            x, _ = torch.min(x, dim=0)

        x = self.fc(x)

        if self.embed_statics:
            all_ent = torch.mean(all_ent, dim=1)
            x = torch.mm(x, all_ent.transpose(1, 0))
        else:
            x = torch.mm(x, all_ent.transpose(1, 0))
        score = torch.sigmoid(x)
        return score