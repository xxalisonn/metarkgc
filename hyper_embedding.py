import torch
import torch.nn as nn


class H_Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(H_Embedding, self).__init__()
        self.device = parameter['device']
        self.es = parameter['embed_dim']
        self.rel2id = dataset['rel2id']

        num_rel = len(self.rel2id)
        self.norm_vector = nn.Embedding(num_rel, self.es)

        nn.init.xavier_uniform_(self.norm_vector.weight)

    def forward(self, triples):
        # triple:[batchsize,few,3],形式为tuple
        rel_emb = [[[self.rel2id[t[1]]] for t in batch] for batch in triples]
        # rel_emb: [batchsize,few,1],形式为tensor
        rel_emb = torch.LongTensor(rel_emb).to(self.device)
        return self.norm_vector(rel_emb)
